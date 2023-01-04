'''train model on all META training data file and eval on the valid files'''
import os
import glob
import time
import pickle
import torch
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
# from model import CGRU as Agent
from model import CGRU_v2 as Agent
from model import SimpleContext
from utils import to_np, to_pth, split_video_id, context_to_bound_vec, loss_to_bound_vec, save_ckpt, padded_corr
from utils import EventLabel, TrainValidSplit, DataLoader, Parameters, HumanBondaries
from scipy.stats import pointbiserialr, pearsonr
sns.set(style='white', palette='colorblind', context='talk')

'''e.g.
python train.py --subj_id 99 --lr 1e-3 --update_freq 10 --dim_hidden 16 --dim_context 256 --ctx_wt .5 --penalty_new_context .5 --stickiness .5
sbatch train.sh 99 1e-3 10 16 256 .5 .5 .5
'''
parser = argparse.ArgumentParser()
parser.add_argument('--subj_id', default=99, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--update_freq', default=100, type=int)
parser.add_argument('--dim_hidden', default=16, type=int)
parser.add_argument('--dim_context', default=32, type=int)
parser.add_argument('--ctx_wt', default=.5, type=float)
parser.add_argument('--penalty_new_context', default=.5, type=float)
parser.add_argument('--stickiness', default=.5, type=float)
parser.add_argument('--log_root', default='../log', type=str)
args = parser.parse_args()
print(args)

'''params for the model'''

# training param
subj_id = args.subj_id
lr = args.lr
update_freq = args.update_freq
dim_hidden = args.dim_hidden
dim_context = args.dim_context
ctx_wt = args.ctx_wt
penalty_new_context = args.penalty_new_context
stickiness = args.stickiness
log_root = args.log_root

# # training param
# subj_id = 0
# lr = 1e-3
# update_freq = 10
# # model param
# dim_hidden = 16
# dim_context = 256
# ctx_wt = .5
# # ctx_wt = 0
# penalty_new_context = .5
# stickiness = .5

# set seed
np.random.seed(subj_id)
torch.manual_seed(subj_id)
# init util objects
dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()
hb = HumanBondaries()
p = Parameters(
    dim_hidden = dim_hidden, dim_context = dim_context, ctx_wt = ctx_wt,
    penalty_new_context = penalty_new_context, stickiness = stickiness, lr = lr,
    update_freq = update_freq, subj_id = subj_id, log_root=log_root,
)
# init model
agent = Agent(
    p.dim_input, p.dim_hidden, p.dim_output,
    ctx_wt=p.ctx_wt, context_dim=p.dim_context
)
optimizer = torch.optim.Adam(agent.parameters(), lr=p.lr)
# context management
sc = SimpleContext(p.dim_context, penalty_new_context, stickiness)
c_id, c_vec = sc.init_context()

'''train the model'''

def run_model(event_id_list, p, save_weights=True, learning=True, save_freq=10):
    # prealooc
    loss_by_events = [[] for _ in range(len(event_id_list))]
    log_cid = [[] for _ in range(len(event_id_list))]

    for i, event_id in enumerate(event_id_list):
        # save data for every other k epochs
        if save_weights and i % save_freq == 0:
            save_ckpt(i, p.log_dir, agent, optimizer, sc.to_dict(), verbose=True)
        print(f'Learning event {i} / {len(event_id_list)} - {event_id}')
        t_start = time.time()
        # get data
        X = dl.get_data(event_id)
        T = len(X) - 1
        # prealloc
        log_cid_i = np.zeros(T, )
        # run the model over time
        loss = 0
        h_t = agent.get_init_states()
        for t in tqdm(range(T)):
            # context - full inference
            # pe = agent.try_all_contexts(X[t+1], X[t], h_t, sc.context, sc.prev_cluster_id)
            pe = agent.try_all_contexts(X[t+1], X[t], h_t, sc.context)
            pe[0] = pe[0] + p.penalty_new_context
            pe[sc.prev_cluster_id] = pe[sc.prev_cluster_id] - p.stickiness
            log_cid_i[t], c_vec = sc.assign_context(-pe, get_context_vector=True, verbose=1)
            # forward
            [y_t_hat, h_t], cache = agent.forward(X[t], h_t, to_pth(c_vec))
            # record losses
            loss_it = agent.criterion(torch.squeeze(y_t_hat), X[t+1])
            loss += loss_it
            loss_by_events[i].append(loss_it)
            # update weights for every other t time points
            if learning and t % update_freq == 0:
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        log_cid[i] = log_cid_i
        print('Time elapsed = %.2f sec' % (time.time() - t_start))
    # save the final weights
    if save_weights:
        save_ckpt(len(event_id_list), p.log_dir, agent, optimizer, sc.to_dict(), verbose=True)
    print('done')
    return log_cid, loss_by_events


'''evaluate loss on the validation set'''
log_cid_tr, loss_by_events_tr = run_model(tvs.train_ids, p=p, save_weights=True, learning=True)
log_cid, loss_by_events = run_model(tvs.valid_ids, p=p, save_weights=False, learning=False)


'''plot the data '''
loss_mu_by_events = [torch.stack(loss_event_i).mean() for loss_event_i in loss_by_events]
f, ax = plt.subplots(1,1, figsize=(7,4))
ax.plot(loss_mu_by_events)
ax.set_title('%.3f' % torch.stack(loss_mu_by_events).mean())
ax.set_xlabel('validation video id')
ax.set_ylabel('loss')
sns.despine()
fig_path = os.path.join(p.fig_dir, f'final-loss-by-event.png')
f.savefig(fig_path, dpi=100)


def compute_n_ctx_over_time(log_cid_):
    max_ctx_by_events = [np.max(log_cid_tr_i) for log_cid_tr_i in log_cid_]
    max_ctx_so_far = 0
    for i in range(len(max_ctx_by_events)):
        if max_ctx_by_events[i] > max_ctx_so_far:
            max_ctx_so_far = max_ctx_by_events[i]
        else:
            max_ctx_by_events[i] = max_ctx_so_far
    return max_ctx_by_events

n_ctx_over_time = compute_n_ctx_over_time(log_cid_tr)
f, ax = plt.subplots(1,1, figsize=(5,4))
ax.plot(n_ctx_over_time)
ax.set_xlabel('training video id')
ax.set_ylabel('# of contexts inferred')
sns.despine()
fig_path = os.path.join(p.fig_dir, f'final-n-ctx-over-training.png')
f.savefig(fig_path, dpi=100)


# for i, loss_by_events_i in enumerate(loss_by_events):
#     if i > 3: break
#     f, ax = plt.subplots(1,1, figsize=(10,3))
#     ax.plot(loss_by_events_i)
#     ax.set_title(tvs.valid_ids[i])
#     ax.set_xlabel('time')
#     ax.set_ylabel('loss')
#     event_bound_times, event_bound_vec = evlab.get_bounds(tvs.valid_ids[i])
#     for eb in event_bound_times:
#         ax.axvline(eb, ls='--', color='grey')
#     sns.despine()
#
#
# for i, log_cid_i in enumerate(log_cid):
#     if i > 3: break
#     event_bound_times, event_bound_vec = evlab.get_bounds(tvs.valid_ids[i])
#     f, ax = plt.subplots(1,1, figsize=(10,3))
#     ax.plot(log_cid_i)
#     # ax.set_title('%.3f' % torch.stack(loss_mu_by_events).mean())
#     ax.set_title(tvs.valid_ids[i])
#     ax.set_xlabel('time')
#     ax.set_ylabel('Context')
#
#     for eb in event_bound_times:
#         ax.axvline(eb, ls='--', color='grey')
#     sns.despine()


'''correlation with human boundaries'''

r_crse = np.zeros(len(log_cid),)
r_fine = np.zeros(len(log_cid),)
p_crse = np.zeros(len(log_cid),)
p_fine = np.zeros(len(log_cid),)
for i in range(tvs.n_valid_files):


    model_loss_bound_vec = loss_to_bound_vec(loss_by_events[i])
    model_ctx_bound_vec = context_to_bound_vec(log_cid[i])
    p_b_c = hb.get_bound_prob(tvs.valid_ids[i], 'coarse')
    p_b_f = hb.get_bound_prob(tvs.valid_ids[i], 'fine')
    # r_crse[i], p_crse[i] = padded_corr(to_np(torch.stack(loss_by_events[i])), p_b_c, corr_f=pearsonr, porp=.1)
    # r_fine[i], p_fine[i] = padded_corr(to_np(torch.stack(loss_by_events[i])), p_b_f, corr_f=pearsonr, porp=.1)
    # r_crse[i], p_crse[i] = padded_corr(model_ctx_bound_vec, p_b_c)
    # r_fine[i], p_fine[i] = padded_corr(model_ctx_bound_vec, p_b_f)
    r_crse[i], p_crse[i] = padded_corr(model_ctx_bound_vec, p_b_c, porp=.1)
    r_fine[i], p_fine[i] = padded_corr(model_ctx_bound_vec, p_b_f, porp=.1)
    # r_crse[i], p_crse[i] = padded_corr(model_ctx_bound_vec, p_b_c, shift=False)
    # r_fine[i], p_fine[i] = padded_corr(model_ctx_bound_vec, p_b_f, shift=False)


f, axes = plt.subplots(2, 1, figsize=(5,7), sharex=True)
sns.violinplot(r_crse, ax=axes[0])
sns.violinplot(r_fine, ax=axes[1])
for ax in axes:
    ax.axvline(0, ls='--', c='grey', label='0', zorder=-1)
    ax.legend()
axes[0].set_title(f'mean r = %.3f' % (r_crse.mean()))
axes[1].set_title(f'mean r = %.3f' % (r_fine.mean()))
axes[1].set_xlabel('Point biserial correlation')
axes[0].set_ylabel('Coarse')
axes[1].set_ylabel('Fine')
f.tight_layout()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'final-r-mode-vs-human-bounds.png')
f.savefig(fig_path, dpi=100)


f, axes = plt.subplots(2, 1, figsize=(5,7))
sns.histplot(p_crse, ax=axes[0])
sns.histplot(p_fine, ax=axes[1])
for ax in axes:
    ax.axvline(0.05, ls='--', c='grey', label='0.05', zorder=-1)
    ax.legend()
axes[0].set_title(f'mean p = %.3f' % (p_crse.mean()))
axes[1].set_title(f'mean p = %.3f' % (p_fine.mean()))
axes[1].set_xlabel('p value')
axes[0].set_ylabel('Freq')
axes[1].set_ylabel('Freq')
f.tight_layout()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'final-p-mode-vs-human-bounds.png')
f.savefig(fig_path, dpi=100)
