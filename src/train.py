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
from utils import to_np, to_pth, split_video_id, context_to_bound_vec, loss_to_bound_vec, save_ckpt, padded_corr, pickle_save, get_point_biserial
from utils import EventLabel, TrainValidSplit, DataLoader, Parameters, HumanBondaries
from utils import ID2CHAPTER
from scipy.stats import pointbiserialr, pearsonr
sns.set(style='white', palette='colorblind', context='talk')

'''e.g.
python train.py --subj_id 99 --lr 1e-3 --update_freq 10 --dim_hidden 16 --dim_context 256 --ctx_wt .5  --stickiness .5 --try_reset_h 1
sbatch train.sh 99 1e-3 10 16 256 .5 .5 .5
'''
# parser = argparse.ArgumentParser()
# parser.add_argument('--subj_id', default=99, type=int)
# parser.add_argument('--lr', default=1e-3, type=float)
# parser.add_argument('--update_freq', default=64, type=int)
# parser.add_argument('--dim_hidden', default=16, type=int)
# parser.add_argument('--dim_context', default=128, type=int)
# parser.add_argument('--ctx_wt', default=.5, type=float)
# parser.add_argument('--stickiness', default=1, type=float)
# parser.add_argument('--lik_softmax_beta', default=.33, type=float)
# parser.add_argument('--try_reset_h', default=0, type=int)
# parser.add_argument('--log_root', default='../log', type=str)
# args = parser.parse_args()
# print(args)
#
# '''params for the model'''
#
# # training param
# subj_id = args.subj_id
# lr = args.lr
# update_freq = args.update_freq
# dim_hidden = args.dim_hidden
# dim_context = args.dim_context
# ctx_wt = args.ctx_wt
# stickiness = args.stickiness
# lik_softmax_beta = args.lik_softmax_beta
# try_reset_h = bool(args.try_reset_h)
# log_root = args.log_root

# training param
subj_id = 0
lr = 1e-3
update_freq = 64
# model param
dim_hidden = 16
dim_context = 128
ctx_wt = .5
# ctx_wt = 0
stickiness = 1
lik_softmax_beta = .33
try_reset_h = 1
log_root = '../log'

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
    stickiness = stickiness, lr = lr, update_freq = update_freq,
    subj_id = subj_id, lik_softmax_beta=lik_softmax_beta, try_reset_h = try_reset_h,
    log_root=log_root,
)
# init model
agent = Agent(
    p.dim_input, p.dim_hidden, p.dim_output,
    ctx_wt=p.ctx_wt, context_dim=p.dim_context,
    softmax_beta=p.lik_softmax_beta, try_reset_h=p.try_reset_h,
)
optimizer = torch.optim.Adam(agent.parameters(), lr=p.lr)
# context management
sc = SimpleContext(p.dim_context, p.stickiness, p.try_reset_h)
c_id, c_vec = sc.init_context()

'''train the model'''

def run_model(event_id_list, p, train_mode, save_freq=10):
    if train_mode:
        save_weights = True
        learning = True
    else:
        save_weights = True
        learning = True

    # prealooc
    loss_by_events = [[] for _ in range(len(event_id_list))]
    log_cid = [[] for _ in range(len(event_id_list))]
    permed_order = np.random.permutation(range(len(event_id_list)))
    for i, pi in enumerate(permed_order):
        event_id = event_id_list[pi]
        # save data for every other k epochs
        if save_weights and i % save_freq == 0:
            save_ckpt(i, p.log_dir, agent, optimizer, sc.to_dict(), verbose=True)
        print(f'Learning event {i} / {len(event_id_list)} - {event_id}')
        t_start = time.time()
        # get data
        X, t_f1 = dl.get_data(event_id, get_t_frame1=True)

        T = len(X) - 1
        # prealloc
        log_cid_i = np.zeros(T, )
        # run the model over time
        loss = 0
        h_t = agent.get_init_states()
        for t in tqdm(range(T)):
            # context - full inference
            lik = agent.try_all_contexts(X[t+1], X[t], h_t, sc.context, sc.prev_cluster_id)
            log_cid_i[t], c_vec, reset_h = sc.assign_context(lik, verbose=1)
            h_t = agent.get_init_states() if reset_h else h_t
            # forward
            [y_t_hat, h_t], cache = agent.forward(X[t], h_t, to_pth(c_vec))
            # record losses
            loss_it = agent.criterion(torch.squeeze(y_t_hat), X[t+1])
            loss += loss_it
            loss_by_events[i].append(loss_it.clone().detach())

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

    result_dict = {
        'log_cid' : log_cid,
        'loss_by_events' : loss_by_events,
    }
    result_fname = os.path.join(p.result_dir, f'results-train-{train_mode}.pkl')
    pickle_save(result_dict, result_fname)
    print('done')
    return log_cid, loss_by_events


'''evaluate loss on the validation set'''
log_cid_tr, loss_by_events_tr = run_model(tvs.train_ids, p=p, train_mode=True)
log_cid, loss_by_events = run_model(tvs.valid_ids, p=p, train_mode=False)


'''plot the data '''
# plot loss by valid event
loss_mu_by_events = [torch.stack(loss_event_i).mean() for loss_event_i in loss_by_events]
f, ax = plt.subplots(1,1, figsize=(7,4))
ax.plot(loss_mu_by_events)
ax.set_title('%.3f' % torch.stack(loss_mu_by_events).mean())
ax.set_xlabel('validation video id')
ax.set_ylabel('loss')
sns.despine()
fig_path = os.path.join(p.fig_dir, f'final-loss-by-event.png')
f.savefig(fig_path, dpi=100)


def num_boundaries_per_event(log_cid_):
    model_ctx_bound_vec_list = []
    for i, log_cid_i in enumerate(log_cid_):
        model_ctx_bound_vec_list.append(context_to_bound_vec(log_cid_i))
    return np.array([np.sum(x) for x in model_ctx_bound_vec_list])

num_boundaries_tr = num_boundaries_per_event(log_cid_tr)
num_boundaries_te = num_boundaries_per_event(log_cid)

f, ax = plt.subplots(1,1, figsize=(7,4))
sns.kdeplot(num_boundaries_tr, ax=ax, label='train')
sns.kdeplot(num_boundaries_te, ax=ax, label='test')
ax.set_ylabel('p')
ax.set_xlabel('# boundaries')
ax.legend()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'distribution-n-bounds.png')
f.savefig(fig_path, dpi=100)

# plot n context over time
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


# plot average event duration
def get_event_len(log_cid_):
    event_len = []
    for cid_i in log_cid_:
        loc_boundaries = np.where(np.diff(cid_i)!=0)[0]
        if len(loc_boundaries) == 0:
            event_len.append(len(cid_i))
        else:
            event_len.extend(loc_boundaries)
            event_len.append(len(cid_i) - loc_boundaries[-1])
    return np.array(event_len)

def plot_event_len_distribution(event_len_, to_sec=True):
    f, ax = plt.subplots(1,1, figsize=(7,4))
    if to_sec:
        event_len_ = event_len_ / 3
        xlabel = 'Length of inferred events (sec)'
        title = f'mean = %.2f sec' % (np.mean(event_len_))
    else:
        xlabel = 'Length of inferred events'
        title = f'mean = %.2f' % (np.mean(event_len_))
    sns.histplot(event_len_, kde=False, stat='probability', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    sns.despine()
    f.tight_layout()
    return f, ax


event_len_tr = get_event_len(log_cid_tr)
event_len_te = get_event_len(log_cid)
f, ax = plot_event_len_distribution(event_len_tr)
fig_path = os.path.join(p.fig_dir, f'len-event-train.png')
f.savefig(fig_path, dpi=100)
f, ax = plot_event_len_distribution(event_len_te)
fig_path = os.path.join(p.fig_dir, f'len-event-test.png')
f.savefig(fig_path, dpi=100)

'''correlation with human boundaries'''

r_m_crse = np.zeros(len(log_cid),)
r_m_fine = np.zeros(len(log_cid),)
r_l_crse = np.zeros(len(log_cid),)
r_l_fine = np.zeros(len(log_cid),)
model_bounds_c, model_bounds_f, chbs, fhbs = [], [], [], []
event_id_list = tvs.valid_ids
t_f1 = dl.get_1st_frame_ids(event_id_list)
for i, event_id in enumerate(event_id_list):
    actor_id, chapter_id, run_id = split_video_id(event_id)
    chb = hb.get_bound_prob(event_id_list[i], 'coarse')
    fhb = hb.get_bound_prob(event_id_list[i], 'fine')
    # get model bounds
    model_ctx_bound_vec = context_to_bound_vec(log_cid[i])
    model_ctx_bound_loc = np.where(model_ctx_bound_vec)[0]
    model_loss_bound_vec = loss_to_bound_vec(loss_by_events[i], model_ctx_bound_vec)
    # get the true event label
    event_bound_times, event_bound_vec = evlab.get_bounds(event_id_list[i])

    # left pad by the 1st frame index
    pad_l = int(t_f1[i] * 3)
    model_ctx_bound_vec = np.concatenate([np.zeros(pad_l), model_ctx_bound_vec])
    model_loss_bound_vec = np.concatenate([np.zeros(pad_l), model_loss_bound_vec])
    # if the model data is longer, trim the human data
    chb = chb[:len(model_ctx_bound_vec)]
    fhb = fhb[:len(model_ctx_bound_vec)]
    # if human data is longer trim the model data
    model_ctx_bound_vec_c = model_ctx_bound_vec[:len(chb)]
    model_ctx_bound_vec_f = model_ctx_bound_vec[:len(fhb)]
    model_loss_bound_vec_c = model_loss_bound_vec[:len(chb)]
    model_loss_bound_vec_f = model_loss_bound_vec[:len(fhb)]

    # compute corr
    r_m_crse[i], _ = get_point_biserial(model_ctx_bound_vec_c, chb)
    r_m_fine[i], _ = get_point_biserial(model_ctx_bound_vec_f, fhb)
    r_l_crse[i], _ = get_point_biserial(model_loss_bound_vec_c, chb)
    r_l_fine[i], _ = get_point_biserial(model_loss_bound_vec_f, fhb)

    model_bounds_c.append(model_ctx_bound_vec_c)
    model_bounds_f.append(model_ctx_bound_vec_f)
    chbs.append(chbs)
    fhbs.append(fhbs)

    '''plot this event'''
    alpha = .75
    f, ax = plt.subplots(1,1, figsize=(12,4))
    ax.set_xlabel('Time unit (X)')
    ax.set_title(f'{event_id} (actor: {actor_id}, chapter: {ID2CHAPTER[chapter_id]}, run: {run_id}) \n correlation with corase/fine boundaries = %.3f / %.3f' % (r_m_crse[i], r_m_fine[i]))

    for j, mb in enumerate(model_ctx_bound_loc):
        label = 'model bound' if j == 0 else None
        ax.axvline(mb, ls='--', color='grey', label=label)
    ax.plot(chb, label='coarse bounds', alpha=alpha)
    ax.plot(fhb, label='fine bounds', alpha=alpha)
    ax.legend()
    sns.despine()
    fig_path = os.path.join(p.fig_dir, f'event-{event_id}-mb-vs-hb.png')
    f.savefig(fig_path, dpi=100)


f, axes = plt.subplots(2, 1, figsize=(5,7), sharex=True)
sns.violinplot(r_m_crse, ax=axes[0])
sns.violinplot(r_m_fine, ax=axes[1])
for ax in axes:
    ax.axvline(0, ls='--', c='grey', label='0', zorder=-1)
    ax.legend()
axes[0].set_title(f'mean r = %.3f' % (np.nanmean(r_m_crse)))
axes[1].set_title(f'mean r = %.3f' % (np.nanmean(r_m_fine)))
axes[1].set_xlabel('Point biserial correlation')
axes[0].set_ylabel('Coarse')
axes[1].set_ylabel('Fine')
f.tight_layout()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'final-r-mode-vs-human-bounds.png')
f.savefig(fig_path, dpi=100)

f, axes = plt.subplots(2, 1, figsize=(5,7), sharex=True)
sns.violinplot(r_l_crse, ax=axes[0])
sns.violinplot(r_l_fine, ax=axes[1])
for ax in axes:
    ax.axvline(0, ls='--', c='grey', label='0', zorder=-1)
    ax.legend()
axes[0].set_title(f'mean r = %.3f' % (r_l_crse.mean()))
axes[1].set_title(f'mean r = %.3f' % (r_l_fine.mean()))
axes[1].set_xlabel('Point biserial correlation')
axes[0].set_ylabel('Coarse')
axes[1].set_ylabel('Fine')
f.tight_layout()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'final-r-loss-vs-human-bounds.png')
f.savefig(fig_path, dpi=100)


# mb_ = model_bounds_c
# hb_ = chbs
#
# hb_cat = np.concatenate(hb_)
#
# model_bounds_f
# fhbs
