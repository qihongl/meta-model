'''train model on all META training data file and eval on the valid files'''
import os
import glob
import time
import pickle
import torch
import random
import argparse
import matplotlib
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix, mutual_info_score
from scipy.stats import pointbiserialr, pearsonr
# from model import CGRU as Agent
from model import CGRU_v2 as Agent
from model import SimpleContext, SimpleShortcut, SimpleTracker
from utils import ID2CHAPTER
from utils import EventLabel, TrainValidSplit, DataLoader, Parameters, HumanBondaries
from utils import to_np, to_pth, split_video_id, context_to_bound_vec, \
    loss_to_bound_vec, save_ckpt, pickle_save, pickle_load, compute_stats, \
    get_point_biserial

sns.set(style='white', palette='colorblind', context='talk')

'''e.g.
python train.py --subj_id 99 --lr 1e-3 --update_freq 10 --dim_hidden 16 --dim_context 256 --ctx_wt .5  --stickiness .5 --try_reset_h 1
sbatch train.sh 99 1e-3 10 16 256 .5 .5 .5
'''
matplotlib.use('Agg')
parser = argparse.ArgumentParser()
parser.add_argument('--subj_id', default=99, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--update_freq', default=1, type=int)
parser.add_argument('--dim_hidden', default=16, type=int)
parser.add_argument('--dim_context', default=128, type=int)
parser.add_argument('--use_shortcut', default=1, type=float)
parser.add_argument('--gen_grad', default=1.5, type=float)
parser.add_argument('--ctx_wt', default=.5, type=float)
parser.add_argument('--concentration', default=1, type=float)
parser.add_argument('--stickiness', default=8, type=float)
parser.add_argument('--lik_softmax_beta', default=.33, type=float)
parser.add_argument('--try_reset_h', default=0, type=int)
parser.add_argument('--pe_tracker_size', default=256, type=int)
parser.add_argument('--match_tracker_size', default=8, type=int)
parser.add_argument('--n_pe_std', default=2, type=int)
parser.add_argument('--exp_name', default='testing', type=str)
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
use_shortcut = bool(args.use_shortcut)
gen_grad = args.gen_grad
ctx_wt = args.ctx_wt
stickiness = args.stickiness
concentration = args.concentration
lik_softmax_beta = args.lik_softmax_beta
try_reset_h = bool(args.try_reset_h)
pe_tracker_size = args.pe_tracker_size
match_tracker_size = args.match_tracker_size
n_pe_std = args.n_pe_std
exp_name = args.exp_name
log_root = args.log_root

# # training param
# exp_name = 'testing'
# log_root = '../log'
# subj_id = 0
# #
# lr = 1e-3
# update_freq = 16
# # model param
# dim_hidden = 16
# dim_context = 128
# # shortcut params
# use_shortcut = False
# gen_grad = 1.0
# # full inference param
# ctx_wt = .5
# concentration = 1
# stickiness = 2.0
# lik_softmax_beta = .33
# try_reset_h = False
# # handoff param
# pe_tracker_size = 256
# match_tracker_size = 8
# n_pe_std = 3

# set seed
np.random.seed(subj_id+777)
torch.manual_seed(subj_id+777)
# init util objects
dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()
hb = HumanBondaries()
p = Parameters(
    dim_hidden = dim_hidden, dim_context = dim_context, ctx_wt = ctx_wt,
    stickiness = stickiness, concentration=concentration,
    gen_grad=gen_grad, lr = lr, update_freq = update_freq, lik_softmax_beta=lik_softmax_beta,
    try_reset_h = try_reset_h, use_shortcut=use_shortcut,
    pe_tracker_size = pe_tracker_size, match_tracker_size = match_tracker_size, n_pe_std= n_pe_std,
    log_root=log_root, exp_name=exp_name, subj_id = subj_id,
)


'''
def load_data(train_mode):
    result_fname = os.path.join(p.result_dir, f'results-train-{train_mode}.pkl')
    result_dict = pickle_load(result_fname)
    log_cid_fi_ = result_dict['log_cid_fi']
    log_cid_sc_ = result_dict['log_cid_sc']
    loss_by_events_ = result_dict['loss_by_events']
    log_reset_h_ = result_dict['log_reset_h']
    return log_cid_fi_, log_cid_sc_, loss_by_events_, log_reset_h_

log_cid_fi_tr, log_cid_sc_tr, loss_by_events_tr, log_reset_h_tr = load_data(True)
log_cid_fi_te, log_cid_sc_te, loss_by_events_te, log_reset_h_te = load_data(False)
'''

# init model
agent = Agent(
    p.dim_input, p.dim_hidden, p.dim_output,
    ctx_wt=p.ctx_wt, context_dim=p.dim_context,
    softmax_beta=p.lik_softmax_beta, try_reset_h=p.try_reset_h,
)
optimizer = torch.optim.Adam(agent.parameters(), lr=p.lr)
# context management
sc = SimpleContext(p.dim_context, p.stickiness, p.concentration, p.try_reset_h)
# c_id, c_vec = sc.init_context()
# init the shortcut
ssc = SimpleShortcut(input_dim=p.dim_input, d=p.gen_grad, use_model=False)

pe_tracker = SimpleTracker(size=pe_tracker_size)
match_tracker = SimpleTracker(size=match_tracker_size)

'''train the model'''

def run_model(event_id_list, p, train_mode, save_freq=10):
    if train_mode:
        save_weights = True
        learning = True
        _, c_vec = sc.init_context()
    else:
        save_weights = True
        learning = True
        c_vec = sc.context[sc.prev_cluster_id]
        # c_vec = sc.context[0]

    # prealooc
    loss_by_events = [[] for _ in range(len(event_id_list))]
    log_cid = [[] for _ in range(len(event_id_list))]
    log_cid_fi = [[] for _ in range(len(event_id_list))]
    log_cid_sc = [[] for _ in range(len(event_id_list))]
    log_reset_h = [[] for _ in range(len(event_id_list))]
    log_use_sc = [[] for _ in range(len(event_id_list))]
    log_pe_peak = [[] for _ in range(len(event_id_list))]

    permed_order = np.random.permutation(range(len(event_id_list)))
    for i, pi in enumerate(permed_order):
        event_id = event_id_list[pi]
        # save data for every other k epochs
        if save_weights and i % save_freq == 0:
            save_ckpt(i, p.log_dir, agent, optimizer, sc.to_dict(), verbose=0)
        print(f'Learning event {i} / {len(event_id_list)} - {event_id}')
        t_start = time.time()
        # get data
        X, t_f1 = dl.get_data(event_id, get_t_frame1=True)
        T = len(X) - 1
        # prealloc
        log_cid_fi_i = np.zeros(T, dtype=int)
        log_cid_sc_i = np.zeros(T, dtype=int)
        log_cid_i = np.zeros(T, dtype=int)
        log_reset_h_i = np.zeros(T, )
        log_use_sc_i = np.zeros(T, )
        log_pe_peak_i = np.zeros(T, )

        # run the model over time
        loss = 0
        h_t = agent.get_init_states()
        for t in range(T):
            # forward
            [y_t_hat, h_t], cache = agent.forward(X[t], h_t, to_pth(c_vec))
            # record losses
            loss_it = agent.criterion(torch.squeeze(y_t_hat), X[t+1])
            loss += loss_it
            loss_by_events[i].append(loss_it.clone().detach())

            # if use shortcut, decide whether to turn off shortcut based on PE
            if use_shortcut and t > 0 and log_use_sc_i[t-1]:
                if pe_tracker.peaked(log_cid_sc_i[t-1], n_pe_std, to_np(loss_it)):
                    match_tracker.reinit_ctx_buffer(log_cid_sc_i[t-1])
                    log_pe_peak_i[t] = True
                    # print(f'PE peaked ({to_np(loss_it)}) while using the shortcut, turn off ctx {log_cid_sc_i[t]}')
            else:
                # if use full inference, record full inf PE as the baseline
                pe_tracker.add(log_cid_fi_i[t-1], to_np(loss_by_events[i][t-1]))

            # short cut inference
            log_cid_sc_i[t] = ssc.predict(to_np(X[t]))
            log_use_sc_i[t] = match_tracker.use_shortcut_t(log_cid_sc_i[t])

            if use_shortcut and log_use_sc_i[t]:
                log_cid_i[t] = log_cid_sc_i[t]
                # print(f'use short cut on ctx {log_cid_sc_i[t]}')
            else:
                # context - full inference
                lik = agent.try_all_contexts(X[t+1], X[t], h_t, sc.context, sc.prev_cluster_id)
                log_cid_fi_i[t], log_reset_h_i[t] = sc.assign_context(lik, verbose=1)
                # if use full inference...
                log_cid_i[t] = log_cid_fi_i[t]
                # if reset h is the best, then reset it
                h_t = agent.get_init_states() if log_reset_h_i[t] else h_t
                # add experience to the shortcut buffer, and record match
                ssc.add_data(to_np(X[t]), log_cid_fi_i[t])
                match_tracker.add(log_cid_sc_i[t], log_cid_fi_i[t] == log_cid_sc_i[t])
            # print(float(torch.sum(h_t)), log_reset_h_i[t])
            # get the context vector
            c_vec = sc.context[log_cid_i[t]]

            # update weights for every other t time points
            if learning and t % update_freq == 0:
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        ssc.update_model()
        log_cid[i] = log_cid_i
        log_cid_fi[i] = log_cid_fi_i
        log_cid_sc[i] = log_cid_sc_i
        log_reset_h[i] = log_reset_h_i
        log_use_sc[i] = log_use_sc_i
        log_pe_peak[i] = log_pe_peak_i

        print('Time elapsed = %.2f sec' % (time.time() - t_start))
    # save the final weights
    if save_weights:
        save_ckpt(len(event_id_list), p.log_dir, agent, optimizer, sc.to_dict(), verbose=True)

    result_dict = {
        'log_cid': log_cid,
        'log_cid_fi' : log_cid_fi,
        'log_cid_sc' : log_cid_sc,
        'loss_by_events' : loss_by_events,
        'log_reset_h' : log_reset_h,
        'log_use_sc' : log_use_sc,
        'log_pe_peak' : log_pe_peak,
    }
    result_fname = os.path.join(p.result_dir, f'results-train-{train_mode}.pkl')
    pickle_save(result_dict, result_fname)
    print('done')
    return log_cid, log_cid_fi, log_cid_sc, loss_by_events, log_reset_h, log_use_sc, log_pe_peak


'''evaluate loss on the validation set'''
results_tr = run_model(tvs.train_ids, p=p, train_mode=True)
[log_cid_tr, log_cid_fi_tr, log_cid_sc_tr, loss_by_events_tr, log_reset_h_tr, log_use_sc_tr, log_pe_peak_tr] = results_tr
results_te = run_model(tvs.valid_ids, p=p, train_mode=False)
[log_cid_te, log_cid_fi_te, log_cid_sc_te, loss_by_events_te, log_reset_h_te, log_use_sc_te, log_pe_peak_te] = results_te


'''plot the data '''
# plot loss by valid event
loss_mu_by_events = [torch.stack(loss_event_i).mean() for loss_event_i in loss_by_events_te]
f, ax = plt.subplots(1,1, figsize=(7,4))
ax.plot(loss_mu_by_events)
ax.set_title('%.3f' % torch.stack(loss_mu_by_events).mean())
ax.set_xlabel('validation video id')
ax.set_ylabel('loss')
sns.despine()
fig_path = os.path.join(p.fig_dir, f'final-loss-by-event.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


def num_boundaries_per_event(log_cid_):
    model_ctx_bound_vec_list = []
    for i, log_cid_j in enumerate(log_cid_):
        model_ctx_bound_vec_list.append(context_to_bound_vec(log_cid_j))
    return np.array([np.sum(x) for x in model_ctx_bound_vec_list])

num_boundaries_tr = num_boundaries_per_event(log_cid_fi_tr)
num_boundaries_te = num_boundaries_per_event(log_cid_fi_te)

f, ax = plt.subplots(1,1, figsize=(7,4))
sns.kdeplot(num_boundaries_tr, ax=ax, label='train')
sns.kdeplot(num_boundaries_te, ax=ax, label='test')
ax.set_title('boundaries under full inference ')
ax.set_ylabel('p')
ax.set_xlabel('# boundaries')
ax.legend()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'distribution-n-bounds.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

# plot n context over time
def compute_n_ctx_over_time(log_cid_):
    max_ctx_by_events = [np.max(log_cid_fi_tr_i) for log_cid_fi_tr_i in log_cid_]
    max_ctx_so_far = 0
    for i in range(len(max_ctx_by_events)):
        if max_ctx_by_events[i] > max_ctx_so_far:
            max_ctx_so_far = max_ctx_by_events[i]
        else:
            max_ctx_by_events[i] = max_ctx_so_far
    return max_ctx_by_events

log_cid_fi = log_cid_fi_tr + log_cid_fi_te
log_cid = log_cid_tr + log_cid_te
n_ctx_over_time_fi = compute_n_ctx_over_time(log_cid_fi)
n_ctx_over_time = compute_n_ctx_over_time(log_cid)
f, ax = plt.subplots(1,1, figsize=(5,4))
ax.plot(n_ctx_over_time_fi)
# ax.plot(n_ctx_over_time, label = 'actual')
ax.set_xlabel('training video id')
ax.set_ylabel('# of contexts inferred')
ax.axvline(len(log_cid_fi_tr), label='start testing', ls='--', color='grey')
ax.legend()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'ctx-num-over-training.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


# are the context being used?
n_ctx = int(n_ctx_over_time_fi[-1])+1
ctx_usage = np.zeros((tvs.n_files, n_ctx))
for i in range(tvs.n_files):
    # check for the ith video, what context were used
    # ctx_used, counts = np.unique(log_cid_fi[i], return_counts=True)
    ctx_used, counts = np.unique(log_cid[i], return_counts=True)

    for ctx_id_i, count_i in zip(ctx_used, counts):
        ctx_usage[i, int(ctx_id_i)] += count_i
    ctx_usage[i,:] /= len(log_cid[i])

f, ax = plt.subplots(1,1, figsize=(12, 10))
sns.heatmap(ctx_usage, cmap='viridis', ax=ax)
ax.set_ylabel('video id (during learning)')
ax.set_xlabel('inferred latent cause id')
ax.set_title('% latent cause used for each video during learning')
fig_path = os.path.join(p.fig_dir, f'ctx-use.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

low_use_threshold = 5
low_use_ctx = []
n_ctx_tr = int(compute_n_ctx_over_time(log_cid_tr)[-1])
f, ax = plt.subplots(1,1, figsize=(14, 6))
for i in np.arange(1, n_ctx_tr):
    if np.sum(ctx_usage[tvs.n_train_files:,i]) < low_use_threshold:
        ax.plot(ctx_usage[:,i], label = i)
        low_use_ctx.append(i)
ax.set_ylabel('# time points')
ax.set_title(f'contexts that are barely used during test (used less than {low_use_threshold} time points) ')
ax.set_xlabel('video id (during learning)')
ax.legend()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'ctx-barely-used.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


# plot average event duration
def get_event_len(log_cid_):
    event_len = []
    for cid_i in log_cid_:
        loc_boundaries = np.where(np.diff(cid_i)!=0)[0]
        if len(loc_boundaries) == 0:
            event_len.append(len(cid_i))
        else:
            # event_len.extend(loc_boundaries)
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

# event length under full inference
# event_len_tr = get_event_len(log_cid_fi_tr)
# event_len_te = get_event_len(log_cid_fi_te)
# f, ax = plot_event_len_distribution(event_len_tr)
# fig_path = os.path.join(p.fig_dir, f'len-event-train.png')
# f.savefig(fig_path, dpi=100, bbox_inches='tight')
# f, ax = plot_event_len_distribution(event_len_te)
# fig_path = os.path.join(p.fig_dir, f'len-event-test.png')
# f.savefig(fig_path, dpi=100, bbox_inches='tight')

# actual event length
event_len_tr = get_event_len(log_cid_tr)
f, ax = plot_event_len_distribution(event_len_tr)
fig_path = os.path.join(p.fig_dir, f'len-event-train.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

event_len_te = get_event_len(log_cid_te)
f, ax = plot_event_len_distribution(event_len_te)
fig_path = os.path.join(p.fig_dir, f'len-event-test.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

# def resize_bound_vecs(model_bound_vec_, pad_l_, hb_):
#     model_bound_vec_ = np.concatenate([np.zeros(pad_l_), model_bound_vec_])
#     # trim one vector if the other is too long
#     hb_ = hb_[:len(model_bound_vec_)]
#     model_bound_vec_ = model_bound_vec_[:len(hb_)]
#     return model_bound_vec_, hb_

'''correlation with human boundaries'''


def vec_to_sec(vec):
    return [np.any(vec[t:t+3])==1 for t in range(0, len(vec), 3)]

def ctx_vec_to_sec(vec):
    return [Counter(vec[t:t+3]).most_common()[0][0] for t in range(0, len(vec), 3)]

r_m_crse = np.zeros(len(log_cid_fi_te),)
r_m_fine = np.zeros(len(log_cid_fi_te),)
r_l_crse = np.zeros(len(log_cid_fi_te),)
r_l_fine = np.zeros(len(log_cid_fi_te),)
model_bounds_c, model_bounds_f, chbs, fhbs = [], [], [], []
sub_evn_label, mod_evn_label = [], []

event_id_list = tvs.valid_ids
t_f1 = dl.get_1st_frame_ids(event_id_list)
mi = np.zeros(len(event_id_list))
for i, event_id in enumerate(event_id_list):
    # if i == 0: break

    actor_id, chapter_id, run_id = split_video_id(event_id)

    chb = hb.get_bound_prob(event_id_list[i], 'coarse', to_sec=True)
    fhb = hb.get_bound_prob(event_id_list[i], 'fine', to_sec=True)

    model_ctx_bound_vec = context_to_bound_vec(log_cid_te[i])
    model_ctx_bound_vec_fi = context_to_bound_vec(log_cid_fi_te[i])
    model_ctx_bound_vec_sc = context_to_bound_vec(log_cid_sc_te[i])
    model_loss_bound_vec = loss_to_bound_vec(loss_by_events_te[i], model_ctx_bound_vec)

    model_ctx_bound_vec = vec_to_sec(model_ctx_bound_vec)
    model_ctx_bound_vec_fi = vec_to_sec(model_ctx_bound_vec_fi)
    model_ctx_bound_vec_sc = vec_to_sec(model_ctx_bound_vec_sc)
    model_loss_bound_vec = vec_to_sec(model_loss_bound_vec)

    # get the true event label
    event_bound_times, event_bound_vec = evlab.get_bounds(event_id_list[i], to_sec=True)
    # get PE peaks
    pe_peak_locs = np.where(log_pe_peak_te[i])[0]

    # left pad by the 1st frame index
    # pad_l = int(t_f1[i] * 3)
    pad_l = int(np.round(t_f1[i]))
    model_ctx_bound_vec = np.concatenate([np.zeros(pad_l), model_ctx_bound_vec])
    model_ctx_bound_vec_fi = np.concatenate([np.zeros(pad_l), model_ctx_bound_vec_fi])
    model_ctx_bound_vec_sc = np.concatenate([np.zeros(pad_l), model_ctx_bound_vec_sc])
    model_loss_bound_vec = np.concatenate([np.zeros(pad_l), model_loss_bound_vec])
    reset_h_vec = np.concatenate([np.zeros(pad_l), log_reset_h_te[i]])

    # if the model data is longer, trim the human data
    chb = chb[:len(model_ctx_bound_vec)]
    fhb = fhb[:len(model_ctx_bound_vec)]
    # if human data is longer trim the model data
    model_ctx_bound_vec_c = model_ctx_bound_vec[:len(chb)]
    model_ctx_bound_vec_f = model_ctx_bound_vec[:len(fhb)]
    model_ctx_bound_vec_fi_c = model_ctx_bound_vec_fi[:len(chb)]
    model_ctx_bound_vec_fi_f = model_ctx_bound_vec_fi[:len(fhb)]
    model_ctx_bound_vec_sc_c = model_ctx_bound_vec_sc[:len(chb)]
    model_ctx_bound_vec_sc_f = model_ctx_bound_vec_sc[:len(fhb)]
    model_loss_bound_vec_c = model_loss_bound_vec[:len(chb)]
    model_loss_bound_vec_f = model_loss_bound_vec[:len(fhb)]
    reset_h_vec_c = reset_h_vec[:len(chb)]
    reset_h_vec_f = reset_h_vec[:len(fhb)]

    model_ctx_bound_loc_c = np.where(model_ctx_bound_vec_c)[0]
    model_ctx_bound_loc_f = np.where(model_ctx_bound_vec_f)[0]
    model_ctx_bound_loc_fi_c = np.where(model_ctx_bound_vec_fi_c)[0]
    model_ctx_bound_loc_fi_f = np.where(model_ctx_bound_vec_fi_f)[0]
    model_ctx_bound_loc_sc_c = np.where(model_ctx_bound_vec_sc_c)[0]
    model_ctx_bound_loc_sc_f = np.where(model_ctx_bound_vec_sc_f)[0]
    reset_h_loc_c = np.where(reset_h_vec_c)[0]
    reset_h_loc_f = np.where(reset_h_vec_f)[0]

    # compute corr
    r_m_crse[i], _ = get_point_biserial(model_ctx_bound_vec_c, chb)
    r_m_fine[i], _ = get_point_biserial(model_ctx_bound_vec_f, fhb)
    r_l_crse[i], _ = get_point_biserial(model_loss_bound_vec_c, chb)
    r_l_fine[i], _ = get_point_biserial(model_loss_bound_vec_f, fhb)

    model_bounds_c.append(model_ctx_bound_vec_c)
    model_bounds_f.append(model_ctx_bound_vec_f)
    chbs.append(chb)
    fhbs.append(fhb)

    # get the context vector and event label into sec space
    log_cid_te_i_sec = ctx_vec_to_sec(log_cid_te[i])
    sub_evn_label_i = evlab.get_subev_labels(event_id, to_sec=True)
    # left trim the vectors
    sub_evn_label_i = sub_evn_label_i[pad_l:]
    # right trim the vector
    min_len = np.min([len(sub_evn_label_i), len(log_cid_te_i_sec)])
    sub_evn_label_i = sub_evn_label_i[:min_len]
    log_cid_te_i_sec = log_cid_te_i_sec[:min_len]
    # compute MI
    nan_mask = np.logical_or(np.isnan(log_cid_te_i_sec), np.isnan(sub_evn_label_i))
    sub_evn_label.append(np.array(sub_evn_label_i)[~nan_mask])
    mod_evn_label.append(np.array(log_cid_te_i_sec)[~nan_mask])
    mi[i] = mutual_info_score(sub_evn_label[i], mod_evn_label[i])

    # assert set(model_ctx_bound_loc_fi_f).issubset(set(model_ctx_bound_loc_f))
    # assert set(model_ctx_bound_loc_fi_c).issubset(set(model_ctx_bound_loc_c))

    '''plot this event'''
    f, axes = plt.subplots(2, 1, figsize=(12, 9))

    for j, mb in enumerate(model_ctx_bound_loc_c):
        ls = '--' if mb in model_ctx_bound_loc_fi_f else ':'
        label = 'model bound' if j == 0 else None
        color = 'black' if mb in reset_h_loc_c else 'grey'
        axes[0].axvline(mb, ls=ls, color=color, label=label)

    for j, mb in enumerate(model_ctx_bound_loc_f):

        ls = '--' if mb in model_ctx_bound_loc_fi_f else ':'
        label = 'model bound' if j == 0 else None
        color = 'black' if mb in reset_h_loc_f else 'grey'
        axes[1].axvline(mb, ls=ls, color=color, label=label)

    for j, pe_peak_loc in enumerate(pe_peak_locs):
        if pe_peak_loc > len(chb): continue
        label = 'shortcut PE peak' if j == 0 else None
        axes[0].scatter(pe_peak_loc, 0, marker='x', color='red', label=label)
        axes[1].scatter(pe_peak_loc, 0, marker='x', color='red', label=label)

    assert len(chb) == len(model_ctx_bound_vec_c)
    assert len(fhb) == len(model_ctx_bound_vec_f)
    axes[0].plot(chb, label='coarse bounds')
    axes[1].plot(fhb, label='fine bounds')
    axes[0].legend()
    axes[1].legend()
    axes[0].set_ylabel('p')
    axes[1].set_ylabel('p')
    axes[0].set_title(f'mean correlation with coarse human boundaries = %.3f' % (r_m_crse[i]))
    axes[1].set_title(f'mean correlation with fine human boundaries = %.3f' % (r_m_fine[i]))
    axes[1].set_xlabel('Time unit (X)')
    f.suptitle(f'{event_id} (actor: {actor_id}, chapter: {ID2CHAPTER[chapter_id]}, run: {run_id})')
    f.tight_layout()
    sns.despine()
    fig_path = os.path.join(p.fig_dir, f'event-{event_id}-mb-vs-hb.png')
    f.savefig(fig_path, dpi=100, bbox_inches='tight')


def compute_corr_with_perm(model_bounds_list, phuman_bounds_list, n_perms = 500):
    r_perm = np.zeros(n_perms, )
    all_event_segs = []
    for i, model_bounds_list_i in enumerate(model_bounds_list):
        all_event_segs.extend(get_event_segments(model_bounds_list_i))
    for i in range(n_perms):
        random.shuffle(all_event_segs)
        r_perm[i], _ = get_point_biserial(
            np.concatenate(all_event_segs), np.concatenate(phuman_bounds_list)
        )
    r, _ = get_point_biserial(
        np.concatenate(model_bounds_list), np.concatenate(phuman_bounds_list)
    )
    return r_perm, r


def get_event_segments(model_bounds_arr):
    bound_locs = list(np.where(model_bounds_arr)[0])
    if len(bound_locs) == 0:
        return [model_bounds_arr]
    bound_locs = [0] + bound_locs
    all_event_segs = [model_bounds_arr[bound_locs[i]+1 : bound_locs[i+1]+1] for i in range(len(bound_locs)-1)]
    all_event_segs.append(model_bounds_arr[bound_locs[-1]:])
    return all_event_segs

def get_mod_evn_label_segments(mod_evn_label_i):
    mod_evn_label_seg = []
    bound_locs = list(np.where(np.diff(mod_evn_label_i))[0])
    if len(bound_locs) == 0:
        mod_evn_label_seg.append(mod_evn_label_i)
    else:
        bound_locs = [0] + bound_locs + [len(mod_evn_label_i)]
        for j in range(len(bound_locs)-1):
            l, r = bound_locs[j], bound_locs[j+1]
            mod_evn_label_seg.append(mod_evn_label_i[l:r])
    return mod_evn_label_seg

def compute_mi_with_perm(mod_evn_label, sub_evn_label, n_perms = 500):
    mi_perm = np.zeros(n_perms, )
    all_mod_evn_label_seg = []
    for i, mod_evn_label_i in enumerate(mod_evn_label):
        mod_evn_label_seg_i = get_mod_evn_label_segments(mod_evn_label_i)
        all_mod_evn_label_seg.extend(mod_evn_label_seg_i)
    for i in range(n_perms):
        random.shuffle(all_mod_evn_label_seg)
        mi_perm[i] = mutual_info_score(
            np.concatenate(all_mod_evn_label_seg), np.concatenate(sub_evn_label)
        )
    mi = mutual_info_score(
        np.concatenate(mod_evn_label), np.concatenate(sub_evn_label)
    )
    return mi_perm, mi



mi_perm, mi = compute_mi_with_perm(mod_evn_label, sub_evn_label)
f, ax = plt.subplots(1,1, figsize=(7, 4))
sns.kdeplot(mi_perm, label='permutation')
ax.axvline(mi, ls='--', color='k', label='observed = %.2f' % mi)
ax.set_title('MI model vs. truth')
ax.set_xlabel('Point biserial correlation')
ax.legend()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'final-mi.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

human_r_crse, human_r_fine = hb.get_average_human_ceiling(tvs.valid_ids)

r_perm, r = compute_corr_with_perm(model_bounds_c, chbs)
f, axes = plt.subplots(2, 1, figsize=(8, 8), sharey=True)
sns.kdeplot(r_perm, label='permutation', ax=axes[0])
axes[0].axvline(np.nanmean(r_m_crse), ls='--', color='k', label='observed = %.2f' % np.nanmean(r_m_crse))
axes[0].axvline(np.nanmean(human_r_crse), ls='--', color='red', label='human baseline = %.2f' % np.nanmean(human_r_crse))
axes[0].set_title('model boundaries vs. corase human boundaries')
axes[0].set_xlabel('Point biserial correlation')
axes[0].set_xlim([0, None])
axes[0].legend()

r_perm, r = compute_corr_with_perm(model_bounds_f, fhbs)
sns.kdeplot(r_perm, label='permutation', ax=axes[1])
axes[1].axvline(np.nanmean(r_m_fine), ls='--', color='k', label='observed = %.2f' % np.nanmean(r_m_fine))
axes[1].axvline(np.nanmean(human_r_fine), ls='--', color='red', label='human baseline = %.2f' % np.nanmean(human_r_fine))
axes[1].set_title('model boundaries vs. fine human boundaries')
axes[1].set_xlabel('Point biserial correlation')
axes[1].set_xlim([0, None])
axes[1].legend()
sns.despine()
f.tight_layout()
fig_path = os.path.join(p.fig_dir, f'final-r-model-vs-hb-permutation.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

n_bounds = [np.sum(x) for x in model_bounds_c]
f, ax = plt.subplots(1,1, figsize=(7,4))
ax.hist(n_bounds)
ax.set_title('mean = %.2f' % (np.nanmean(n_bounds)))
ax.set_xlabel('# boundaries')
ax.set_ylabel('Frequency')
sns.despine()
fig_path = os.path.join(p.fig_dir, f'n-bounds.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

# f, axes = plt.subplots(2, 1, figsize=(5,7), sharex=True)
# sns.violinplot(r_m_crse, ax=axes[0])
# sns.violinplot(r_m_fine, ax=axes[1])
# for ax in axes:
#     ax.axvline(0, ls='--', c='grey', label='0', zorder=-1)
#     ax.legend()
# axes[0].set_title(f'mean r = %.3f' % (np.nanmean(r_m_crse)))
# axes[1].set_title(f'mean r = %.3f' % (np.nanmean(r_m_fine)))
# axes[1].set_xlabel('Point biserial correlation')
# axes[0].set_ylabel('Coarse')
# axes[1].set_ylabel('Fine')
# f.tight_layout()
# sns.despine()
# fig_path = os.path.join(p.fig_dir, f'final-r-model-vs-hb.png')
# f.savefig(fig_path, dpi=100, bbox_inches='tight')
#
#
# f, axes = plt.subplots(2, 1, figsize=(5,7), sharex=True)
# sns.violinplot(r_l_crse, ax=axes[0])
# sns.violinplot(r_l_fine, ax=axes[1])
# for ax in axes:
#     ax.axvline(0, ls='--', c='grey', label='0', zorder=-1)
#     ax.legend()
# axes[0].set_title(f'mean r = %.3f' % (np.nanmean(r_l_crse)))
# axes[1].set_title(f'mean r = %.3f' % (np.nanmean(r_l_fine)))
# axes[1].set_xlabel('Point biserial correlation')
# axes[0].set_ylabel('Coarse')
# axes[1].set_ylabel('Fine')
# f.tight_layout()
# sns.despine()
# fig_path = os.path.join(p.fig_dir, f'final-r-loss-vs-hb.png')
# f.savefig(fig_path, dpi=100, bbox_inches='tight')

# compute shortcut accuracy over time
sc_acc_tr, sc_acc_te = [], []
for i, (log_cid_fi_tr_i, log_cid_sc_tr_i) in enumerate(zip(log_cid_fi_tr, log_cid_sc_tr)):
    # if i == 0: break
    sc_acc_tr.append(np.mean(log_cid_fi_tr_i == log_cid_sc_tr_i))
for i, (log_cid_fi_te_i, log_cid_sc_te_i) in enumerate(zip(log_cid_fi_te, log_cid_sc_te)):
    sc_acc_te.append(np.mean(log_cid_fi_te_i == log_cid_sc_te_i))

sc_acc_tr_mu, sc_acc_tr_se = compute_stats(sc_acc_tr)
sc_acc_te_mu, sc_acc_te_se = compute_stats(sc_acc_te)
# compute precent null
percent_sc_null_tr = [np.mean(x == -1) for x in log_cid_sc_tr]
percent_sc_null_te = [np.mean(x == -1) for x in log_cid_sc_te]

sc_pnull_tr_mu, sc_pnull_tr_se = compute_stats(percent_sc_null_tr)
sc_pnull_te_mu, sc_pnull_te_se = compute_stats(percent_sc_null_te)

f, axes = plt.subplots(1,2, figsize=(10,4), sharey=True)
axes[0].plot(sc_acc_tr)
axes[1].plot(sc_acc_te)
axes[0].set_ylim([0,1])
axes[0].set_ylabel('% match shortcut vs. full inference')
axes[0].set_xlabel('training videos')
axes[1].set_xlabel('validation videos')
axes[0].set_title('mean = %.2f' % np.mean(sc_acc_tr_mu))
axes[1].set_title('mean = %.2f' % np.mean(sc_acc_te_mu))
sns.despine()
fig_path = os.path.join(p.fig_dir, f'sc-acc.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

f, ax = plt.subplots(1,1, figsize=(5,4))
ax.plot(percent_sc_null_tr + percent_sc_null_te)
ax.set_ylabel('% null')
ax.set_xlabel('video id')
sns.despine()
fig_path = os.path.join(p.fig_dir, f'sc-null.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

width = .7
f, ax = plt.subplots(1,1, figsize=(4.5, 4))
xticks = range(2)
ax.bar(x=xticks, width=width, height=[sc_acc_tr_mu, sc_acc_te_mu], yerr=[sc_acc_tr_se, sc_acc_te_se], label='match with full inference')
ax.bar(x=xticks, width=width, height=[sc_pnull_tr_mu, sc_pnull_te_mu], yerr=[sc_pnull_tr_se, sc_pnull_te_se], bottom=[sc_acc_tr_mu, sc_acc_te_mu], label='null response')
ax.set_xticks(xticks)
ax.set_xticklabels(['train', 'validation'])
ax.set_title('short cut performance')
ax.set_ylabel('%')
ax.set_ylim([0,1])
ax.legend()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'sc-bar.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


log_cid_sc_cat = np.concatenate(log_cid_sc_tr+log_cid_sc_te)
log_cid_fi_cat = np.concatenate(log_cid_fi_tr+log_cid_fi_te)
log_cid_sc_cat[np.isnan(log_cid_sc_cat)] = -1

log_cid_sc_cat_nonull = log_cid_sc_cat[log_cid_sc_cat != -1]
log_cid_fi_cat_nonull = log_cid_fi_cat[log_cid_sc_cat != -1]

cfmat = confusion_matrix(log_cid_fi_cat_nonull, log_cid_sc_cat_nonull, normalize='true')

f, ax = plt.subplots(1,1, figsize=(12,10))
sns.heatmap(cfmat, cmap='viridis', square=True)
ax.set_xlabel('context id, shortcut')
ax.set_ylabel('context id, full inference')
fig_path = os.path.join(p.fig_dir, f'sc-confusion-mat.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

f, ax = plt.subplots(1,1, figsize=(5,4))
ax.plot(np.diag(cfmat))
ax.set_ylabel('% match shortcut vs full inference')
ax.set_xlabel('context id')
sns.despine()
fig_path = os.path.join(p.fig_dir, f'sc-confusion-mat-diag.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

'''percent shortcut activation '''
log_use_sc_te_mu, log_use_sc_te_se = compute_stats([np.mean(x) for x in log_use_sc_te])

width = .7
f, ax = plt.subplots(1,1, figsize=(2.5, 4))
xticks = range(1)
ax.bar(x=xticks, width=width, height=[log_use_sc_te_mu], yerr=[log_use_sc_te_se])
ax.set_xticks([])
ax.set_title('percent shortcut activation')
ax.set_ylabel('%')
ax.set_ylim([0,1])
# ax.legend()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'sc-p-act.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


'''more segs if new actor or new chapter? - NO '''
# boundaries vs. k-th time seeing an chapter/actor
# - by # of new subevents is better?
# log_cid_fi_all = log_cid_fi_tr + log_cid_fi_te
# num_boundaries = np.concatenate([num_boundaries_tr, num_boundaries_te])
# actor_ids = [None] * len(log_cid_fi_all)
# chapter_ids = [None] * len(log_cid_fi_all)
# n_actor_obs = np.zeros(len(log_cid_fi_all),)
# n_chapter_obs = np.zeros(len(log_cid_fi_all),)
# used_ctxs = set()
# n_new_ctxs = np.zeros(len(log_cid_fi_all),)
# for i, (event_id, log_cid_fi_i) in enumerate(zip(tvs.all_ids, log_cid_fi_all)):
#     # if i > 10: break
#     used_ctxs_up_to_i = set(log_cid_fi_i).union(used_ctxs)
#     n_new_ctxs[i] = len(used_ctxs_up_to_i) - len(used_ctxs)
#     actor_id, chapter_id, _ = split_video_id(event_id)
#     # compute dependent vars - related to novelty
#     n_actor_obs[i] = np.sum(np.array(actor_ids[:i]) == actor_id)
#     n_chapter_obs[i] = np.sum(np.array(actor_ids[:i]) == chapter_id)
#     # print(n_actor_obs[i], n_chapter_obs[i])
#     # print(n_actor_obs, n_chapter_obs)
#     # add this event id
#     actor_ids[i], chapter_ids[i] = actor_id, chapter_id
#
# r,p = pearsonr(n_actor_obs, n_new_ctxs)
# r,p = pearsonr(n_chapter_obs, n_new_ctxs)
# r,p = pearsonr(n_actor_obs, num_boundaries)
# r,p = pearsonr(n_chapter_obs, num_boundaries)


# evlab.
'''permutation and MI'''
# evlab
