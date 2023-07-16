'''eval trained model group on valid files'''
import os
import glob
import time
import pickle
import torch
import random
import argparse
import itertools
import matplotlib
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dabest
from scipy.stats import norm
from pathlib import Path
from itertools import islice
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix, mutual_info_score
from scipy.stats import pointbiserialr, pearsonr
from model import CGRU_v2 as Agent
from utils import ID2CHAPTER
from utils import EventLabel, TrainValidSplit, DataLoader, Parameters, HumanBondaries
from utils import to_np, to_pth, split_video_id, context_to_bound_vec, \
    loss_to_bound_vec, save_ckpt, pickle_save, pickle_load, compute_stats, \
    get_point_biserial

sns.set(style='white', palette='colorblind', context='talk')

log_root = '/tigress/qlu/logs/meta-model/log/'

# general objs
dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()
hb = HumanBondaries()

# util funcs to be added
def list_fnames(data_dir, fpattern):
    '''
    list all fnames/fpaths with a particular fpattern (e.g. *pca.pkl)
    '''
    fpaths = glob.glob(os.path.join(data_dir, fpattern))
    n_data_files = len(fpaths)
    fnames = [None] * n_data_files
    for i, fpath in enumerate(fpaths):
        # get file info
        fnames[i] = os.path.basename(fpath)
    return fpaths, fnames

# import os

def dir_tree(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

def get_result_fpath(result_dir, is_train=False):
    fpath = result_dir + f'results-train-{is_train}.pkl'
    if os.path.isfile(fpath):
        return fpath
    return None

def load_result(result_dir, is_train=False):
    fpath = get_result_fpath(result_dir, is_train=is_train)
    if fpath is None:
        return None
    return pickle_load(fpath)

def get_subj_ids(result_dir):
    result_dir_dir = os.path.dirname(os.path.dirname(result_dir))
    _, fnames = list_fnames(result_dir_dir, '*')
    subj_ids = sorted([int(fname.split('-')[1]) for fname in fnames])
    return subj_ids, len(subj_ids)

# helper funcs

def num_boundaries_per_event(log_cid_):
    model_ctx_bound_vec_list = []
    for i, log_cid_j in enumerate(log_cid_):
        model_ctx_bound_vec_list.append(context_to_bound_vec(log_cid_j))
    return np.array([np.sum(x) for x in model_ctx_bound_vec_list])

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

# are the context being used?
def compute_ctx_usage(n_ctx_over_time_fi, log_cid):
    n_ctx = int(n_ctx_over_time_fi[-1])+1
    ctx_usage = np.zeros((tvs.n_files, n_ctx))
    for i in range(tvs.n_files):
        # check for the ith video, what context were used
        ctx_used, counts = np.unique(log_cid[i], return_counts=True)
        for ctx_id_i, count_i in zip(ctx_used, counts):
            ctx_usage[i, int(ctx_id_i)] += count_i
        return ctx_usage


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


def vec_to_sec(vec):
    return [np.any(vec[t:t+3])==1 for t in range(0, len(vec), 3)]

def ctx_vec_to_sec(vec):
    return [Counter(vec[t:t+3]).most_common()[0][0] for t in range(0, len(vec), 3)]

# def compute_corr_with_perm(model_bounds_list, phuman_bounds_list, n_perms = 500):
#     r_perm = np.zeros(n_perms, )
#     all_event_segs = []
#     for i, model_bounds_list_i in enumerate(model_bounds_list):
#         all_event_segs.extend(get_event_segments(model_bounds_list_i))
#     for i in range(n_perms):
#         random.shuffle(all_event_segs)
#         r_perm[i], _ = get_point_biserial(
#             np.concatenate(all_event_segs), np.concatenate(phuman_bounds_list)
#         )
#     r, _ = get_point_biserial(
#         np.concatenate(model_bounds_list), np.concatenate(phuman_bounds_list)
#     )
#     return r_perm, r

def compute_corr_with_perm(model_bounds_list, phuman_bounds_list, n_perms = 500):
    r_perm = np.zeros(n_perms, )
    # get all event segments
    all_event_segs = []
    for mbci in model_bounds_list:
        mbci[-1] = 1
        event_bound_locs = np.where(mbci)[0]
        event_bound_locs = [0] + list(event_bound_locs)
        for j in range(1, len(event_bound_locs)):
            pad = 0 if j == 1 else 1
            event_ij = mbci[event_bound_locs[j-1]+pad : event_bound_locs[j]+1]
            assert event_ij[-1] == 1
            assert sum(event_ij) == 1
            all_event_segs.append(event_ij)
    # for each permutation
    for i in range(n_perms):
        # shuffle all events
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


def tree(dir_path: Path, level: int=-1, limit_to_directories: bool=False,
         length_limit: int=1000):
    space =  '    '
    branch = '│   '
    tee =    '├── '
    last =   '└── '
    """Given a directory Path object print a visual tree structure"""
    dir_path = Path(dir_path) # accept string coerceable to Path
    files = 0
    directories = 0
    def inner(dir_path: Path, prefix: str='', level=-1):
        nonlocal files, directories
        if not level:
            return # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
            contents = list(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(path, prefix=prefix+extension, level=level-1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1
    print(dir_path.name)
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        print(line)
    if next(iterator, None):
        print(f'... length_limit, {length_limit}, reached, counted:')
    print(f'\n{directories} directories' + (f', {files} files' if files else ''))


matplotlib.use('Agg')
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--update_freq', default=1, type=int)
parser.add_argument('--dim_hidden', default=16, type=int)
parser.add_argument('--dim_context', default=128, type=int)
parser.add_argument('--use_shortcut', default=1, type=float)
parser.add_argument('--gen_grad', default=5, type=float)
parser.add_argument('--ctx_wt', default=.5, type=float)
parser.add_argument('--stickiness', default=4, type=float)
parser.add_argument('--concentration', default=1, type=float)
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

# set param to create logdir
subj_id = 0
p = Parameters(
    dim_hidden = dim_hidden, dim_context = dim_context, ctx_wt = ctx_wt,
    stickiness = stickiness, gen_grad=gen_grad, lr = lr, update_freq = update_freq,
    subj_id = subj_id, lik_softmax_beta=lik_softmax_beta, concentration=concentration,
    try_reset_h = try_reset_h, use_shortcut=use_shortcut,
    pe_tracker_size = pe_tracker_size, match_tracker_size = match_tracker_size, n_pe_std= n_pe_std,
    log_root=log_root, exp_name=exp_name, verbose=False
)
assert os.path.isdir(p.result_dir), f'DIR NOT FOUND {p.result_dir}'

fig_dir = os.path.dirname(os.path.dirname(p.fig_dir))

def unpack_data(result_data_):
    log_cid_ = result_data_['log_cid']
    log_cid_fi_ = result_data_['log_cid_fi']
    log_cid_sc_ = result_data_['log_cid_sc']
    loss_by_events_ = result_data_['loss_by_events']
    log_reset_h_ = result_data_['log_reset_h']
    log_use_sc_ = result_data_['log_use_sc']
    log_pe_peak_ = result_data_['log_pe_peak']
    return log_cid_, log_cid_fi_, log_cid_sc_, loss_by_events_, log_reset_h_, log_use_sc_, log_pe_peak_

# get all subjs
subj_ids, n_subjs = get_subj_ids(p.result_dir)
event_id_list = tvs.valid_ids

# get actual n subjs
subj_ids_actual = []
for subj_id in subj_ids:
    p = Parameters(
        dim_hidden = dim_hidden, dim_context = dim_context, ctx_wt = ctx_wt,
        stickiness = stickiness, gen_grad=gen_grad, lr = lr, update_freq = update_freq,
        subj_id = subj_id, lik_softmax_beta=lik_softmax_beta, concentration=concentration,
        try_reset_h = try_reset_h, use_shortcut=use_shortcut,
        pe_tracker_size = pe_tracker_size, match_tracker_size = match_tracker_size, n_pe_std= n_pe_std,
        log_root=log_root, exp_name=exp_name, verbose=False
    )
    # load training and test data
    result_data_tr = load_result(p.result_dir, is_train=True)
    result_data_te = load_result(p.result_dir, is_train=False)
    if result_data_tr is not None and result_data_te is not None:
        subj_ids_actual.append(subj_id)
# update subj info
subj_ids, n_subjs = subj_ids_actual, len(subj_ids_actual)
assert n_subjs > 0, f'no data found, please check directory\n{p.result_dir}'
print(f'subj_ids: {subj_ids} found in \n{p.result_dir}')

# get human boundary baseline
human_r_crse, human_r_fine = hb.get_average_human_ceiling(tvs.valid_ids)

# prealloc
loss_mu_by_events_g = [None] * n_subjs
n_ctx_over_time_fi_g = [None] * n_subjs
event_len_te_g = [None] * n_subjs
n_bounds_g = [None] * n_subjs

mi_perm_g, mi_ob_g = [None] * n_subjs, [None] * n_subjs
r_perm_c_g, r_ob_c_g = [None] * n_subjs, [None] * n_subjs
r_perm_f_g, r_ob_f_g = [None] * n_subjs, [None] * n_subjs

sc_pnull_tr_mu_g, sc_pnull_te_mu_g = [None] * n_subjs, [None] * n_subjs
sc_acc_tr_mu_g, sc_acc_te_mu_g = [None] * n_subjs, [None] * n_subjs
cfmat_g = [None] * n_subjs

p_use_sc_te_g = [None] * n_subjs
p_use_sc_tr_g = [None] * n_subjs

mi_g = np.zeros((n_subjs, len(event_id_list)))
r_m_crse_g = np.zeros((n_subjs, len(event_id_list)))
r_m_fine_g = np.zeros((n_subjs, len(event_id_list)))
r_l_crse_g = np.zeros((n_subjs, len(event_id_list)))
r_l_fine_g = np.zeros((n_subjs, len(event_id_list)))

for subj_id, subj_id_ in enumerate(subj_ids):
    print(f'subj_id = {subj_id}')
    p = Parameters(
        dim_hidden = dim_hidden, dim_context = dim_context, ctx_wt = ctx_wt,
        stickiness = stickiness, gen_grad=gen_grad, lr = lr, update_freq = update_freq,
        subj_id = subj_id_, lik_softmax_beta=lik_softmax_beta, concentration=concentration,
        try_reset_h = try_reset_h, use_shortcut=use_shortcut,
        pe_tracker_size = pe_tracker_size, match_tracker_size = match_tracker_size, n_pe_std= n_pe_std,
        log_root=log_root, exp_name=exp_name, verbose=False
    )

    # load training and test data
    result_data_tr = load_result(p.result_dir, is_train=True)
    result_data_te = load_result(p.result_dir, is_train=False)
    if result_data_tr is None or result_data_te is None:
        print(f'no data for {subj_id}, continue')
        continue

    [log_cid_tr,log_cid_fi_tr,log_cid_sc_tr,loss_by_events_tr,log_reset_h_tr,log_use_sc_tr,log_pe_peak_tr] = unpack_data(result_data_tr)
    [log_cid_te,log_cid_fi_te,log_cid_sc_te,loss_by_events_te,log_reset_h_te,log_use_sc_te,log_pe_peak_te] = unpack_data(result_data_te)

    # combine log_cid
    log_cid_fi = log_cid_fi_tr + log_cid_fi_te
    log_cid = log_cid_tr + log_cid_te

    # collect data
    loss_mu_by_events_g[subj_id] = [np.stack(loss_event_i).mean() for loss_event_i in loss_by_events_te]

    n_ctx_over_time_fi_g[subj_id] = compute_n_ctx_over_time(log_cid_fi)

    event_len_te_g[subj_id] = get_event_len(log_cid_te)

    p_use_sc_te_g[subj_id], _ = compute_stats([np.mean(x) for x in log_use_sc_te])
    p_use_sc_tr_g[subj_id], _ = compute_stats([np.mean(x) for x in log_use_sc_tr])

    '''correlation with human boundaries'''

    model_bounds_c, model_bounds_f, chbs, fhbs = [], [], [], []
    sub_evn_label, mod_evn_label = [], []

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
        r_m_crse_g[subj_id, i], _ = get_point_biserial(model_ctx_bound_vec_c, chb)
        r_m_fine_g[subj_id, i], _ = get_point_biserial(model_ctx_bound_vec_f, fhb)
        r_l_crse_g[subj_id, i], _ = get_point_biserial(model_loss_bound_vec_c, chb)
        r_l_fine_g[subj_id, i], _ = get_point_biserial(model_loss_bound_vec_f, fhb)

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
        mi_g[subj_id, i] = mutual_info_score(sub_evn_label[i], mod_evn_label[i])

    mi_perm_g[subj_id], mi_ob_g[subj_id] = compute_mi_with_perm(mod_evn_label, sub_evn_label)
    r_perm_c_g[subj_id], r_ob_c_g[subj_id] = compute_corr_with_perm(model_bounds_c, chbs)
    r_perm_f_g[subj_id], r_ob_f_g[subj_id] = compute_corr_with_perm(model_bounds_f, fhbs)

    n_bounds_g[subj_id] = [np.sum(x) for x in model_bounds_c]

    # compute shortcut accuracy over time
    sc_acc_tr, sc_acc_te = [], []
    for i, (log_cid_fi_tr_i, log_cid_sc_tr_i) in enumerate(zip(log_cid_fi_tr, log_cid_sc_tr)):
        # if i == 0: break
        sc_acc_tr.append(np.mean(log_cid_fi_tr_i == log_cid_sc_tr_i))
    for i, (log_cid_fi_te_i, log_cid_sc_te_i) in enumerate(zip(log_cid_fi_te, log_cid_sc_te)):
        sc_acc_te.append(np.mean(log_cid_fi_te_i == log_cid_sc_te_i))
    sc_acc_tr_mu_g[subj_id] = np.nanmean(sc_acc_tr)
    sc_acc_te_mu_g[subj_id] = np.nanmean(sc_acc_te)

    # compute precent null response
    percent_sc_null_tr = [np.mean(x == -1) for x in log_cid_sc_tr]
    percent_sc_null_te = [np.mean(x == -1) for x in log_cid_sc_te]
    sc_pnull_tr_mu_g[subj_id] = np.nanmean(percent_sc_null_tr)
    sc_pnull_te_mu_g[subj_id] = np.nanmean(percent_sc_null_te)

    # confusion matrix shortcut vs full inf
    log_cid_sc_cat = np.concatenate(log_cid_sc_tr+log_cid_sc_te)
    log_cid_fi_cat = np.concatenate(log_cid_fi_tr+log_cid_fi_te)
    log_cid_sc_cat[np.isnan(log_cid_sc_cat)] = -1
    log_cid_sc_cat_nonull = log_cid_sc_cat[log_cid_sc_cat != -1]
    log_cid_fi_cat_nonull = log_cid_fi_cat[log_cid_sc_cat != -1]
    cfmat_g[subj_id] = confusion_matrix(log_cid_fi_cat_nonull, log_cid_sc_cat_nonull, normalize='true')


# group level mean and se
mi_g_mu, mi_g_se = compute_stats(np.mean(mi_g,axis=1), omitnan=True)
mi_ob_g_mu, mi_ob_g_se = compute_stats(mi_ob_g, omitnan=True)

r_m_crse_g_mu, r_m_crse_g_se = compute_stats(np.mean(r_m_crse_g, axis=1), omitnan=True)
r_m_fine_g_mu, r_m_fine_g_se = compute_stats(np.mean(r_m_fine_g, axis=1), omitnan=True)

r_ob_c_g_mu, r_ob_c_g_se = compute_stats(r_ob_c_g, omitnan=True)
r_ob_f_g_mu, r_ob_f_g_se = compute_stats(r_ob_f_g, omitnan=True)

sc_acc_tr_g_mu, sc_acc_tr_g_se = compute_stats(sc_acc_tr_mu_g, omitnan=True)
sc_acc_te_g_mu, sc_acc_te_g_se = compute_stats(sc_acc_te_mu_g, omitnan=True)

sc_pnull_tr_g_mu, sc_pnull_tr_g_se = compute_stats(sc_pnull_tr_mu_g, omitnan=True)
sc_pnull_te_g_mu, sc_pnull_te_g_se = compute_stats(sc_pnull_te_mu_g, omitnan=True)

p_use_sc_te_g_mu, p_use_sc_te_g_se = compute_stats(p_use_sc_te_g, omitnan=True)
p_use_sc_tr_g_mu, p_use_sc_tr_g_se = compute_stats(p_use_sc_tr_g, omitnan=True)

'''plot the data '''
loss_mu, loss_se = compute_stats(np.mean(loss_mu_by_events_g,axis=0),axis=0)

# plot loss by valid event
f, ax = plt.subplots(1,1, figsize=(2,4))
ax.bar(0, loss_mu, yerr=loss_se, align='center', ecolor='black', capsize=10)
ax.set_title('%.3f' % loss_mu)
ax.set_xlabel(None)
ax.set_xticks([])
ax.set_ylabel('MSE loss')
sns.despine()
fig_path = os.path.join(fig_dir, f'loss.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


mean_n_ctx_at_the_end = np.mean([x[-1] for x in n_ctx_over_time_fi_g])

f, ax = plt.subplots(1,1, figsize=(5,4))
for x in n_ctx_over_time_fi_g:
    ax.plot(x)
ax.set_xlabel('training video id')
ax.set_ylabel('# of LCs inferred')
ax.set_title(f'mean # of LCs at the end = %.2f' % mean_n_ctx_at_the_end)
ax.axvline(len(log_cid_fi_tr), label='start testing', ls='--', color='grey')
ax.legend()
sns.despine()
fig_path = os.path.join(fig_dir, f'n-contexts-ovt.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


f, ax = plot_event_len_distribution(np.concatenate(event_len_te_g))
fig_path = os.path.join(fig_dir, f'event-length.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


f, ax = plt.subplots(1,1, figsize=(7,4))
ax.hist(np.concatenate(n_bounds_g))
ax.set_title('mean = %.2f' % (np.nanmean(np.concatenate(n_bounds_g))))
ax.set_xlabel('# boundaries')
ax.set_ylabel('Frequency')
sns.despine()
fig_path = os.path.join(fig_dir, f'n-event-bounds.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


f, ax = plt.subplots(1,1, figsize=(7, 4))
# sns.kdeplot(np.concatenate(mi_perm_g), label='permutation')
sns.kdeplot(np.mean(mi_perm_g, axis=1), label='permutation')
ax.axvline(mi_ob_g_mu, ls='--', color='k', label='observed = %.2f' % mi_ob_g_mu)
ax.set_title('MI model vs. truth')
ax.set_xlabel('Point biserial correlation')
ax.legend()
sns.despine()
fig_path = os.path.join(fig_dir, f'mi-obs.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


f, ax = plt.subplots(1,1, figsize=(7, 4))
# sns.kdeplot(np.concatenate(mi_perm_g), label='permutation')
sns.kdeplot(np.mean(mi_perm_g, axis=1), label='permutation')
ax.axvline(mi_g_mu, ls='--', color='k', label='observed = %.2f' % mi_ob_g_mu)
ax.set_title('MI model vs. truth')
ax.set_xlabel('Point biserial correlation')
ax.legend()
sns.despine()
fig_path = os.path.join(fig_dir, f'mi-inloop.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')



f, ax = plt.subplots(1, 1, figsize=(7, 4), sharey=True)
sns.kdeplot(np.mean(r_perm_c_g,axis=1), label='permutation', ax=ax)
# ax.axvline(r_m_crse_g_mu, ls='--', color='k', label='observed = %.2f' % r_m_crse_g_mu)
ax.axvline(r_ob_c_g_mu, ls='--', color='k', label='observed = %.2f' % r_ob_c_g_mu)
ax.axvline(np.nanmean(human_r_crse), ls='--', color='red', label='human baseline = %.2f' % np.nanmean(human_r_crse))
ax.set_title('model boundaries vs. corase human boundaries')
ax.set_xlabel('Point biserial correlation')
# ax.set_xlim([0, None])
ax.legend()
sns.despine()
f.tight_layout()
fig_path = os.path.join(fig_dir, f'bound-r-c.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


f, ax = plt.subplots(1, 1, figsize=(7, 4), sharey=True)
sns.kdeplot(np.mean(r_perm_f_g,axis=1), label='permutation', ax=ax)
# ax.axvline(r_m_fine_g_mu, ls='--', color='k', label='observed = %.2f' % r_m_fine_g_mu)
ax.axvline(r_ob_f_g_mu, ls='--', color='k', label='observed = %.2f' % r_ob_f_g_mu)
ax.axvline(np.nanmean(human_r_fine), ls='--', color='red', label='human baseline = %.2f' % np.nanmean(human_r_fine))
ax.set_title('model boundaries vs. fine human boundaries')
ax.set_xlabel('Point biserial correlation')
# ax.set_xlim([0, None])
ax.legend()
sns.despine()
f.tight_layout()
fig_path = os.path.join(fig_dir, f'bound-r-f.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')



f, ax = plt.subplots(1, 1, figsize=(7, 4), sharey=True)
sns.kdeplot(np.mean(r_perm_c_g,axis=1), label='permutation', ax=ax)
ax.axvline(r_m_crse_g_mu, ls='--', color='k', label='observed = %.2f' % r_m_crse_g_mu)
# ax.axvline(r_ob_c_g_mu, ls='--', color='k', label='observed = %.2f' % r_ob_c_g_mu)
ax.axvline(np.nanmean(human_r_crse), ls='--', color='red', label='human baseline = %.2f' % np.nanmean(human_r_crse))
ax.set_title('model boundaries vs. corase human boundaries')
ax.set_xlabel('Point biserial correlation')
# ax.set_xlim([0, None])
ax.legend()
sns.despine()
f.tight_layout()
fig_path = os.path.join(fig_dir, f'bound-r-c-inloop.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


f, ax = plt.subplots(1, 1, figsize=(7, 4), sharey=True)
sns.kdeplot(np.mean(r_perm_f_g,axis=1), label='permutation', ax=ax)
ax.axvline(r_m_fine_g_mu, ls='--', color='k', label='observed = %.2f' % r_m_fine_g_mu)
# ax.axvline(r_ob_f_g_mu, ls='--', color='k', label='observed = %.2f' % r_ob_f_g_mu)
ax.axvline(np.nanmean(human_r_fine), ls='--', color='red', label='human baseline = %.2f' % np.nanmean(human_r_fine))
ax.set_title('model boundaries vs. fine human boundaries')
ax.set_xlabel('Point biserial correlation')
# ax.set_xlim([0, None])
ax.legend()
sns.despine()
f.tight_layout()
fig_path = os.path.join(fig_dir, f'bound-r-f-inloop.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

width = .7
f, ax = plt.subplots(1,1, figsize=(4.5, 4))
xticks = range(2)
ax.bar(x=xticks, width=width, height=[sc_acc_tr_g_mu, sc_acc_te_g_mu], yerr=[sc_acc_tr_g_se, sc_acc_te_g_se], label='match with full inference')
ax.bar(x=xticks, width=width, height=[sc_pnull_tr_g_mu, sc_pnull_te_g_mu], yerr=[sc_pnull_tr_g_se, sc_pnull_te_g_se], bottom=[sc_acc_tr_g_mu, sc_acc_te_g_mu], label='null response')
ax.bar(x=xticks, width=width, height=[1-sc_acc_tr_g_mu-sc_pnull_tr_g_mu, 1-sc_acc_te_g_mu-sc_pnull_te_g_mu], bottom=[sc_acc_tr_g_mu+sc_pnull_tr_g_mu, sc_acc_te_g_mu+sc_pnull_te_g_mu], label='mismatch', color=sns.color_palette()[3])
ax.set_xticks(xticks)
ax.set_xticklabels(['train', 'validation'])
ax.set_title('short cut performance')
ax.set_ylabel('%')
ax.set_ylim([0,1])
# ax.legend()
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
sns.despine()
fig_path = os.path.join(fig_dir, f'sc-acc.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')



def plot_dabest(perm_vals, observed_vals, ax, ylabel, human_baseline=None, use_slope_plot=True, float_contrast=False):
    assert len(perm_vals) == len(observed_vals)
    if use_slope_plot:
        paired=True
        id_col='ids'
    else:
        paired=False
        id_col=None

    n_data = len(perm_vals)
    data_dict = {
        'permutation': perm_vals, 'observed': observed_vals,
    }

    df = pd.DataFrame(data_dict)
    df['ids'] = np.arange(n_data)

    # Load the data into dabest
    dabest_data = dabest.load(
        data=df, idx=list(data_dict.keys()),
        paired=paired, id_col=id_col,
    )
    dabest_data.mean_diff.plot(swarm_label=ylabel, ax=ax, float_contrast=float_contrast)

    if human_baseline is not None:
        ax.axhline(human_baseline, ls='--', color='red', label='human baseline')
        ax.set_ylim([None, human_baseline + human_baseline *.1])

    return f, ax

f, ax = plt.subplots(1,1, figsize=(3, 8))
ax = plot_dabest(np.mean(mi_perm_g,axis=1), mi_ob_g, ax=ax, ylabel='mutual infomation', human_baseline=None, use_slope_plot=True)
fig_path = os.path.join(fig_dir, f'slope-mi.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


f, ax = plt.subplots(1,1, figsize=(3, 8))
ax = plot_dabest(np.mean(r_perm_c_g,axis=1), r_ob_c_g, ax=ax, ylabel='point biserial correlation', human_baseline=np.nanmean(human_r_crse))
fig_path = os.path.join(fig_dir, f'slope-r-c.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

f, ax = plt.subplots(1,1, figsize=(3, 8))
ax = plot_dabest(np.mean(r_perm_f_g,axis=1), r_ob_f_g, ax=ax, ylabel='point biserial correlation', human_baseline=np.nanmean(human_r_fine))
fig_path = os.path.join(fig_dir, f'slope-r-f.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


def violin_obs_perm_diff(perm_vals, obs_vals, ax=ax):
    n_std_to_mu = np.zeros(n_subjs, )
    for i in range(n_subjs):
        mu, sd = norm.fit(perm_vals[i])
        n_std_to_mu[i] = (obs_vals[i] - mu) / sd
    sns.violinplot(n_std_to_mu,ax=ax)
    ax.set_xticks([])
    ax.set_ylabel('# of stds')
    ax.set_title('distance to mean of \nthe permutation distribution')
    ax.axhline(0, ls='--', color='grey')
    sns.despine()
    return ax

f, ax = plt.subplots(1,1, figsize=(3, 4))
ax = violin_obs_perm_diff(mi_perm_g, mi_ob_g, ax=ax)
fig_path = os.path.join(fig_dir, f'n-std-perm-mi.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

f, ax = plt.subplots(1,1, figsize=(3, 4))
ax = violin_obs_perm_diff(r_perm_c_g, r_ob_c_g, ax=ax)
fig_path = os.path.join(fig_dir, f'n-std-r-c.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')

f, ax = plt.subplots(1,1, figsize=(3, 4))
ax = violin_obs_perm_diff(r_perm_f_g, r_ob_f_g, ax=ax)
fig_path = os.path.join(fig_dir, f'n-std-r-f.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')


width = .7
f, ax = plt.subplots(1,1, figsize=(4.5, 4))
xticks = range(2)
ax.bar(x=xticks, width=width, height=[p_use_sc_te_g_mu, p_use_sc_tr_g_mu], yerr=[p_use_sc_te_g_se, p_use_sc_tr_g_se])
ax.set_xticks(xticks)
ax.set_xticklabels(['train', 'validation'])
ax.set_title('percent shortcut activation')
ax.set_ylabel('%')
ax.set_ylim([0,1])
# ax.legend()
sns.despine()
fig_path = os.path.join(p.fig_dir, f'sc-p-act.png')
f.savefig(fig_path, dpi=100, bbox_inches='tight')
