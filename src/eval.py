'''eval trained models on the valid files'''
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
import itertools
from tqdm import tqdm
# from model import CGRU as Agent
from model import CGRU_v2 as Agent
from model import SimpleContext
from utils import to_np, to_pth, split_video_id, context_to_bound_vec, loss_to_bound_vec, load_ckpt, padded_corr
from utils import EventLabel, TrainValidSplit, DataLoader, Parameters, HumanBondaries
from scipy.stats import pointbiserialr, pearsonr


sns.set(style='white', palette='colorblind', context='talk')
log_root = '../log'

'''set params, for example:
subj_id = 99
lr = 1e-3
update_freq = 100
dim_hidden = 16
dim_context = 32
ctx_wt = .5
penalty_new_context = .5
stickiness = .5
'''

subj_id_list = [0, 1, 2]
lr_list = [1e-3, 1e-4]
# update_freq_list = [2,4,8]
update_freq_list = [8]
dim_hidden_list = [16]
dim_context_list = [64, 256]
ctx_wt_list = [.5]
penalty_new_context_list = [.1, .5, 1]
stickiness_list = [.1, .5, 1]

param_combos = itertools.product(
    subj_id_list,
    lr_list,
    update_freq_list,
    dim_hidden_list,
    dim_context_list,
    ctx_wt_list,
    penalty_new_context_list,
    stickiness_list
)

for params in param_combos:
    print(params)
    (subj_id, lr, update_freq, dim_hidden, dim_context, ctx_wt, penalty_new_context,stickiness) = params

    # # training param
    # subj_id = 0
    # lr = 1e-3
    # update_freq = 8
    # dim_hidden = 16
    # dim_context = 256
    # ctx_wt = .5
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
    sc = SimpleContext(p.dim_context, p.penalty_new_context, p.stickiness)
    # get all ckpt files
    # ckpt_fpaths, ckpt_fnames = list_fnames(p.log_dir, 'ckpt*')
    # load ckpt
    agent, optimizer, sc_dict = load_ckpt(tvs.n_train_files, p.log_dir, agent, optimizer)
    # agent, optimizer, sc_dict = load_ckpt(10, p.log_dir, agent, optimizer)
    sc.init_from_dict(sc_dict)

    if agent is None:
        continue

    def run_model(event_id_list, p, save_weights=True, learning=True, save_freq=10):
        loss_by_events = [[] for _ in range(len(event_id_list))]
        log_cid = [[] for _ in range(len(event_id_list))]

        for i, event_id in enumerate(event_id_list):
            # save data for every other k epochs
            if save_weights and i % save_freq == 0:
                save_ckpt(i, p.log_dir, agent, optimizer, verbose=True)
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
            save_ckpt(len(event_id_list), p.log_dir, agent, optimizer, verbose=True)
        print('done')
        return log_cid, loss_by_events


    '''evaluate loss on the validation set'''
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
    for i in range(tvs.n_valid_files):
        model_ctx_bound_vec = context_to_bound_vec(log_cid[i])
        p_b_c = hb.get_bound_prob(tvs.valid_ids[i], 'coarse')
        p_b_f = hb.get_bound_prob(tvs.valid_ids[i], 'fine')
        r_crse[i], _ = padded_corr(model_ctx_bound_vec, p_b_c, shift=False)
        r_fine[i], _ = padded_corr(model_ctx_bound_vec, p_b_f, shift=False)

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
    fig_path = os.path.join(p.fig_dir, f'final-r-model-vs-human-bounds.png')
    f.savefig(fig_path, dpi=100)

    r_crse = np.zeros(len(log_cid),)
    r_fine = np.zeros(len(log_cid),)
    for i in range(tvs.n_valid_files):
        model_ctx_bound_vec = context_to_bound_vec(log_cid[i])
        p_b_c = hb.get_bound_prob(tvs.valid_ids[i], 'coarse')
        p_b_f = hb.get_bound_prob(tvs.valid_ids[i], 'fine')
        r_crse[i], _ = padded_corr(to_np(torch.stack(loss_by_events[i])), p_b_c, corr_f=pearsonr)
        r_fine[i], _ = padded_corr(to_np(torch.stack(loss_by_events[i])), p_b_f, corr_f=pearsonr)

    f, axes = plt.subplots(2, 1, figsize=(5,7), sharex=True)
    sns.violinplot(r_crse, ax=axes[0])
    sns.violinplot(r_fine, ax=axes[1])
    for ax in axes:
        ax.axvline(0, ls='--', c='grey', label='0', zorder=-1)
        ax.legend()
    axes[0].set_title(f'mean r = %.3f' % (r_crse.mean()))
    axes[1].set_title(f'mean r = %.3f' % (r_fine.mean()))
    axes[1].set_xlabel('Correlation')
    axes[0].set_ylabel('Coarse')
    axes[1].set_ylabel('Fine')
    f.tight_layout()
    sns.despine()
    fig_path = os.path.join(p.fig_dir, f'final-r-loss-vs-human-bounds.png')
    f.savefig(fig_path, dpi=100)
