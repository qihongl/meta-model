'''train model on all META training data file and eval on the valid files'''
import os
import glob
import time
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from model import CGRU as Agent
from model import SimpleContext
from utils import to_np, to_pth, split_video_id, context_to_bound_vec
from utils import EventLabel, TrainValidSplit, DataLoader, Parameters, HumanBondaries
from utils import padded_pointbiserialr
sns.set(style='white', palette='colorblind', context='talk')

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

'''params for the model'''
# training param
lr = 1e-3
# model param
dim_input = dim_output = 30
dim_hidden = 16
dim_context = 128
ctx_wt = .5
# ctx_wt = 0
penalty_new_context = .1
stickiness = .1

# init util objects
dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()
hb = HumanBondaries()
p = Parameters(
    dim_hidden = dim_hidden,
    dim_context = dim_context,
    ctx_wt = ctx_wt,
    penalty_new_context = penalty_new_context,
    stickiness = stickiness,
    lr = lr,
)
# init model
agent = Agent(
    p.dim_input, p.dim_hidden, p.dim_output,
    ctx_wt=p.ctx_wt, context_dim=p.dim_context
)
# criterion = nn.MSELoss()
optimizer = torch.optim.Adam(agent.parameters(), lr=p.lr)
# context management
sc = SimpleContext(p.dim_context)
c_id, c_vec = sc.reset_context()

'''train the model'''
event_id_list = tvs.train_ids
loss_by_events = [[] for _ in range(tvs.n_train_files)]
log_cid = [[] for _ in range(tvs.n_train_files)]
for i, event_id in enumerate(event_id_list):
    print(f'Learning event {i} / {tvs.n_train_files} - {event_id}')
    actor_id, chapter_id, run_id = split_video_id(event_id)
    X = dl.get_data(event_id)
    T = len(X)-1
    # prealloc
    log_cid_i = np.zeros(T, )
    # run the model over time
    loss = 0
    h_t = agent.get_zero_states()
    for t in tqdm(range(T)):
        # context - full inference
        pe = agent.try_all_contexts(X[t+1], X[t], h_t, sc.context)
        pe[0] = pe[0] + p.penalty_new_context
        pe[sc.prev_cluster_id] = pe[sc.prev_cluster_id] - p.stickiness
        log_cid_i[t] = sc.assign_context(-pe, verbose=1)
        c_vec = sc.context[int(log_cid_i[t])]
        # forward
        [y_t_hat, h_t], cache = agent.forward(X[t], h_t, to_pth(c_vec))
        # record losses
        loss_it = agent.criterion(torch.squeeze(y_t_hat), X[t+1])
        loss += loss_it
        loss_by_events[i].append(loss_it)
    # update weights
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    log_cid[i] = log_cid_i
print('done')

'''evaluate loss on the validation set'''
event_id_list = tvs.valid_ids
loss_by_events = [[] for _ in range(tvs.n_valid_files)]
log_cid = [[] for _ in range(tvs.n_valid_files)]
for i, event_id in enumerate(event_id_list):
    actor_id, chapter_id, run_id = split_video_id(event_id)
    X = dl.get_data(event_id)
    T = len(X)-1
    log_cid_i = np.zeros(T, )
    loss = 0
    h_t = agent.get_zero_states()
    for t in tqdm(range(T)):
        # context - full inference
        pe = agent.try_all_contexts(X[t+1], X[t], h_t, sc.context)
        pe[0] = pe[0] + penalty_new_context
        pe[sc.prev_cluster_id] = pe[sc.prev_cluster_id] - p.stickiness
        log_cid_i[t] = sc.assign_context(-pe, verbose=1)
        c_vec = sc.context[int(log_cid_i[t])]
        # forward
        [y_t_hat, h_t], cache = agent.forward(X[t], h_t, to_pth(c_vec))
        # record losses
        loss_it = agent.criterion(torch.squeeze(y_t_hat), X[t+1])
        loss += loss_it
        loss_by_events[i].append(loss_it)
    log_cid[i] = log_cid_i

'''plot the data '''
loss_mu_by_events = [torch.stack(loss_event_i).mean() for loss_event_i in loss_by_events]
f, ax = plt.subplots(1,1, figsize=(7,4))
ax.plot(loss_mu_by_events)
ax.set_title('%.3f' % torch.stack(loss_mu_by_events).mean())
ax.set_xlabel('validation video id')
ax.set_ylabel('loss')
sns.despine()




for i, loss_by_events_i in enumerate(loss_by_events):
    if i > 3: break
    f, ax = plt.subplots(1,1, figsize=(10,3))
    ax.plot(loss_by_events_i)
    ax.set_title(tvs.valid_ids[i])
    ax.set_xlabel('time')
    ax.set_ylabel('loss')
    event_bound_times, event_bound_vec = evlab.get_bounds(tvs.valid_ids[i])
    for eb in event_bound_times:
        ax.axvline(eb, ls='--', color='grey')
    sns.despine()



for i, log_cid_i in enumerate(log_cid):
    if i > 3: break

    event_bound_times, event_bound_vec = evlab.get_bounds(tvs.valid_ids[i])
    f, ax = plt.subplots(1,1, figsize=(10,3))
    ax.plot(log_cid_i)
    # ax.set_title('%.3f' % torch.stack(loss_mu_by_events).mean())
    ax.set_title(tvs.valid_ids[i])
    ax.set_xlabel('time')
    ax.set_ylabel('Context')

    for eb in event_bound_times:
        ax.axvline(eb, ls='--', color='grey')
    sns.despine()


r_crse = np.zeros(len(log_cid),)
r_fine = np.zeros(len(log_cid),)
p_crse = np.zeros(len(log_cid),)
p_fine = np.zeros(len(log_cid),)
for i in range(tvs.n_valid_files):
    model_loss_bound_vec = loss_to_bound_vec(loss_by_events[i])
    model_ctx_bound_vec = context_to_bound_vec(log_cid[i])
    p_b_c = hb.get_bound_prob(event_id, 'coarse')
    p_b_f = hb.get_bound_prob(event_id, 'fine')
    r_crse[i], p_crse[i] = padded_pointbiserialr(model_loss_bound_vec, p_b_c)
    r_fine[i], p_fine[i] = padded_pointbiserialr(model_loss_bound_vec, p_b_f)
    # r_crse[i], p_crse[i] = padded_pointbiserialr(model_ctx_bound_vec, p_b_c)
    # r_fine[i], p_fine[i] = padded_pointbiserialr(model_ctx_bound_vec, p_b_f)



f, ax = plt.subplots(1,1, figsize=(5,4))
sns.violinplot(r_crse, ax=ax)
sns.despine()
ax.set_title(f'mean r = %.3f' % (r_crse.mean()))
ax.set_xlabel('r')
f.tight_layout()


f, ax = plt.subplots(1,1, figsize=(5,4))
sns.violinplot(r_fine, ax=ax)
ax.set_title(f'mean r = %.3f' % (r_fine.mean()))
ax.set_xlabel('r')
sns.despine()
f.tight_layout()


f, ax = plt.subplots(1,1, figsize=(5,4))
sns.histplot(p_crse, ax=ax)
sns.despine()
ax.set_title(f'mean p = %.3f' % (p_crse.mean()))
ax.set_xlabel('p')
f.tight_layout()


f, ax = plt.subplots(1,1, figsize=(5,4))
sns.histplot(p_fine, ax=ax)
ax.set_title(f'mean p = %.3f' % (p_fine.mean()))
ax.set_xlabel('p')
sns.despine()
f.tight_layout()
