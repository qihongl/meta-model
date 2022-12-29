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
from utils import to_np, to_pth
from utils import EventLabel, TrainValidSplit, DataLoader

sns.set(style='white', palette='colorblind', context='talk')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

'''params for the model'''
# training param
n_epochs = 100
learning_rate = 1e-3
# model param
dim_input = dim_output = 30
dim_hidden = 16
dim_context = 16
ctx_wt = .5
# ctx_wt = 0

# init model
agent = Agent(dim_input, dim_hidden, dim_output, ctx_wt=ctx_wt, context_dim=dim_context)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

'''train the model'''
event_id_list = tvs.train_ids
loss_by_events = [[] for _ in range(tvs.n_train_files)]
for i, event_id in enumerate(event_id_list):
    X = dl.get_data(event_id)
    loss = 0
    h_t = agent.get_zero_states()
    context_t = agent.get_zero_states()
    for t in tqdm(range(len(X)-1)):
        [y_t_hat, h_t], cache = agent.forward(X[t], h_t, context_t)
        # record losses
        loss_it = criterion(torch.squeeze(y_t_hat), X[t+1])
        loss += loss_it
        loss_by_events[i].append(loss_it)
    # update weights
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

print('done')

# evaluate loss
event_id_list = tvs.valid_ids
loss_by_events = [[] for _ in range(tvs.n_valid_files)]
for i, event_id in enumerate(event_id_list):
    X = dl.get_data(event_id)
    loss = 0
    h_t = agent.get_zero_states()
    context_t = agent.get_zero_states()
    for t in tqdm(range(len(X)-1)):
        [y_t_hat, h_t], cache = agent.forward(X[t], h_t, context_t)
        # record losses
        loss_it = criterion(torch.squeeze(y_t_hat), X[t+1])
        loss += loss_it
        loss_by_events[i].append(loss_it)

loss_mu_by_events = [torch.stack(loss_event_i).mean() for loss_event_i in loss_by_events]
f, ax = plt.subplots(1,1, figsize=(7,4))
ax.plot(loss_mu_by_events)
ax.set_title('%.3f' % torch.stack(loss_mu_by_events).mean())
sns.despine()
