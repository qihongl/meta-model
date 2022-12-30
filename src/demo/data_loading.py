import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import ID2CHAPTER, split_video_id, padded_pointbiserialr
from utils import EventLabel, TrainValidSplit, DataLoader, HumanBondaries
sns.set(style='white', palette='colorblind', context='talk')

dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()
hb = HumanBondaries()

# choose dataset
event_id_list = tvs.train_ids
# loop over events
for i, event_id in enumerate(event_id_list):
    print(f'Event {i} / {tvs.n_train_files} - {event_id}')
    if i > 5: break

    p_b_c = hb.get_bound_prob(event_id, 'coarse')
    p_b_f = hb.get_bound_prob(event_id, 'fine')

    # load data
    X = dl.get_data(event_id)
    # get basic info
    actor_id, chapter_id, run_id = split_video_id(event_id)

    T = len(X)-1
    # get ground truth boundaries
    event_bound_times, event_bound_vec = evlab.get_bounds(tvs.train_ids[i])

    c_r, c_p = padded_pointbiserialr(event_bound_vec, p_b_c)
    f_r, f_p = padded_pointbiserialr(event_bound_vec, p_b_f)

    '''plot this event'''
    alpha = .5
    f, ax = plt.subplots(1,1, figsize=(10,3.5))
    ax.set_xlabel('Time unit (X)')
    # ax.set_title(f'{event_id}')
    ax.set_title(f'{event_id} (actor: {actor_id}, chapter: {ID2CHAPTER[chapter_id]}, run: {run_id}) \n correlation with corase/fine boundaries = %.3f / %.3f' % (c_r, f_r))

    for j, eb in enumerate(event_bound_times):
        label = 'true event bound' if j == 0 else None
        ax.axvline(eb, ls='--', color='grey', label=label)
    for j, eb in enumerate(event_bound_times):
        label = '# time points in X' if j == 0 else None
        ax.axvline(T, ls='--', color='red', label=label)
    ax.plot(p_b_c, label='coarse bounds', alpha=alpha)
    ax.plot(p_b_f, label='fine bounds', alpha=alpha)
    ax.legend()
    sns.despine()
