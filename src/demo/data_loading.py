'''
compare 1) len(X), 2) human boundaries data and 3) event labels

the time data of len(X) and humam boundaries need to match
- if range(human boundaries) > range(X), hb need to be removed
- if range(human boundaries) < range(X),


'''
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import ID2CHAPTER, split_video_id, padded_corr
from utils import EventLabel, TrainValidSplit, DataLoader, HumanBondaries
sns.set(style='white', palette='colorblind', context='talk')

dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()
hb = HumanBondaries()

fhb = hb.get_precomputed_hb('fine')
chb = hb.get_precomputed_hb('coarse')

# choose dataset
event_id_list = tvs.valid_ids
t_f1 = np.zeros(len(event_id_list))

c_r = np.zeros(len(event_id_list), )
f_r = np.zeros(len(event_id_list), )
# loop over events
for i, event_id in enumerate(event_id_list):
    print(f'Event {i} / {len(event_id_list)} - {event_id}')

    p_b_c = hb.get_bound_prob(event_id, 'coarse')
    p_b_f = hb.get_bound_prob(event_id, 'fine')

    # load data
    X, t_f1 = dl.get_data(event_id, get_t_frame1=True)
    # get basic info
    actor_id, chapter_id, run_id = split_video_id(event_id)
    X_start_sec = np.round(t_f1)
    X_end_sec = T = len(X) - 1 + X_start_sec

    # get ground truth boundaries
    event_bound_times, event_bound_vec = evlab.get_bounds(event_id_list[i])

    c_r[i], _ = padded_corr(event_bound_vec, p_b_c, shift=False)
    f_r[i], _ = padded_corr(event_bound_vec, p_b_f, shift=False)

    '''plot this event'''
    alpha = .5
    f, ax = plt.subplots(1,1, figsize=(12,4))
    ax.set_xlabel('Time unit (X)')
    # ax.set_title(f'{event_id}')
    ax.set_title(f'{event_id} (actor: {actor_id}, chapter: {ID2CHAPTER[chapter_id]}, run: {run_id}) \n correlation with corase/fine boundaries = %.3f / %.3f' % (c_r[i], f_r[i]))

    for j, eb in enumerate(event_bound_times):
        label = 'true event bound' if j == 0 else None
        ax.axvline(eb, ls='--', color='grey', label=label)
    for j, eb in enumerate(event_bound_times):
        label = 'time range in X' if j == 0 else None
        ax.axvline(X_start_sec, ls='--', color='red')
        ax.axvline(X_end_sec, ls='--', color='red', label=label)
    ax.plot(p_b_c, label='coarse bounds', alpha=alpha)
    ax.plot(p_b_f, label='fine bounds', alpha=alpha)
    ax.legend()
    sns.despine()

'''
problematic events
1.3.3
6.1.5
'''

f, axes = plt.subplots(2, 1, figsize=(5,7), sharex=True)
sns.violinplot(c_r, ax=axes[0])
sns.violinplot(f_r, ax=axes[1])
for ax in axes:
    ax.axvline(0, ls='--', c='grey', label='0', zorder=-1)
    ax.legend()
axes[0].set_title(f'mean r = %.3f' % (c_r.mean()))
axes[1].set_title(f'mean r = %.3f' % (f_r.mean()))
axes[1].set_xlabel('Point biserial correlation')
axes[0].set_ylabel('Coarse')
axes[1].set_ylabel('Fine')
f.tight_layout()
sns.despine()
