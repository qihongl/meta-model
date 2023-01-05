import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import padded_corr
from utils import EventLabel, TrainValidSplit, HumanBondaries, DataLoader
sns.set(style='white', palette='colorblind', context='talk')

tvs = TrainValidSplit()
evlab = EventLabel()
hb = HumanBondaries()
dl = DataLoader()

# choose dataset
event_id_list = tvs.train_ids
# event_id_list = tvs.valid_ids

# loop over events
time_truth = np.zeros(len(event_id_list), )
time_X = np.zeros(len(event_id_list), )
bound_times_min = np.zeros(len(event_id_list), )
bound_times_max = np.zeros(len(event_id_list), )
t_f1 = np.zeros(len(event_id_list), )
for i, event_id in enumerate(event_id_list):

    event_st_t, event_ed_t = evlab.get_start_end_times(event_id)
    X_i, t_f1[i] = dl.get_data(event_id, get_t_frame1=True)
    # t_f1[i]
    bound_times = hb.get_bound_times(event_id, to_sec=True)
    bound_times_min[i], bound_times_max[i] = np.min(bound_times), np.max(bound_times)

    time_truth[i] = event_ed_t - event_st_t
    time_X[i] = len(X_i)


maxsmin_bound_time = np.array(bound_times_max-bound_times_min)
sort = np.argsort(time_truth)


f, ax = plt.subplots(1,1, figsize=(13, 10))
ax.plot(time_truth[sort], label = 'end time - start time from event labels')
ax.plot(time_X[sort], label = 'len(x_pca)')
ax.plot(maxsmin_bound_time[sort], ls = '--', label = 'max - min (human boundary time * 3)')
# ax.plot(time_truth, label = 'end time - start time from event labels')
# ax.plot(time_X, label = 'len(x_pca)')
# ax.plot(, ls = '--', label = 'max - min (boundary time)')
# ax.plot(bound_times_max, ls = '--', label = 'max(boundary time)')
# ax.plot(bound_times_min, ls = '--', label = 'min(boundary time)')
ax.set_ylabel('video duration')
ax.set_xlabel('video id (sorted by video length)')
ax.legend()
sns.despine()
