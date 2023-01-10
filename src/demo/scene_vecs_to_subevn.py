'''
check if we can cluster the scene vectors
in particular, can we derive good clustering parameters from the training set
and generalize to the validation set
this is essential for the clustering based shortcut model to work

'''
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from utils import ID2CHAPTER, split_video_id
from utils import EventLabel, TrainValidSplit, DataLoader, HumanBondaries
sns.set(style='white', palette='colorblind', context='talk')

dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()

# choose dataset
event_id_list = tvs.all_ids
t_f1 = np.zeros(len(event_id_list))

subev_ids = []
scene_vecs = []
n_train_scene_vecs = None

# loop over events
# event_id = '1.3.4'
for i, event_id in enumerate(event_id_list):
    print(f'Event {i} / {len(event_id_list)} - {event_id}')
    # load data
    X, t_f1 = dl.get_data(event_id, get_t_frame1=True, to_torch=False)
    t_f1 = np.round(t_f1)
    # get ground truth boundaries
    df_i = evlab.get_subdf(event_id)
    for evname, evnum, t_start, t_end in zip(df_i['evname'], df_i['evnum'], df_i['startsec'], df_i['endsec']):
        # adjust time index
        if t_start > t_f1:
            t_start -= t_f1
            t_end -= t_f1
        if t_end > len(X):
            t_end = len(X)
        if t_start > len(X):
            continue
        t_start, t_end = int(t_start * 3), int(t_end * 3)
        # compute scene vector
        sv = np.nanmean(X[t_start: t_end, :], axis=0)
        subev_id = list(evlab.all_evnames).index(evname)
        # remove nan vector 
        if np.sum(np.isnan(sv)) > 0:
            continue
        scene_vecs.append(sv)
        subev_ids.append(subev_id)

        if i == tvs.n_train_files:
            n_train_scene_vecs = len(scene_vecs)

scene_vecs = np.array(scene_vecs)
subev_ids = np.array(subev_ids)
print(f'data shape : {np.shape(scene_vecs)}')
print(f'detected nan data {np.sum(np.isnan(scene_vecs))}')

'''split data '''
X_tr = scene_vecs[:n_train_scene_vecs,:]
X_te = scene_vecs[n_train_scene_vecs:,:]
Y_tr = subev_ids[:n_train_scene_vecs]
Y_te = subev_ids[n_train_scene_vecs:]

unique, counts = np.unique(Y_te, return_counts=True)

majority_guess_baseline = np.max(counts) / len(Y_te)
uniform_guess_baseline = 1 / evlab.n_evnames
print(majority_guess_baseline)
print(uniform_guess_baseline)

'''classification - chapter'''
svm = LinearSVC()
svm.fit(X_tr, Y_tr)
Y_pred = svm.predict(X_te)
test_acc = np.mean(Y_pred == Y_te)
print('decode sub event - test accuracy %.3f' % test_acc)
