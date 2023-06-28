'''
check if we can cluster the scene vectors
in particular, can we derive good clustering parameters from the training set
and generalize to the validation set
this is essential for the clustering based shortcut model to work

'''
import os
import pickle
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.metrics import mutual_info_score
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from model import Vanilla_iSITH
from utils import ID2CHAPTER, split_video_id, to_np
from utils import EventLabel, TrainValidSplit, DataLoader, HumanBondaries
sns.set(style='white', palette='colorblind', context='talk')

def erwa(data, decay_factor):
    data = data[:,::-1]
    num_features, num_time_steps = data.shape
    weights = np.power(decay_factor, np.arange(num_time_steps))
    weighted_data = data * weights.reshape(1, -1)
    erwa_result = np.sum(weighted_data, axis=1) / np.sum(weights)
    return erwa_result

# SITH
input_dim = 30
ntau = 4
tau_min = 1
tau_max = 200
buff_max = 600
k = 5
dt = 1
g = .5
sith_output_dim = input_dim * ntau
isith = Vanilla_iSITH(
    ntau = ntau, tau_min = tau_min, tau_max = tau_max, buff_max = buff_max,
    k = k, dt = dt, g = g
)

# EWMA
decay_factor = .8

# data utils
dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()

# choose dataset
event_id_list = tvs.all_ids
t_f1 = np.zeros(len(event_id_list))

subev_ids = []
scene_vecs = []
scene_vecs_sith = []
scene_vec_len = []
n_train_scene_vecs = None

# loop over events
# event_id = '1.3.4'
for i, event_id in enumerate(event_id_list):
    # print(f'Event {i} / {len(event_id_list)} - {event_id}')
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
        # compute scene vector as the mean
        # sv = np.mean(X[t_start: t_end, :], axis=0)
        # EWMA
        sv = erwa(X[t_start: t_end, :].T, decay_factor = decay_factor)

        # remove nan vector
        if np.sum(np.isnan(sv)) > 0:
            continue

        # SITH
        sith_in = torch.tensor(X[t_start: t_end, :].T, dtype=torch.float).view((1, 1, input_dim, -1))
        sv_sith = isith.simple_forward(sith_in)

        # aggregate info
        scene_vec_len.append(t_end - t_start)
        subev_id = list(evlab.all_subev_names).index(evname)
        subev_ids.append(subev_id)
        scene_vecs.append(sv)
        scene_vecs_sith.append(to_np(sv_sith))

        if i == tvs.n_train_files:
            n_train_scene_vecs = len(scene_vecs)

scene_vecs = np.array(scene_vecs)
subev_ids = np.array(subev_ids)
scene_vecs_sith = np.array(scene_vecs_sith)
print(f'data shape : {np.shape(scene_vecs)}')
print(f'detected nan data {np.sum(np.isnan(scene_vecs))}')

print(f'sith data shape : {np.shape(scene_vecs_sith)}')

# f, ax = plt.subplots(1,1, figsize = (7, 4))
# sns.histplot(scene_vec_len,ax=ax)
# mean_scene_vec_len = np.mean(scene_vec_len)
# ax.axvline(mean_scene_vec_len, ls='--', color='k',label='mean = %.2f' % (mean_scene_vec_len))
# ax.set_title('distribution of event length')
# ax.legend()
# sns.despine()


'''split data '''
X_tr = scene_vecs[:n_train_scene_vecs,:]
X_te = scene_vecs[n_train_scene_vecs:,:]
Y_tr = subev_ids[:n_train_scene_vecs]
Y_te = subev_ids[n_train_scene_vecs:]

X_sith_tr = scene_vecs_sith[:n_train_scene_vecs,:]
X_sith_te = scene_vecs_sith[n_train_scene_vecs:,:]


'''chance baseline'''
unique, counts = np.unique(Y_te, return_counts=True)

majority_guess_baseline = np.max(counts) / len(Y_te)
uniform_guess_baseline = 1 / evlab.n_subev_names
print(majority_guess_baseline)
print(uniform_guess_baseline)

'''classification'''
svm = LinearSVC()
svm.fit(X_tr, Y_tr)
Y_pred = svm.predict(X_te)
test_acc = np.mean(Y_pred == Y_te)
print('decode sub event - test accuracy %.3f' % test_acc)

'''classification'''
svm = LinearSVC()
svm.fit(X_sith_tr, Y_tr)
Y_pred = svm.predict(X_sith_te)
test_acc = np.mean(Y_pred == Y_te)
print('decode sub event - SITH accuracy %.3f' % test_acc)


# '''clustering'''
# from sklearn.cluster import KMeans
#
# k = 35
# kmeans = KMeans(n_clusters=k, random_state=0).fit(X_tr)
# clustering_result = kmeans.predict(X_te)
# mi = mutual_info_score(clustering_result, Y_te)
#
# n_perms = 1000
# mi_perm = [mutual_info_score(Y_te, np.random.choice(range(k), len(Y_te))) for _ in range(n_perms)]
#
# f, ax = plt.subplots(1,1, figsize=(7,4))
# ax.hist(mi_perm, label='null distribution')
# ax.axvline(mi, ls='--', color='black', label='kNN clustering')
# ax.set_xlabel('mutual information')
# ax.legend()
# sns.despine()
