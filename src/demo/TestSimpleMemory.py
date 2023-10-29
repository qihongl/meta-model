'''test our simple memory mechanism on meta scene vectors '''
import os
import pickle
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from scipy.spatial import distance_matrix
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import ID2CHAPTER, split_video_id
from utils import EventLabel, TrainValidSplit, DataLoader, HumanBondaries
from model import SimpleMemory
sns.set(style='white', palette='colorblind', context='talk')

dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()

ss = StandardScaler()


# choose dataset
event_id_list = tvs.all_ids
t_f1 = np.zeros(len(event_id_list))

subev_ids = []
scene_vecs = []
n_train_scene_vecs = None

max_len = 5
# loop over events
# event_id = '1.3.4'
for i, event_id in enumerate(event_id_list):
    # print(f'Event {i} / {len(event_id_list)} - {event_id}')
    # load data
    X, t_f1 = dl.get_data(event_id, get_t_frame1=True, to_torch=False)
    X = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
    t_f1 = np.round(t_f1)
    # get ground truth boundaries
    df_i = evlab.get_subdf(event_id)
    for evname, evnum, t_start, t_end in zip(df_i['evname'], df_i['evnum'], df_i['startsec'], df_i['endsec']):
        t_start, t_end = int(t_start * 3), int(t_end * 3)
        # adjust time index
        if t_start > t_f1:
            t_start -= t_f1
            t_end -= t_f1
        if t_end > len(X):
            t_end = len(X)
        if t_start > len(X):
            continue
        t_start, t_end = int(t_start), int(t_end)
        # compute scene vector
        sv = np.nanmean(X[t_start: t_end, :], axis=0)
        subev_id = list(evlab.all_subev_names).index(evname)
        # remove nan vector
        if np.sum(np.isnan(sv)) > 0:
            continue

        this_event_len = t_end - t_start
        l_ = np.min([this_event_len, max_len])

        # scene_vecs.append(X[t_start: t_end, :])
        scene_vecs.append(X[t_start: t_start + l_, :])
        subev_ids.append([subev_id] * l_)

        assert np.shape(X[t_start: t_end, :])[0] == (t_end - t_start)

        if i == tvs.n_train_files:
            n_train_scene_vecs = len(scene_vecs)

scene_vecs_s = np.vstack(scene_vecs)
subev_ids_s = np.concatenate(subev_ids)
n_train_vecs = np.shape(np.vstack(scene_vecs[:n_train_scene_vecs]))[0]

input_dim = np.shape(scene_vecs_s)[1]
n_samples = 3000
n_samples_test = np.shape(scene_vecs_s)[0] - n_samples
data, labels = scene_vecs_s[:n_samples,:], subev_ids_s[:n_samples]
data_test, labels_test = scene_vecs_s[n_samples:,:], subev_ids_s[n_samples:]

print(f'X shape : {np.shape(data)}')
print(f'Y shape : {np.shape(labels)}')
print(f'X test shape : {np.shape(data_test)}')
print(f'Y test shape : {np.shape(labels_test)}')


# pca = PCA(n_components=2)
# data_pc = pca.fit_transform(data)
# data_pc_test = pca.fit_transform(data_test)
#
# f, ax = plt.subplots(1,1, figsize=(7,5))
# ax.scatter(data_pc[:,0], data_pc[:,1], c=labels, cmap = 'viridis')
# ax.legend()

# f, ax = plt.subplots(1,1, figsize=(7,5))
# ax.scatter(data_pc[:,0], data_pc[:,1])
# ax.scatter(data_pc_test[:,0], data_pc_test[:,1])
# ax.legend()

# sns.clustermap(data_pc)
# sns.clustermap(data)


lr = .5
# d = .2
# ds = [.1, .2, .4, 1, 2]
majority_p = .2
# ds = [3, 5, 8]
ds = [5]

for d in ds:

    sm = SimpleMemory(input_dim=input_dim, d=d, majority_p=majority_p)
    Y_hat = np.zeros((n_samples, n_samples_test))
    acc = np.zeros((n_samples, ))
    mis = np.zeros((n_samples, ))
    dnk = np.zeros((n_samples, ))

    for i, (data_i, label_i) in tqdm(enumerate(zip(data, labels))):
        sm.add_data(data_i, label_i)

        # Y_hat[i, :] = [sm.predict(x) for x in data_test]
        # [np.linalg.norm(data_test[0] - x) for x in sm.X]
        # len(sm.X)

        Y_hat[i, :] = [sm.rnc_predict(x) for x in data_test]
        n_recalls = np.sum(Y_hat[i,:]!=-1)
        correct_recalls = np.logical_and(Y_hat[i,:]!=-1, Y_hat[i,:] == labels_test)
        incorrect_recalls = np.logical_and(Y_hat[i,:]!=-1, Y_hat[i,:] != labels_test)

        # acc[i] = np.sum(Y_hat[i,:] == labels) / n_samples
        acc[i] = np.sum(correct_recalls) / len(labels_test)
        mis[i] = np.sum(incorrect_recalls) / len(labels_test)
        dnk[i] = (len(labels_test) - n_recalls) / len(labels_test)


    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(acc,color = 'k')
    ax.plot(acc+mis,color = 'k')
    ax.plot(np.ones_like(acc),color = 'k')
    ax.fill_between(np.arange(n_samples), np.zeros_like(acc), acc, alpha=0.2, label='acc', color='green')
    ax.fill_between(np.arange(n_samples), acc, acc+mis, alpha=0.2, label='err', color='red')
    ax.fill_between(np.arange(n_samples), acc, acc+mis+dnk, alpha=0.2, label='dnk', color='blue')
    ax.set_title(f'd = {d}, acc = %.2f, mis = %.2f' % (np.mean(acc), np.mean(mis)))
    sns.despine()
    ax.legend()

'''
data    time
500     47
1000    5:56
'''
