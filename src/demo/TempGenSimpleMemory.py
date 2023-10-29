'''test our simple memory mechanism on meta scene vectors
using a encoding-recall temporal generalization test
assume:
- the model encode at time t, where t can be some early time points or at the end of each event
- the model recall LC (context classification) at time t
- for both retrieval and encoding, the model use the running mean of the input representation
    - which is known to be better than single-frame representation
'''
import os
import pickle
import itertools
import numpy as np
import numpy.ma as ma
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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from model import Vanilla_iSITH
from utils import ID2CHAPTER, split_video_id
from utils import EventLabel, TrainValidSplit, DataLoader, HumanBondaries
from model import SimpleMemory
sns.set(style='white', palette='colorblind', context='talk')

dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()

ss = StandardScaler()

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

# choose dataset
event_id_list = tvs.all_ids
t_f1 = np.zeros(len(event_id_list))

scene_id_enc = []
scene_id_rcl = []
scene_enc = [] # scene vec stored
scene_rcl = [] # scene vec used for recall (i.e. cues)
# n_train_scene_enc = None

enc_t = 32
rcl_t = 32
# loop over events
# event_id = '1.3.4'
for i, event_id in enumerate(event_id_list):
    print(f'Event {i} / {len(event_id_list)} - {event_id}')
    # load metadata
    X, t_f1 = dl.get_data(event_id, get_t_frame1=True, to_torch=False)
    # X = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
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
        this_evlen = t_end - t_start
        sv = np.nanmean(X[t_start: t_end, :], axis=0)
        subev_id = list(evlab.all_subev_names).index(evname)
        # remove nan vector
        if np.sum(np.isnan(sv)) > 0 or rcl_t > this_evlen:
            continue

        enc_t_bounded = np.min([this_evlen, enc_t])
        scene_enc.append(np.mean(X[t_start: t_start + enc_t_bounded, :], axis=0))
        scene_rcl.append(np.mean(X[t_start: t_start + rcl_t, :], axis=0))
        scene_id_enc.append([subev_id] * 1)
        scene_id_rcl.append([subev_id] * 1)


        if i == tvs.n_train_files:
            n_samples = len(scene_enc)

# reformat data
scene_enc_s = np.vstack(scene_enc)
scene_rcl_s = np.vstack(scene_rcl)
scene_id_enc_s = np.concatenate(scene_id_enc)
scene_id_rcl_s = np.concatenate(scene_id_rcl)
print(f'scene_enc shape : {np.shape(scene_enc_s)}')
print(f'scene_rcl shape : {np.shape(scene_rcl_s)}')


# get data params
input_dim = np.shape(scene_enc_s)[1]
# get test set size
n_samples_test = np.shape(scene_enc_s)[0] - n_samples

# split the data
data_tr, label_tr = scene_enc_s[:n_samples,:], scene_id_enc_s[:n_samples]
data_te, label_te = scene_rcl_s[n_samples:,:], scene_id_rcl_s[n_samples:]
np.shape(label_tr)
np.shape(label_te)

# print data shape
print(f'X shape : {np.shape(data_tr)}')
print(f'Y shape : {np.shape(label_tr)}')
print(f'X test shape : {np.shape(data_te)}')
print(f'Y test shape : {np.shape(label_te)}')


'''PCA the scene representation'''

# embedding = PCA(n_components=2)
# # embedding = TSNE(n_components=2)
#
# data_tr_pc = embedding.fit_transform(data_tr)
# data_te_pc = embedding.transform(data_te)
#
# f, ax = plt.subplots(1,1, figsize=(6,5))
# ax.scatter(data_tr_pc[:,0], data_tr_pc[:,1], c=label_tr, cmap = 'viridis')
# ax.legend()
#
# f, ax = plt.subplots(1,1, figsize=(6,5))
# ax.scatter(data_te_pc[:,0], data_te_pc[:,1], c=label_te, cmap = 'viridis')
# ax.legend()

# f, ax = plt.subplots(1,1, figsize=(7,5))
# ax.scatter(data_tr_pc[:,0], data_tr_pc[:,1])
# ax.scatter(data_te_pc[:,0], data_te_pc[:,1])
# ax.legend()

# sns.clustermap(data_tr)
# sns.clustermap(data_te)

'''compute pairwise distance distribution w.r.t same/diff event label'''

def compare_arrays(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    m = x.shape[0]
    n = y.shape[0]
    x_2d = x[:, np.newaxis]
    y_2d = y[np.newaxis, :]
    Z = np.equal(x_2d, y_2d).astype(int)
    return Z


def compute_d_prime(samples1, samples2):
    """
    Calculate d-prime (Cohen's d) for two arrays of samples from different distributions.

    Parameters:
        samples1 (numpy.ndarray): Array containing samples from distribution 1.
        samples2 (numpy.ndarray): Array containing samples from distribution 2.

    Returns:
        float: The calculated d-prime value.
    """
    mean1 = np.mean(samples1)
    mean2 = np.mean(samples2)
    std_dev1 = np.std(samples1, ddof=1)  # Use ddof=1 for sample standard deviation
    std_dev2 = np.std(samples2, ddof=1)

    d_prime = (mean1 - mean2) / np.sqrt(std_dev1 ** 2 + std_dev2 ** 2)

    return d_prime

from sklearn.metrics.pairwise import cosine_distances
# from scipy.spatial.distance import correlation
dist_mat = distance_matrix(data_tr, data_te)
# np.shape(dist_mat)
dist_mat = cosine_distances(data_tr, data_te)

mask_same_event_id = compare_arrays(label_tr, label_te)
dist_mat_same_event_id = dist_mat[np.where(mask_same_event_id)[0], np.where(mask_same_event_id)[1]]
dist_mat_diff_event_id = dist_mat[np.where(1 - mask_same_event_id)[0], np.where(1 - mask_same_event_id)[1]]

dprime = compute_d_prime(dist_mat_diff_event_id, dist_mat_same_event_id)

f, ax = plt.subplots(1,1, figsize=(6,3))
sns.kdeplot(dist_mat_diff_event_id.ravel(), label = 'different event ids', ax=ax)
sns.kdeplot(dist_mat_same_event_id.ravel(), label = 'same event id', ax=ax)
ax.set_xlabel('cosine distances')
ax.set_title('cohen\'s d = %.2f' % dprime)
# v = 5.075
# ax.axvline(v, ls='--', color='grey', label = 'd = %.2f' % v)
ax.legend()
sns.despine()



# d = .2
# ds = [.1, .2, .4, 1, 2]
# majority_p = .8
# ds = [3, 5, 8]
# ds = [2, 3, 4, 5]
ds = [.2, .25, .3, .35, .4, .45, .5]

add_data_freq = 10
train_freq_id = np.arange(0, n_samples, add_data_freq)

for d in ds:

    sm = SimpleMemory(input_dim=input_dim, d=d)
    Y_hat = np.zeros((len(train_freq_id), n_samples_test))
    acc = np.zeros((len(train_freq_id), ))
    mis = np.zeros((len(train_freq_id), ))
    dnk = np.zeros((len(train_freq_id), ))

    # for i, (data_i, label_i) in tqdm(enumerate(zip(data_tr, label_tr))):
    for i, ii in enumerate(train_freq_id):
        # print(ii, ii+add_data_freq)

        data_batch_i = zip(data_tr[ii:ii+add_data_freq], label_tr[ii:ii+add_data_freq])
        for j, (data_j, label_j) in enumerate(data_batch_i):
            sm.add_data(data_j, label_j)

        Y_hat[i, :] = [sm.predict(x) for x in data_te]
        n_recalls = np.sum(Y_hat[i,:]!=-1)
        correct_recalls = np.logical_and(Y_hat[i,:]!=-1, Y_hat[i,:] == label_te)
        incorrect_recalls = np.logical_and(Y_hat[i,:]!=-1, Y_hat[i,:] != label_te)

        acc[i] = np.sum(correct_recalls) / len(label_te)
        mis[i] = np.sum(incorrect_recalls) / len(label_te)
        dnk[i] = (len(label_te) - n_recalls) / len(label_te)


    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(acc,color = 'k')
    ax.plot(acc+mis,color = 'k')
    ax.plot(np.ones_like(acc),color = 'k')
    ax.fill_between(range(len(train_freq_id)), np.zeros_like(acc), acc, alpha=0.2, label='acc', color='green')
    ax.fill_between(range(len(train_freq_id)), acc, acc+mis, alpha=0.2, label='err', color='red')
    ax.fill_between(range(len(train_freq_id)), acc, acc+mis+dnk, alpha=0.2, label='dnk', color='blue')
    ax.set_title(f'd = {d}\nfinal acc = %.2f, final mis = %.2f' % (acc[-1], mis[-1]))
    ax.set_xticks(np.arange(0, len(train_freq_id), 10))
    ax.set_xticklabels(np.arange(0, len(train_freq_id), 10) * 10)
    ax.set_xlabel('# events used')
    sns.despine()
    ax.legend()

'''
#data_points    time
500     47
1000    5:56
'''
