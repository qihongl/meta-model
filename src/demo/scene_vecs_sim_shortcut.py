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
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
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
        subev_id = list(evlab.all_evnames).index(evname)
        # remove nan vector
        if np.sum(np.isnan(sv)) > 0:
            continue
        scene_vecs.append(X[t_start: t_end, :])
        subev_ids.append([subev_id] * (t_end - t_start))
        assert np.shape(X[t_start: t_end, :])[0] == (t_end - t_start)

        if i == tvs.n_train_files:
            n_train_scene_vecs = len(scene_vecs)

scene_vecs_s = np.vstack(scene_vecs)
subev_ids_s = np.concatenate(subev_ids)
n_train_vecs = np.shape(np.vstack(scene_vecs[:n_train_scene_vecs]))[0]


print(f'X shape : {np.shape(scene_vecs_s)}')
print(f'Y shape : {np.shape(subev_ids_s)}')

'''split data '''
X_tr = scene_vecs_s[:n_train_vecs,:]
X_te = scene_vecs_s[n_train_vecs:,:]
Y_tr = subev_ids_s[:n_train_vecs]
Y_te = subev_ids_s[n_train_vecs:]

# # for i, x in enumerate(tqdm(X_tr)):
#
# '''clustering - try what '''
# k = 45
# kmeans = KMeans(n_clusters=k, random_state=0).fit(X_tr)
# clustering_result = kmeans.predict(X_te)
# mi = mutual_info_score(clustering_result, Y_te)
#
# n_perms = 1000
# mi_perm = [mutual_info_score(clustering_result, np.random.choice(range(k), len(Y_te))) for _ in range(n_perms)]
#
# f, ax = plt.subplots(1,1, figsize=(7,4))
# # sns.violinplot(mi_perm, label='null distribution',ax=ax)
# ax.hist(mi_perm, label='null distribution')
# # ax.axvline(mi, ls='--', color='black', label='kNN clustering')
# ax.set_xlabel('mutual information')
# # ax.set_xlim([0, 2])
# ax.legend()
# sns.despine()
