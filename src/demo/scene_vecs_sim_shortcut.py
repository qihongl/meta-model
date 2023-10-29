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
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils import ID2CHAPTER, split_video_id
from utils import EventLabel, TrainValidSplit, DataLoader, HumanBondaries
from model import SimpleShortcut
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

np.shape(subev_ids)

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
n_X_tr = len(X_tr)

'''classification - individual scene vector'''
unique, counts = np.unique(Y_te, return_counts=True)

majority_guess_baseline = np.max(counts) / len(Y_te)
uniform_guess_baseline = 1 / evlab.n_evnames
print(majority_guess_baseline)
print(uniform_guess_baseline)

# svm = LinearSVC()
# svm.fit(X_tr, Y_tr)
# Y_pred = svm.predict(X_te)
# test_acc = np.mean(Y_pred == Y_te)
# print('decode sub event - test accuracy %.3f' % test_acc)

''' classification output
0.0988981909160893
0.021739130434782608
decode sub event - test accuracy 0.516
'''

'''clustering - individual scene vector'''

k = 45
kmeans = KMeans(n_clusters=k, random_state=0).fit(X_tr)
clustering_result = kmeans.predict(X_te)
mi = adjusted_mutual_info_score(clustering_result, Y_te)
print(f'mi = {mi}')

# n_perms = 1000
# mi_perm = [adjusted_mutual_info_score(clustering_result, np.random.choice(range(k), len(Y_te))) for _ in range(n_perms)]
#
# f, ax = plt.subplots(1,1, figsize=(7,4))
# # sns.violinplot(mi_perm, label='null distribution',ax=ax)
# ax.hist(mi_perm, label='null distribution')
# # ax.axvline(mi, ls='--', color='black', label='kNN clustering')
# ax.set_xlabel('mutual information')
# # ax.set_xlim([0, 2])
# ax.legend()
# sns.despine()


def norm_1vec_vs_nvecs(vector, vectors, ord=2):
    '''compute the distance between 1 vector vs. a bunch of vectors'''
    assert len(vector) == np.shape(vectors)[1]
    x_t_rep = np.tile(vector, (np.shape(vectors)[0], 1))
    return np.linalg.norm(x_t_rep - vectors, axis=1, ord=ord)

f, axes = plt.subplots(2,1,figsize=(9,7))
ds = [3, 3.5, 4, 4.5, 5, 5.5, 6]
cpal = sns.color_palette(n_colors=len(ds), palette='viridis')
for di, d in enumerate(ds):
    # d = 4
    ssc = SimpleShortcut(input_dim=30, d=d, use_model=True)

    Y_hats = []

    batch_size = 10000
    for i in tqdm(np.arange(0, n_X_tr, batch_size)):
        # print(np.shape(X_tr[i:i+batch_size,:]))
        ssc.add_data_list(list(X_tr[i:i+batch_size,:]), list(Y_tr[i:i+batch_size]), )
        ssc.update_model()
        Y_hats.append([ssc.predict(x) for x in X_te])


    mis = np.full(len(Y_hats), fill_value=np.nan)
    percent_none = np.full(len(Y_hats), fill_value=np.nan)
    for i, Y_hats_i in enumerate(Y_hats):
        Y_hats_i = np.array(Y_hats_i)
        none_mask = Y_hats_i == None
        percent_none[i] = np.mean(none_mask)
        if percent_none[i] < 1:
            mis[i] = adjusted_mutual_info_score(Y_hats_i[~none_mask], Y_te[~none_mask])

    axes[0].plot(mis, color=cpal[di], label=f'd = {d}')
    axes[1].plot(percent_none, color=cpal[di])
axes[0].axhline(mi, ls='--', color='grey', label='KNN baseline')
axes[0].set_ylabel('mutual infomation')
axes[0].set_title(f'shortcut MI as the data grows')
axes[1].set_ylabel('percent null')
axes[1].set_xlabel('amount of data')
axes[0].legend()
sns.despine()
