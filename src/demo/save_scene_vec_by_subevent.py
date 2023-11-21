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

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from utils import ID2CHAPTER, split_video_id
from utils import EventLabel, TrainValidSplit, DataLoader, HumanBondaries
from utils import pickle_save
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

data = {k: [] for k in evlab.all_subev_names}

video_data = {event_id:[] for event_id in event_id_list}
video_data_llc = {event_id:[] for event_id in event_id_list}

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
        subev_id = list(evlab.all_subev_names).index(evname)
        # remove nan vector
        if np.sum(np.isnan(sv)) > 0:
            continue
        scene_vecs.append(sv)
        subev_ids.append(subev_id)

        if i == tvs.n_train_files:
            n_train_scene_vecs = len(scene_vecs)

        data[evname].append(X[t_start: t_end, :])
        video_data[event_id].append(X[t_start: t_end, :])
        video_data_llc[event_id].append(evname)

scene_vecs = np.array(scene_vecs)
subev_ids = np.array(subev_ids)
print(f'data shape : {np.shape(scene_vecs)}')
print(f'detected nan data {np.sum(np.isnan(scene_vecs))}')


np.shape(data)
for k, v in data.items():
    print('%s, # events %3d' % (k, len(v)))

# video_data_dict = {
#     'video_data':video_data,
#     'video_data_llc':video_data_llc,
# }
# pickle_save(video_data_dict, 'meta-videos.pkl')

# n_events_per_event_type = [len(v) for v in data.values()]

# f, ax = plt.subplots()
# sns.histplot(n_events_per_event_type, bins=20, ax=ax)
# ax.set_xlim([0, None])
# sns.despine()


# '''hier cluster - event type level - sns'''
# c_cpal = sns.color_palette('colorblind', n_colors=5)
# c_cpal = np.roll(c_cpal,shift=1,axis=0)
# # compute the mean representation
# mean_sv = np.zeros((evlab.n_subev_names, 30))
# for i, evid  in enumerate(subev_ids):
#     mean_sv[evid,:] = np.mean(scene_vecs[subev_ids == evid, :],axis=0)
# # get the color for high level labels
# row_colors = np.array([c_cpal[i] for i in evlab.subevn2cat.values()])
#
#
# # remove multi-chapter action (mca)
# mask_rm_mca = np.array([v for v in evlab.subevn2cat.values()]) !=0
# mean_sv = mean_sv[mask_rm_mca]
# row_colors = row_colors[mask_rm_mca]
# yticklabels = evlab.all_subev_names[mask_rm_mca]
# metric = 'correlation'
# cg = sns.clustermap(
#     mean_sv,
#     col_cluster=0, row_cluster=1,
#     metric=metric,
#     row_colors=row_colors,
#     yticklabels=yticklabels,
#     dendrogram_ratio=[.5, 0],
#     figsize=(12, 12),
#     # cmap='viridis',
#     cbar_pos=(0.05, 0.05, .03, .15),
#     )
# cg.ax_heatmap.set_xlabel("Feature dimension")
# cg.ax_heatmap.set_xticks([0, 10, 20])
# cg.ax_heatmap.set_xticklabels([0, 10, 20])
# cg.ax_heatmap.set_title(f'metric={metric}')
# # color the text label
# for ytick in cg.ax_heatmap.get_yticklabels():
#     ytick.set_color(c_cpal[evlab.subevn2cat[ytick.get_text()]])
#
# cg.dendrogram_row.reordered_ind
# cg.dendrogram_row.linkage
