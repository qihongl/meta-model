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

chapter_ids = np.zeros(len(event_id_list), dtype=int)
actor_ids = np.zeros(len(event_id_list), dtype=int)
run_ids = np.zeros(len(event_id_list), dtype=int)
scene_vecs = np.zeros((len(event_id_list), 30))

# loop over events
for i, event_id in enumerate(event_id_list):
    print(f'Event {i} / {len(event_id_list)} - {event_id}')

    # load data
    X, t_f1 = dl.get_data(event_id, get_t_frame1=True, to_torch=False)
    actor_ids[i], chapter_ids[i], run_ids[i] = split_video_id(event_id)
    # get ground truth boundaries
    event_bound_times, event_bound_vec = evlab.get_bounds(event_id_list[i])

    # add the average video vector to the df
    scene_vecs[i] = np.mean(X, axis=0)

'''split data '''
X_tr = scene_vecs[:tvs.n_train_files,:]
X_te = scene_vecs[tvs.n_train_files:,:]
Y_tr = chapter_ids[:tvs.n_train_files]
Y_te = chapter_ids[tvs.n_train_files:]

'''PCA'''
pca = PCA(n_components=2)
X_tr_pca = pca.fit_transform(X_tr)
X_te_pca = pca.transform(X_te)

X_pca = np.vstack([X_tr_pca, X_te_pca])
svdf = pd.DataFrame({'PC 1': X_pca[:,0], 'PC 2':X_pca[:,1]})

'''classification - chapter'''
svm = LinearSVC()
svm.fit(X_tr, Y_tr)
Y_pred = svm.predict(X_te)
test_acc = np.mean(Y_pred == Y_te)
print('decode chapter - test accuracy %.3f' % test_acc)

'''scatter - chapter'''
f, ax = plt.subplots(1,1, figsize=(7, 6))
alpha = .8
chapter_ids_str = [ID2CHAPTER[str(i)] for i in chapter_ids]
sns.scatterplot(data=svdf, x='PC 1', y='PC 2', hue=chapter_ids_str, alpha=alpha, ax=ax)
ax.set_title('Projected scene vectors')
ax.legend(title='chapter id')
sns.despine()
f.tight_layout()


'''classification - actor'''
Y_tr = actor_ids[:tvs.n_train_files]
Y_te = actor_ids[tvs.n_train_files:]

svm = LinearSVC()
svm.fit(X_tr, Y_tr)
Y_pred = svm.predict(X_te)
test_acc = np.mean(Y_pred == Y_te)
print('decode actor - test accuracy %.3f' % test_acc)


'''scatter - chapter'''
f, ax = plt.subplots(1,1, figsize=(7, 6))
alpha = .8
sns.scatterplot(data=svdf, x='PC 1', y='PC 2', hue=actor_ids, alpha=alpha, ax=ax, palette='colorblind')
ax.set_title('Projected scene vectors')
ax.legend(title='actor id')
sns.despine()
f.tight_layout()


'''classification - run'''
Y_tr = run_ids[:tvs.n_train_files]
Y_te = run_ids[tvs.n_train_files:]

svm = LinearSVC()
svm.fit(X_tr, Y_tr)
Y_pred = svm.predict(X_te)
test_acc = np.mean(Y_pred == Y_te)
print('decode run - test accuracy %.3f' % test_acc)


'''scatter - chapter'''
f, ax = plt.subplots(1,1, figsize=(7, 6))
alpha = .8
sns.scatterplot(data=svdf, x='PC 1', y='PC 2', hue=run_ids, alpha=alpha, ax=ax, palette='colorblind')
ax.set_title('Projected scene vectors')
ax.legend(title='run id')
sns.despine()
f.tight_layout()
