import os
import pickle
import pandas as pd
from utils.io import pickle_load, list_fnames, pickle_save
# from sklearn.decomposition import PCA

DATA_ROOT = '../data'

print(pickle.format_version)

list_fnames('../data/input_scene_vecs', '*.pkl')

fname = 'input_scene_vecs/1.1.1_kinect_sep_09.pkl'
fpath = os.path.join(DATA_ROOT, fname)
data = pickle.load(open(fpath, 'rb'))
print(type(data))
for k, v in data.items():
    print(k, type(v), len(v.columns))
    print(v.head())
    print('-'*150)

'''
combined_resampled_df
- everything stacked together, same sampling rate
-
'''

fname = 'input_scene_vecs/sep_09_all_appear_pca.pkl'
fpath = os.path.join(DATA_ROOT, fname)
data = pickle.load(open(fpath, 'rb'))

print(type(data))
for k, v in data.items():
    print(k, type(v))
    print(v.head())


fname = 'pca/dec_6_rotated_skel_all_30_appear_pca.pkl'
fpath = os.path.join(DATA_ROOT, fname)
# data = pickle_load(fpath, to_nparray=False)
pca = pickle.load(open(fpath, 'rb'))
print(type(pca))
print(pca.components_)
print(pca.explained_variance_)

fname = 'high_level_events/event_annotation_timing.csv'
fpath = os.path.join(DATA_ROOT, fname)
df = pd.read_csv(fpath)
df.head()


fname = 'object_labels/1.1.1_C1_labels.csv'
fpath = os.path.join(DATA_ROOT, fname)
df = pd.read_csv(fpath)
df.head()

fname = 'object_labels/1.1.1_kinect_labels.csv'
fpath = os.path.join(DATA_ROOT, fname)
df = pd.read_csv(fpath)
df.head()

'''
event annot timing

input scene vecs

pca
- what are all the files
- appear - whether a new obj has appeared?
- emb - ?
- optical - frame by frame difference?
- the last 2

object labels / object tracking
- what's c1 vs c2, where are the png files

are all videos in Full_SEM?
'''



import pickle
import numpy as np
a = dict({'hello': 'world', 'a': [np.zeros(3,), np.ones(4)]})

pickle_save(a, 'filename.pickle')
b = pickle_load('filename.pickle')
print(a == b)
