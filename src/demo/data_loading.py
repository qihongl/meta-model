import os
import pickle
import numpy as np
import pandas as pd
from utils.io import list_fnames, split_video_id
# from utils.PCATransformer import PCATransformer
# from utils.EventLabel import EventLabel
# from utils.TrainValidSplit import TrainValidSplit
from utils import PCATransformer, EventLabel, TrainValidSplit, DataLoader

DATA_ROOT = '../data/input_scene_vecs/'
scene_vec_fpattern = '*_kinect_sep_09.pkl'
event_label_fpath = '../data/high_level_events/event_annotation_timing_average.csv'

pcat = PCATransformer()
tvs = TrainValidSplit()
evlab = EventLabel()


fpaths, fnames = list_fnames(DATA_ROOT, scene_vec_fpattern)
print(f'Number of files found: {len(fpaths)}')

# load all files and transform to 30D PCs
for i, (fpath, fname) in enumerate(zip(fpaths, fnames)):
    # load data
    data = pickle.load(open(fpath, 'rb'))
    # PCA transform
    x_pca = pcat.transform(data['combined_resampled_df'].to_numpy())
    # get event id
    event_id = fname.split('_')[0]

    # get all subevent number
    event_nums = evlab.get_all_evnums(event_id)
    print(f'{event_id}  - {np.shape(x_pca)}')
    # get event time
    for i, evnum in enumerate(event_nums):
        print(f'\t{evnum}, {evlab.get_evtimes(event_id, evnum)}')
