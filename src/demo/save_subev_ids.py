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
from utils import ID2CHAPTER, split_video_id, pickle_save
from utils import EventLabel, TrainValidSplit, DataLoader, HumanBondaries
from model import SimpleShortcut
sns.set(style='white', palette='colorblind', context='talk')

dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()

# choose dataset
event_id_list = tvs.all_ids
t_f1 = np.zeros(len(event_id_list))

subev_ids_dict = {}

# loop over events
# event_id = '1.3.4'
for i, event_id in enumerate(event_id_list):
    print(f'Event {i} / {len(event_id_list)} - {event_id}')
    # load data
    X, t_f1 = dl.get_data(event_id, get_t_frame1=True, to_torch=False)
    t_f1 = np.round(t_f1)

    # get event label info for event i
    df_i = evlab.get_subdf(event_id)
    event_i_len = int(df_i['endsec'].iloc[-1])
    sub_ev_label_i = np.full(event_i_len, np.nan)
    for evname, evnum, t_start, t_end in zip(df_i['evname'], df_i['evnum'], df_i['startsec'], df_i['endsec']):
        t_start, t_end = int(t_start), int(t_end)
        subev_id = list(evlab.all_evnames).index(evname)
        sub_ev_label_i[t_start:t_end] = subev_id

    subev_ids_dict[event_id] = sub_ev_label_i

pickle_save(subev_ids_dict, 'utils/subev_ids_dict.pkl')
