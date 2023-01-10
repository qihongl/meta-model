'''verify that my boundaries match tan's computation'''

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import ID2CHAPTER, split_video_id, padded_corr
from utils import EventLabel, TrainValidSplit, DataLoader, HumanBondaries
sns.set(style='white', palette='colorblind', context='talk')

dl = DataLoader()
tvs = TrainValidSplit()
evlab = EventLabel()
hb = HumanBondaries()

fhb = hb.get_precomputed_hb('fine')
chb = hb.get_precomputed_hb('coarse')

# choose dataset
event_id_list = tvs.train_ids
t_f1 = np.zeros(len(event_id_list))

alpha = .5
for i, event_id in enumerate(fhb.keys()):
    if i > 5:
        break

    p_b_c = hb.get_bound_prob(event_id, 'coarse', to_sec=True)
    p_b_f = hb.get_bound_prob(event_id, 'fine', to_sec=True)

    f, axes = plt.subplots(2, 1, figsize=(10,6))
    axes[0].plot(fhb[event_id], label = 'precomputed', alpha=alpha)
    axes[0].plot(p_b_f, label = 'mine', alpha=alpha)
    axes[0].set_title(f'{event_id}, fine')

    axes[1].plot(chb[event_id], label = 'precomputed', alpha=alpha)
    axes[1].plot(p_b_c, label = 'mine', alpha=alpha)
    axes[1].set_title(f'{event_id}, coarse')

    for ax in axes:
        ax.legend()
    f.tight_layout()
    sns.despine()
