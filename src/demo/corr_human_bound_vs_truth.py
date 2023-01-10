import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import padded_corr
from utils import EventLabel, TrainValidSplit, HumanBondaries
sns.set(style='white', palette='colorblind', context='talk')

tvs = TrainValidSplit()
evlab = EventLabel()
hb = HumanBondaries()

# choose dataset
# event_id_list = tvs.train_ids
event_id_list = tvs.valid_ids

r_crse = np.zeros(len(event_id_list),)
r_fine = np.zeros(len(event_id_list),)
p_crse = np.zeros(len(event_id_list),)
p_fine = np.zeros(len(event_id_list),)
# loop over events
for i, event_id in enumerate(event_id_list):

    event_bound_times, event_bound_vec = evlab.get_bounds(event_id)

    p_b_c = hb.get_bound_prob(event_id, 'coarse')
    p_b_f = hb.get_bound_prob(event_id, 'fine')

    r_crse[i], p_crse[i] = padded_corr(event_bound_vec, p_b_c, porp = .1)
    r_fine[i], p_fine[i] = padded_corr(event_bound_vec, p_b_f, porp = .1)



f, ax = plt.subplots(1,1, figsize=(5,4))
sns.violinplot(r_crse, ax=ax)
sns.despine()
ax.set_title(f'mean r = %.3f' % (r_crse.mean()))
ax.set_xlabel('r')
f.tight_layout()


f, ax = plt.subplots(1,1, figsize=(5,4))
sns.violinplot(r_fine, ax=ax)
ax.set_title(f'mean r = %.3f' % (r_fine.mean()))
ax.set_xlabel('r')
sns.despine()
f.tight_layout()
