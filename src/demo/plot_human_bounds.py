import os
import pickle
import numpy as np
import pandas as pd
from utils.EventLabel import EventLabel

evlab = EventLabel()
human_bond_fpath = '../data/high_level_events/seg_data_analysis_clean.csv'
human_bond_df = pd.read_csv(human_bond_fpath)
human_bond_df.head()
len(human_bond_df)


len(list(human_bond_df['Movie']))
np.sum(['C1_trim' in mname for mname in list(human_bond_df['Movie'])])
