import os
import numpy as np
import pandas as pd
from utils import pickle_load
import matplotlib.pyplot as plt
datapath = '/Users/qlu/Dropbox/github/extended-event-modeling/output/run_sem/dec_18_pe5E-01_s1010_1E-02_1E-01'
fname = '1.1.10_kinect_trimdec_18_pe5E-01_s1010_1E-02_1E-01_diagnostic_13.pkl'
data_dict = pickle_load(os.path.join(datapath, fname))

# print(list(data_dict.keys()))
for k, v in data_dict.items():
    print(k, type(v))

    '''
    pe
    e_hat
    boundaries
    1st frame
    end sec
    fps
    current epoch
    is_train - 0 or 1 -> if it is a training video

    resampled_indices - frame ids

    c
    c eval

    uncertainty - continuous
    xhat - predicted vector
    relu - hidden unit act
    log like
    log prior
    log post
    triggers - 0 or 1 - whether to do full inf based on some condition

    '''

data_dict
print(np.shape(data_dict['pe']))
print(np.shape(data_dict['pe_w']))
print(np.shape(data_dict['pe_w2']))
print(np.shape(data_dict['pe_w3']))
print()
print(np.shape(data_dict['boundaries']))
print(np.shape(data_dict['e_hat']))
print()
print(np.shape(data_dict['x']))
print()
print(np.shape(data_dict['c']))
print(np.shape(data_dict['c_eval']))
print()
print(data_dict['fps'])
print(data_dict['first_frame'])
print(data_dict['end_second'])
print(data_dict['current_epoch'])
print(data_dict['is_train'])
