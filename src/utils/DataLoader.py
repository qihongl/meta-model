import os
import pickle
import numpy as np
import pandas as pd
from utils import to_pth, list_fnames, pickle_load
from utils.PCATransformer import PCATransformer
from sklearn.preprocessing import StandardScaler

DATA_ROOT = '../data/input_scene_vecs/'
scene_vec_fpattern = '*_kinect_sep_09.pkl'
FPS = 25
pcat = PCATransformer()

class DataLoader:

    def __init__(self, feature_tag='sep_09', verbose=False):
        self.fpaths, self.fnames = list_fnames(DATA_ROOT, scene_vec_fpattern)
        self.n_files = len(self.fpaths)
        if verbose:
            print(f'Number of files found: {self.n_files}')

    def event_id_str_to_fpath(self, event_id_str):
        fname = event_id_str + scene_vec_fpattern[1:]
        assert fname in self.fnames, f'File name not found: {fname}'
        fpath = os.path.join(DATA_ROOT, fname)
        return fpath

    def get_x_pca(self, df):
        return pcat.transform(df['combined_resampled_df'].to_numpy())


    def get_frame1_time(self, df, to_sec=True, round=False):
        first_frame_id = df['combined_resampled_df'].index[0]
        if to_sec:
            first_frame_id /= FPS
        if round:
            first_frame_id = np.round(first_frame_id)
        return first_frame_id

    def get_data(self, event_id_str, get_t_frame1=False, to_torch=True):
        fpath = self.event_id_str_to_fpath(event_id_str)
        df = pickle_load(fpath)
        x_pca = self.get_x_pca(df)
        t_f1 = self.get_frame1_time(df)
        if to_torch:
            x_pca = to_pth(x_pca)
        if get_t_frame1:
            return x_pca, t_f1
        return x_pca

    def get_1st_frame_ids(self, event_id_str_list, to_sec=True, round=False):
        f1_ids = np.zeros(len(event_id_str_list))
        for i, event_id_str in enumerate(event_id_str_list):
            fpath = self.event_id_str_to_fpath(event_id_str)
            df = pickle_load(fpath)
            f1_ids[i] = self.get_frame1_time(df, to_sec=to_sec, round=round)
        return f1_ids


    # def get_all_data(self, to_torch=True, get_f1_id=False, verbose=False):
    #     X_pca = [None] * self.n_files
    #     frame1_id = [None] * self.n_files
    #     # load all files and transform to 30D PCs
    #     for i, (fpath, fname) in enumerate(zip(self.fpaths, self.fnames)):
    #         # load data
    #         df = pickle_load(fpath)
    #         X_pca[i] = self.get_x_pca(df, to_torch=True)
    #         frame1_id[i] = self.get_frame1_time(df)
    #         if verbose:
    #             print(f'{fname} - {np.shape(X_pca[i])}')
    #     if get_f1_id:
    #         return X_pca, frame1_id
    #     return X_pca



if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style='white', palette='colorblind', context='talk')
    dl = DataLoader()
    fpath = dl.event_id_str_to_fpath('1.1.1')
    df = pickle_load(fpath)
    df['combined_resampled_df']
    x_pca = dl.get_x_pca(df)
    f1_time = dl.get_frame1_time(df)
    print(f1_time)
    type(df)
