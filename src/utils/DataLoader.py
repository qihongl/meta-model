import os
import pickle
import numpy as np
import pandas as pd
from utils import to_pth
from utils.io import list_fnames
from utils.PCATransformer import PCATransformer
from sklearn.preprocessing import StandardScaler

DATA_ROOT = '../data/input_scene_vecs/'
scene_vec_fpattern = '*_kinect_sep_09.pkl'

pcat = PCATransformer()

class DataLoader:

    def __init__(self, zscore=True, feature_tag='sep_09', verbose=False):
        self.fpaths, self.fnames = list_fnames(DATA_ROOT, scene_vec_fpattern)
        self.n_files = len(self.fpaths)
        self.zscore = zscore
        self.ss = StandardScaler()
        if verbose:
            print(f'Number of files found: {self.n_files}')

    def get_data(self, event_id_str, to_torch=True):
        """
        event_id_str: str in the form of 'x.y.z'
        """
        fname = event_id_str + scene_vec_fpattern[1:]
        assert fname in self.fnames, f'File name not found: {fname} '
        # load data
        fpath = os.path.join(DATA_ROOT, fname)
        data = pickle.load(open(fpath, 'rb'))
        # PCA transform
        x_pca = pcat.transform(data['combined_resampled_df'].to_numpy())
        if self.zscore:
            x_pca = self.ss.fit_transform(x_pca)
        if to_pth:
            x_pca = to_pth(x_pca)
        return x_pca

    def get_all_data(self, to_torch=True, verbose=False):
        X_pca = [None] * self.n_files
        # load all files and transform to 30D PCs
        for i, (fpath, fname) in enumerate(zip(self.fpaths, self.fnames)):
            # load data
            data = pickle.load(open(fpath, 'rb'))
            # PCA transform
            X_pca[i] = pcat.transform(data['combined_resampled_df'].to_numpy())
            if self.zscore:
                X_pca[i] = self.ss.fit_transform(X_pca[i])
            if to_pth:
                X_pca[i] = to_pth(X_pca[i])
            if verbose:
                print(f'{fname} - {np.shape(X_pca[i])}')
        return X_pca


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style='white', palette='colorblind', context='talk')
    dl = DataLoader(zscore=True)
    data = dl.get_data('1.1.1')
    print(np.shape(data))
    # data = dl.get_all_data()
    print(data.mean(axis=0))
    print(data.std(axis=0))
    # np.shape(data[:,0])
    # sns.kdeplot(data[:,0])
