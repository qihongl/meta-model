import pickle
import numpy as np
from sklearn.decomposition import PCA

DATA_ROOT = '../data/input_scene_vecs/'

class PCATransformer:
    """
    This class loads individual PCA matrices and perform transform or invert based on these PCA matrices
    PCA matrices were derived by PCAComputer class
    """

    def __init__(self, feature_tag='sep_09', pca_tag='all'):
        self.feature_tag = feature_tag
        self.pca_tag = pca_tag
        self.pca_all = pickle.load(open(f'{DATA_ROOT}{self.feature_tag}_{self.pca_tag}_pca.pkl', 'rb'))
        self.pca_appear = pickle.load(open(f'{DATA_ROOT}{self.feature_tag}_{self.pca_tag}_appear_pca.pkl', 'rb'))
        self.pca_optical = pickle.load(open(f'{DATA_ROOT}{self.feature_tag}_{self.pca_tag}_optical_pca.pkl', 'rb'))
        self.pca_skel = pickle.load(open(f'{DATA_ROOT}{self.feature_tag}_{self.pca_tag}_skel_pca.pkl', 'rb'))
        self.pca_emb = pickle.load(open(f'{DATA_ROOT}{self.feature_tag}_{self.pca_tag}_emb_pca.pkl', 'rb'))

    def transform(self, original_feature_array: np.ndarray) -> np.ndarray:
        pca_feature_array_appear = self.pca_appear.transform(original_feature_array[:, :2])
        pca_feature_array_optical = self.pca_optical.transform(original_feature_array[:, 2:4])
        pca_feature_array_skel = self.pca_skel.transform(original_feature_array[:, 4:-100])
        pca_feature_array_emb = self.pca_emb.transform(original_feature_array[:, -100:])
        pca_feature_array = np.hstack([pca_feature_array_appear, pca_feature_array_optical,
                                       pca_feature_array_skel, pca_feature_array_emb])
        return pca_feature_array

    def invert_transform(self, pca_feature_array: np.ndarray) -> np.ndarray:
        indices = [self.pca_appear.n_components,
                   self.pca_appear.n_components + self.pca_optical.n_components,
                   self.pca_appear.n_components + self.pca_optical.n_components + self.pca_skel.n_components,
                   self.pca_appear.n_components + self.pca_optical.n_components +
                   self.pca_skel.n_components + self.pca_emb.n_components]
        original_inverted_array_appear = self.pca_appear.inverse_transform(pca_feature_array[:, :indices[0]])
        original_inverted_array_optical = self.pca_optical.inverse_transform(pca_feature_array[:, indices[0]:indices[1]])
        original_inverted_array_skel = self.pca_skel.inverse_transform(pca_feature_array[:, indices[1]:indices[2]])
        original_inverted_array_emb = self.pca_emb.inverse_transform(pca_feature_array[:, indices[2]:])
        original_inverted_array = np.hstack(
            [original_inverted_array_appear, original_inverted_array_optical,
             original_inverted_array_skel, original_inverted_array_emb])

        return original_inverted_array

if __name__ == "__main__":
    '''how to use'''
    from utils.io import list_fnames
    import seaborn as sns

    DATA_ROOT = '../data/input_scene_vecs/'
    scene_vec_fpattern = '*_kinect_sep_09.pkl'

    fpaths, fnames = list_fnames(DATA_ROOT, scene_vec_fpattern)
    print(f'Number of files found: {len(fpaths)}')

    for i, (fpath, fname) in enumerate(zip(fpaths, fnames)):
        if i > 3:
            break
        # fname = '1.1.1_kinect_sep_09.pkl'
        # fpath = os.path.join(DATA_ROOT, fname)
        data = pickle.load(open(fpath, 'rb'))
        data['combined_resampled_df'].head()


    pca_transformer = PCATransformer()
    # forward transform
    x_train_pca = pca_transformer.transform(data['combined_resampled_df'].to_numpy())
    # backward transform
    x_train_inverted = pca_transformer.invert_transform(x_train_pca)

    sns.heatmap(x_train_pca, cmap='viridis')
