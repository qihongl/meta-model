import numpy as np
import pandas as pd
import dabest


cond_means = {'control': 0, 'test': 2}
scale = 1
n_data = 64
data_dict = {
    cond: np.random.normal(loc=cond_means[cond], scale=scale, size=(n_data,))
    for cond in cond_means.keys()
}


df = pd.DataFrame(data_dict)
df['ids'] = np.arange(n_data)
df.head()

# Load the data into dabest
dabest_data = dabest.load(
    data=df, idx=list(data_dict.keys()), paired=True, id_col='ids'
)
dabest_data.mean_diff.plot(swarm_label='Values')




from sem.event_models import GRUEvent
recurrent_dropout_rate = dropout_rate = .5
optimizer_kwargs = dict(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
f_opts = dict(var_df0=10., var_scale0=0.06, l2_regularization=0.0,
              n_epochs=1, t=4, batch_update=True, n_hidden=15, variance_window=None,
              optimizer_kwargs=optimizer_kwargs,
              recurrent_dropout=recurrent_dropout_rate, dropout=dropout_rate)
new_model = GRUEvent(30, **f_opts)
new_model.init_model()
new_model.do_reset_weights()
# set True so that the model doesn't return the input
new_model.f_is_trained = True


def generate_predictions(scene_vectors, e_hats, weights, n_predictions=1000):
    """
    Generate dropout inferences for a given scene vector and e_hat, for all timesteps.
    """
    optimizer_kwargs = dict(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
    f_opts = dict(var_df0=10., var_scale0=0.06, l2_regularization=0.0,
                  n_epochs=1, t=4, batch_update=True, n_hidden=15, variance_window=None, optimizer_kwargs=optimizer_kwargs,
                  recurrent_dropout=recurrent_dropout_rate, dropout=dropout_rate)
    new_model = GRUEvent(scene_vectors.shape[1], **f_opts)
    new_model.init_model()
    new_model.do_reset_weights()
    # set True so that the model doesn't return the input
    new_model.f_is_trained = True
    # predict 1733 steps for 1734 time-steps because we don't predict for the first step.
    steps = min(scene_vectors.shape[0] - 1, n_predictions)
    x_hat_both_dropout = np.zeros(shape=(steps + 1, 30))
    e_hat_prev = None
    for i in range(steps):
        e_hat = e_hats[i]
        if e_hat != e_hat_prev:
            new_model.x_history = [np.zeros(shape=(0, 30), dtype=np.float64)]
        new_model.set_model_weights(weights[e_hat])
        # predict_next is a wrapper for _predict_next to return the same vector if the model is untrained.
        # setting trained before so it's the same to use either.
        _, both_dropout = new_model.predict_next(scene_vectors[i, :], dropout=True, recurrent_dropout=True)
        # update_estimate=False so that weights and noise_variance is not updated and
        # only concatenate the current vector to history.
        new_model.update(scene_vectors[i, :], scene_vectors[i + 1, :], update_estimate=False)
        x_hat_both_dropout[i + 1, :] = both_dropout
        e_hat_prev = e_hat
    x_hat_both_dropout[0] = scene_vectors[0]
    return x_hat_both_dropout


import numpy as np
import tensorflow as tf
import  tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

input_dim = 30
hidden_dim = 16
n_time_points = 53

X = np.random.normal(size=(n_time_points, input_dim))

model = Sequential()
model.add(GRU(hidden_dim, input_shape=(1, input_dim), dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model.add(Dense(input_dim))

model.compile(optimizer='adam', loss='mse')

seq_length = 1  # This will ensure that we loop over individual time points
def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)

X_arr, y_arr = sliding_windows(X, seq_length)


epochs = 3

for i in range(10):
    hidden_states = []
    losses = []
    for t in range(len(X_arr)):
        input_data = X_arr[t].reshape(1, seq_length, input_dim)
        output_data = y_arr[t].reshape(1, input_dim)

        # train the model with the single data point
        model.train_on_batch(input_data, output_data)

        # evaluate the model with the single data point
        loss = model.evaluate(input_data, output_data, verbose=0)  # verbose=0 means no logging
        losses.append(loss)

        # get the hidden state for the current data point
        # Here, we'll need to use a Keras function to get the hidden state output of the GRU
        get_hidden_state = tensorflow.keras.backend.function([model.layers[0].input], [model.layers[0].output])
        current_hidden_state = get_hidden_state([input_data])[0]
        hidden_states.append(current_hidden_state)

    print(np.mean(losses))

'''GRU pytroch'''

# import torch
# import torch.nn as nn
#
# class GRUNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
#         super(GRUNet, self).__init__()
#
#         self.gru = nn.GRU(input_dim, hidden_dim, num_layers, dropout=dropout, recurrent_dropout=dropout)
#         self.linear = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x, h):
#         out, h = self.gru(x)
#         print(out.size())
#         out = self.linear(out)
#         return out, h
#
# input_dim = 60
# output_dim = 30
# hidden_dim = 16
# x = torch.randn(1, 1, input_dim)
# h = torch.randn(1, 1, hidden_dim)
#
# model = GRUNet(input_dim, hidden_dim, output_dim, dropout=0.5)
# out, h = model(x, h)
#


# import os
# import pickle
# import pandas as pd
# from utils.io import pickle_load, list_fnames, pickle_save
# # from sklearn.decomposition import PCA
#
# DATA_ROOT = '../data'
#
# print(pickle.format_version)
#
# list_fnames('../data/input_scene_vecs', '*.pkl')
#
# fname = 'input_scene_vecs/1.1.1_kinect_sep_09.pkl'
# fpath = os.path.join(DATA_ROOT, fname)
# data = pickle.load(open(fpath, 'rb'))
# print(type(data))
# for k, v in data.items():
#     print(k, type(v), len(v.columns))
#     print(v.head())
#     print('-'*150)
#
# '''
# combined_resampled_df
# - everything stacked together, same sampling rate
# -
# '''
#
# fname = 'input_scene_vecs/sep_09_all_appear_pca.pkl'
# fpath = os.path.join(DATA_ROOT, fname)
# data = pickle.load(open(fpath, 'rb'))
#
# print(type(data))
# for k, v in data.items():
#     print(k, type(v))
#     print(v.head())
#
#
# fname = 'pca/dec_6_rotated_skel_all_30_appear_pca.pkl'
# fpath = os.path.join(DATA_ROOT, fname)
# # data = pickle_load(fpath, to_nparray=False)
# pca = pickle.load(open(fpath, 'rb'))
# print(type(pca))
# print(pca.components_)
# print(pca.explained_variance_)
#
# fname = 'high_level_events/event_annotation_timing.csv'
# fpath = os.path.join(DATA_ROOT, fname)
# df = pd.read_csv(fpath)
# df.head()
#
#
# fname = 'object_labels/1.1.1_C1_labels.csv'
# fpath = os.path.join(DATA_ROOT, fname)
# df = pd.read_csv(fpath)
# df.head()
#
# fname = 'object_labels/1.1.1_kinect_labels.csv'
# fpath = os.path.join(DATA_ROOT, fname)
# df = pd.read_csv(fpath)
# df.head()
#
# '''
# event annot timing
#
# input scene vecs
#
# pca
# - what are all the files
# - appear - whether a new obj has appeared?
# - emb - ?
# - optical - frame by frame difference?
# - the last 2
#
# object labels / object tracking
# - what's c1 vs c2, where are the png files
#
# are all videos in Full_SEM?
# '''
#
#
#
# import pickle
# import numpy as np
# import random
# from utils import get_point_biserial
# # from utils import to_np, to_pth, split_video_id, context_to_bound_vec, \
# #     loss_to_bound_vec, save_ckpt, pickle_save, pickle_load, compute_stats, \
#
#
# a = dict({'hello': 'world', 'a': [np.zeros(3,), np.ones(4)]})
#
# pickle_save(a, 'filename.pickle')
# b = pickle_load('filename.pickle')
# print(a == b)
#
#
# n_events = 3
# event_lens = [int(np.random.uniform(10, 30)) for _ in range(n_events)]
#
# phuman_bounds_list = [np.random.uniform(size=(l,)) for l in event_lens]
# model_bounds_list = [np.array(chbs_i > .8, dtype=np.float) for chbs_i in phuman_bounds_list]
#
#
# def compute_corr_with_perm(n_perms = 50):
#     r, _ = get_point_biserial(
#         np.concatenate(model_bounds_list), np.concatenate(phuman_bounds_list)
#     )
#     r_perm = np.zeros(n_perms, )
#     for i in range(n_perms):
#         random.shuffle(model_bounds_list)
#         r_perm[i], _ = get_point_biserial(
#             np.concatenate(model_bounds_list), np.concatenate(phuman_bounds_list)
#         )
#     return r, r_perm
