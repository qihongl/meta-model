"""
a shortcut for predicting the context
"""
from sklearn.neighbors import RadiusNeighborsClassifier
# from sklearn.neighbors import NearestNeighbors
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Outlier label -1 is not in training classes.")
NULL_RESPONSE = -1
N_WARMUP = 5

class SimpleMemory():

    def __init__(self, input_dim, d, majority_p=0):
        '''d is the distance threshold that lead to narrow generalization gradient'''
        self.input_dim = input_dim
        self.d = d
        # self.majority_p = majority_p
        self.flush_buffer()
        self.rnc = RadiusNeighborsClassifier(radius = d, weights='distance', metric='cosine', outlier_label=NULL_RESPONSE)
        # self.rnc = RadiusNeighborsClassifier(radius = d, weights='distance', outlier_label=NULL_RESPONSE)
        # self.rnc = NearestNeighbors(n_neighbors=99, radius=d)

    def flush_buffer(self):
        self.X = []
        self.Y = []

    def add_data(self, x_t, y_t):
        self.X.append(x_t)
        self.Y.append(y_t)
        self.rnc.fit(np.array(self.X), self.Y)

    def predict(self, x_t):
        if len(self.X) < N_WARMUP:
            return NULL_RESPONSE
        return self.rnc.predict(x_t.reshape(1,-1))

    def is_not_null(self, inferred_ctx):
        if inferred_ctx == NULL_RESPONSE:
            return False
        return True


    # def rnc_predict(self, x_t):
    #     if len(self.X) < N_WARMUP:
    #         return NULL_RESPONSE
    #     # get the data ids within the radius
    #     _, ids = self.rnc.radius_neighbors(x_t.reshape(1,-1))
    #     ids = np.asarray(ids[0])
    #     if len(ids) > 0:
    #         # find all within-radius LC
    #         radius_neighbors_LCs = np.array(self.Y)[ids]
    #         # get most common LC (mcLC) count info
    #         counter = Counter(radius_neighbors_LCs)
    #         mcLC_id, mcLC_count = counter.most_common()[0]
    #         p = mcLC_count / len(ids)
    #         if p > self.majority_p:
    #             return mcLC_id
    #     return NULL_RESPONSE


    # def rnc_predict(self, x_t):
    #     if len(self.X) < N_WARMUP:
    #         return NULL_RESPONSE
    #     y_hat_t = self.rnc.predict(x_t.reshape(1,-1))
    #     p_y_hat_t = self.rnc.predict_proba(x_t.reshape(1,-1))[0]
    #     # print(p_y_hat_t)
    #     if p_y_hat_t[0] > self.majority_p:
    #         return y_hat_t
    #     return NULL_RESPONSE

    # def predict(self, x_t):
    #     if len(self.X) < N_WARMUP:
    #         return NULL_RESPONSE
    #     # compute distances towards all exisiting centroids
    #     distances = compute_distances(self.X, x_t)
    #     # within d, get all LCs
    #     pts_within_d = distances < self.d
    #     unique_LCs = np.unique(np.array(self.Y)[pts_within_d])
    #     # if uniqueness is statisfied
    #     if len(unique_LCs) == 1:
    #         return unique_LCs
    #     return NULL_RESPONSE
#
#
# def compute_distances(pts, q):
#     '''
#     pts: (n x d) array representing n pts with dim d (each row is a pt)
#     q: (1 x d) or (d,)
#
#     e.g.,
#     n = 3
#     d = 2
#     pts = np.random.normal(size=(n,d))
#     q = np.random.normal(size=(d, ))
#     distances = compute_distances(pts, q)
#     '''
#     assert np.shape(pts)[1] == len(q), 'dim(q) and dim(pts) must match'
#     pts = np.asarray(pts)
#     return np.linalg.norm(pts-q, axis=1)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import itertools
    from copy import deepcopy
    from scipy.spatial import distance_matrix

    sns.set(style='white', palette='colorblind', context='poster')
    np.random.seed(0)

    cpal = sns.color_palette()

    input_dim = 2
    n_samples = 300
    cluster_std = 2
    data_all, labels_all = make_blobs(n_samples=n_samples*2, centers=3, n_features=input_dim, cluster_std=cluster_std, random_state=0)
    data, data_test = data_all[:n_samples,:], data_all[n_samples:,:]
    labels, labels_test = labels_all[:n_samples], labels_all[n_samples:]

    print(data.shape, labels.shape)

    def plot_labeled_scatter(x, y, labels):
        """
        Creates a scatter plot of x and y values with color-coded points based on a 1D label array.

        Parameters:
            x (ndarray): Array of x values for each point.
            y (ndarray): Array of y values for each point.
            labels (ndarray): Array of labels for each point. Must be same length as x and y.
        """
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab10', len(unique_labels)) # get a colormap for the number of unique labels
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, label in enumerate(unique_labels):
            mask = (labels == label) # mask to select points with current label
            ax.scatter(x[mask], y[mask], c=colors(i), label=label)
        ax.legend()
        return fig, ax

    fig, ax = plot_labeled_scatter(data[:,0], data[:,1], labels)
    ax.set_title('data')
    sns.despine()
    # sns.clustermap(distance_matrix(data, data))


    lr = .2
    majority_p = .75
    # d = .2
    ds = [.1, .4, 2, 8]
    # ds = [2]
    # ds = [.1, 8]

    for d in ds:
        # np.array(sm.X), np.array(sm.Y)
        sm = SimpleMemory(input_dim=input_dim, d=d, majority_p=majority_p, lr=lr)
        Y_hat = np.zeros((n_samples, n_samples))
        acc = np.zeros((n_samples, ))
        mis = np.zeros((n_samples, ))
        dnk = np.zeros((n_samples, ))
        for i, (data_i, label_i) in enumerate(zip(data, labels)):
            sm.add_data(data_i, label_i)

            # Y_hat[i, :] = [sm.predict(x) for x in data_test]
            Y_hat[i, :] = [sm.rnc_predict(x) for x in data_test]
            n_recalls = np.sum(Y_hat[i,:]!=-1)
            correct_recalls = np.logical_and(Y_hat[i,:]!=-1, Y_hat[i,:] == labels_test)
            incorrect_recalls = np.logical_and(Y_hat[i,:]!=-1, Y_hat[i,:] != labels_test)

            # acc[i] = np.sum(Y_hat[i,:] == labels) / n_samples
            acc[i] = np.sum(correct_recalls) / n_samples
            mis[i] = np.sum(incorrect_recalls) / n_samples
            dnk[i] = (n_samples - n_recalls) / n_samples


        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(acc,color = 'k')

        ax.plot(acc+mis,color = 'k')
        ax.plot(np.ones_like(acc),color = 'k')
        ax.fill_between(np.arange(n_samples), np.zeros_like(acc), acc, alpha=0.2, label='acc', color='green')
        ax.fill_between(np.arange(n_samples), acc, acc+mis, alpha=0.2, label='err', color='red')
        ax.fill_between(np.arange(n_samples), acc, acc+mis+dnk, alpha=0.2, label='dnk', color='blue')
        ax.set_title(f'd = {d}, acc = %.2f, mis = %.2f' % (np.mean(acc), np.mean(mis)))
        sns.despine()
        ax.legend()
