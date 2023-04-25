"""
a shortcut for predicting the context
"""
import numpy as np
NULL_RESPONSE = -1
N_WARMUP = 5

class SimpleMemory():

    def __init__(self, input_dim, d, lr=.1):
        '''d is the distance threshold that lead to narrow generalization gradient'''
        self.input_dim = input_dim
        self.d = d
        self.lr = lr
        self.flush_buffer()

    def flush_buffer(self):
        self.X = []
        self.Y = []

    def add_data(self, x_t, y_t):
        if len(self.X) == 0:
            self.X.append(x_t)
            self.Y.append(y_t)
        else:
            distances = compute_distances(self.X, x_t)
            if np.any(distances) < self.d:
                min_distance_id = distances.argmin()
                self.X[min_distance_id] = self.X[min_distance_id] * (1-self.lr) + x_t * self.lr
            else:
                self.X.append(x_t)
                self.Y.append(y_t)

    def predict(self, x_t):
        if len(self.X) < N_WARMUP:
            return NULL_RESPONSE
        # compute distances towards all exisiting centroids
        distances = compute_distances(self.X, x_t)
        # within d, get all LCs
        pts_within_d = distances < self.d
        unique_LCs = np.unique(np.array(self.Y)[pts_within_d])
        # if uniqueness is statisfied
        if len(unique_LCs) == 1:
            return unique_LCs
        return NULL_RESPONSE



def compute_distances(pts, q):
    '''
    pts: (n x d) array representing n pts with dim d (each row is a pt)
    q: (1 x d) or (d,)

    e.g.,
    n = 3
    d = 2
    pts = np.random.normal(size=(n,d))
    q = np.random.normal(size=(d, ))
    distances = compute_distances(pts, q)
    '''
    assert np.shape(pts)[1] == len(q), 'dim(q) and dim(pts) must match'
    pts = np.asarray(pts)
    return np.linalg.norm(pts-q, axis=1)



if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import itertools
    from copy import deepcopy
    from scipy.spatial import distance_matrix

    sns.set(style='white', palette='colorblind', context='poster')
    # torch.manual_seed(0)
    # np.random.seed(0)
    cpal = sns.color_palette()
    input_dim = 2
    n_samples = 200
    cluster_std = 1
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

    sns.clustermap(distance_matrix(data, data))

    d = .2
    lr = .1
    for d in [.1, .2, .4, 1, 2, 4]:
            # use_model = False

        sm = SimpleMemory(input_dim=input_dim, d=d, lr=lr)
        Y_hat = np.zeros((n_samples, n_samples))
        acc = np.zeros((n_samples, ))
        mis = np.zeros((n_samples, ))
        dnk = np.zeros((n_samples, ))
        for i, (data_i, label_i) in enumerate(zip(data, labels)):
            sm.add_data(data_i, label_i)

            Y_hat[i, :] = [sm.predict(x) for x in data_test]
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


                # # sm.model
                # if (i+1) % 20 == 0:
                #     print(len(sm.X))
                #     sm.update_model()
                #     print(len(sm.X))
                #     f, ax = plt.subplots(1,1, figsize=(10,8))
                #     alpha = .5
                #     ax.scatter(data[:i,0], data[:i,1], c=[cpal[l] for l in labels[:i]], alpha=alpha)
                #     sns.despine()
                #     ax.scatter(x=sm.model[0][0], y=sm.model[0][1], marker='x', c=[cpal[0]])
                #     ax.scatter(x=sm.model[1][0], y=sm.model[1][1], marker='x', c=[cpal[1]])
                #     ax.set_title(f'n samples = {i+1}')
