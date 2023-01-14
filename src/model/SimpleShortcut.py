"""
a shortcut for predicting the context
"""
import numpy as np


class SimpleShortcut():

    def __init__(self, input_dim, d, lr=.5, use_model=False):
        '''d is the distance threshold that lead to narrow generalization gradient'''
        self.input_dim = input_dim
        self.d = d
        self.lr = lr
        self.use_model = use_model
        self.reinit()

    def reinit(self):
        self.flush_buffer()
        self.model = {}

    def flush_buffer(self):
        self.X = []
        self.Y = []

    def add_data_list(self, x_list, y_list):
        self.X.extend(x_list)
        self.Y.extend(y_list)

    def add_data(self, x_t, y_t, update_model=False):
        self.X.append(x_t)
        self.Y.append(y_t)

    def update_model(self):
        if len(self.Y) == 0:
            raise Warning('empty shortcut buffer')
            pass
        X = np.array(self.X)
        Y = np.array(self.Y)
        for y in np.unique(Y):
            mask = y == Y
            # save the cluster center
            if y not in self.model.keys():
                self.model[y] = np.mean(X[mask,:], axis=0)
            else:
                self.model[y] = self.model[y] * (1-self.lr) + np.mean(X[mask,:], axis=0) * self.lr
        self.flush_buffer()

    def get_cluster_centers(self):
        return np.stack(self.model.values())

    def get_cluster_ids(self):
        return list(self.model.keys())

    def predict(self, x_t):
        if self.use_model:
            return self.cluster_based_predict(x_t)
        else:
            return self.instance_based_predict(x_t)

    def instance_based_predict(self, x_t):
        if len(self.Y) == 0: 
            return None
        distances = norm_1vec_vs_nvecs(x_t, self.X)
        closest_d_i = np.argmin(distances)
        # if the min dist cluster is closer than d, return its id
        if distances[closest_d_i] < self.d:
            return self.Y[closest_d_i]
        return None

    def cluster_based_predict(self, x_t):
        if len(ssc.model) == 0:
            return None
        cluster_ids = self.get_cluster_ids()
        distances = norm_1vec_vs_nvecs(x_t, self.get_cluster_centers())
        closest_d_i = np.argmin(distances)
        if distances[closest_d_i] > self.d:
            return None
        return cluster_ids[closest_d_i]


def norm_1vec_vs_nvecs(vector, vectors, ord=2):
    '''compute the distance between 1 vector vs. a bunch of vectors'''
    assert len(vector) == np.shape(vectors)[1]
    x_t_rep = np.tile(vector, (np.shape(vectors)[0], 1))
    return np.linalg.norm(x_t_rep - vectors, axis=1, ord=ord)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import itertools
    from copy import deepcopy
    sns.set(style='white', palette='colorblind', context='poster')
    # torch.manual_seed(0)
    # np.random.seed(0)
    cpal = sns.color_palette()
    input_dim = 2
    data, labels = make_blobs(n_samples=100, centers=2, n_features=input_dim, random_state=0)

    print(data.shape, labels.shape)

    d = 2
    lr = .5
    use_model = False

    ssc = SimpleShortcut(input_dim=input_dim, d=d, lr=lr, use_model=use_model)

    models = []
    for i, (data_i, label_i) in enumerate(zip(data, labels)):
        ssc.add_data(data_i, label_i)

        if (i+1) % 20 == 0:
            ssc.update_model()
            models.append(deepcopy(ssc.model))


    f, ax = plt.subplots(1,1, figsize=(10,8))
    alpha = .5
    ax.scatter(data[:,0], data[:,1], c=[cpal[l] for l in labels], alpha=alpha)
    sns.despine()

    for model in models:
        ax.scatter(x=model[0][0], y=model[0][1], marker='x', c=[cpal[0]])
        ax.scatter(x=model[1][0], y=model[1][1], marker='x', c=[cpal[1]])
