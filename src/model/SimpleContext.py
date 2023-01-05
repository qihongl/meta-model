import numpy as np


class SimpleContext():

    def __init__(self, context_dim, penalty_new_context, stickiness):
        self.context_dim = context_dim
        self.penalty_new_context = penalty_new_context
        self.stickiness = stickiness
        self.prev_cluster_id = None
        self.n_context = None
        self.context = []

    def to_dict(self):
        return dict({
            'context_dim': self.context_dim,
            'penalty_new_context': self.penalty_new_context,
            'stickiness': self.stickiness,
            'prev_cluster_id': self.prev_cluster_id,
            'n_context': self.n_context,
            'context': self.context
        })

    def init_from_dict(self, input_dict):
        # make sure the loaded params match
        assert self.context_dim == input_dict['context_dim']
        assert self.penalty_new_context == input_dict['penalty_new_context']
        assert self.stickiness == input_dict['stickiness']
        self.context = input_dict['context']
        self.prev_cluster_id = None
        self.n_context = len(self.context)


    def init_context(self):
        self.n_context = 0
        self.context = [self.random_vector()]
        self.prev_cluster_id = 1
        c_id, c_vec = self.add_new_context()
        return c_id, c_vec

    def penalty_new_context(self, pe):
        pe[0] = pe[0] + self.penalty_new_context
        return pe

    def add_stickiness_bonus(self, pe):
        if self.prev_cluster_id is None:
            return pe
        pe[self.prev_cluster_id] = pe[self.prev_cluster_id] - self.stickiness
        return pe

    def add_new_context(self):
        '''
        sample a random vector to represent the k-th context
        this is useful because
        - random vectors are easy to get
        - random vectors are roughly orthogonal
        '''
        new_context = self.random_vector()
        # new_context = self.ortho_mat[self.n_context+1]
        self.context.append(new_context)
        self.n_context += 1
        return self.n_context, new_context

    def compute_posterior(self, likelihood, verbose=False):
        sticky_uniform_vec = np.ones(self.n_context+1)
        if self.prev_cluster_id is not None:
            sticky_uniform_vec[self.prev_cluster_id] += self.stickiness
        prior = sticky_uniform_vec / np.sum(sticky_uniform_vec)
        if verbose:
            print('prior = ', prior)
        return likelihood * prior

    def assign_context(self, likelihood, get_context_vector=False, verbose=1):
        assert np.all(likelihood) > 0
        posterior = self.compute_posterior(likelihood)
        ctx_id = np.argmax(posterior)
        # get the context vector
        if ctx_id == 0:
            ctx_id, ctx_vec = self.add_new_context()
            if verbose >= 1:
                print(f'adding the {self.n_context}-th context!')
                print(f'lik: {likelihood}\nposterior: {posterior}')
        else:
            ctx_vec = self.context[ctx_id]
            if verbose >= 2:
                print(f'posterior: {posterior} \t reusing {ctx_id}-th context!')
        self.prev_cluster_id = ctx_id
        if get_context_vector:
            return ctx_id, ctx_vec
        return ctx_id

    def prev_ctx(self):
        return self.context[self.prev_cluster_id]

    def random_vector(self, loc=0, scale=1):
        return np.random.normal(loc=loc, scale=scale, size=(self.context_dim,))

    def zero_vector(self):
        return np.zeros(self.context_dim,)



if __name__ == "__main__":
    '''how to use'''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    sns.set(style='white', palette='colorblind', context='poster')

    context_dim=32
    penalty_new_context, stickiness = .5, .5

    sc = SimpleContext(context_dim, penalty_new_context, stickiness)
    sc.init_context()
    sc.to_dict()
    # c_it, ctx_it = sc.add_new_context()
    #
    # pe = np.array([.8, .3])
    #
    # c_it = sc.assign_context(-pe, verbose=True)
    # sc.prev_cluster_id
