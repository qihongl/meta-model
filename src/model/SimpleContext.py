import numpy as np


class SimpleContext():

    def __init__(self, context_dim, stickiness, concentration, try_reset_h):
        self.context_dim = context_dim
        self.stickiness = stickiness
        self.concentration = concentration
        self.try_reset_h = try_reset_h
        self.prev_cluster_id = None
        self.n_context = None
        self.context = []

    def to_dict(self):
        return dict({
            'context_dim': self.context_dim,
            'stickiness': self.stickiness,
            'concentration': self.concentration,
            'try_reset_h': self.try_reset_h,
            'prev_cluster_id': self.prev_cluster_id,
            'n_context': self.n_context,
            'context': self.context
        })

    def init_from_dict(self, input_dict):
        # make sure the loaded params match
        assert self.context_dim == input_dict['context_dim']
        assert self.stickiness == input_dict['stickiness']
        assert self.concentration == input_dict['concentration']
        assert self.try_reset_h == input_dict['try_reset_h']
        self.context = input_dict['context']
        self.n_context = len(self.context)
        self.prev_cluster_id = None


    def init_context(self):
        self.n_context = 0
        self.context = [self.random_vector()]
        self.prev_cluster_id = 1
        c_id, c_vec = self.add_new_context()
        return c_id, c_vec


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
        sticky_uniform_vec = np.ones(len(likelihood),)
        sticky_uniform_vec[0] = self.concentration
        if self.prev_cluster_id is not None:
            sticky_uniform_vec[self.prev_cluster_id] += self.stickiness
            # if try reset h then the last dim is the resetted context
            if self.try_reset_h:
                sticky_uniform_vec[-1] += self.stickiness
        prior = sticky_uniform_vec / np.sum(sticky_uniform_vec)
        if verbose:
            print('prior = ', prior)
        return likelihood * prior

    def assign_context(self, likelihood, verbose=1):
        # if np.any(likelihood) <= 0: print(likelihood)
        assert np.all(likelihood) >= 0, f'invalid likelihood {likelihood}'
        reset_h = False
        posterior = self.compute_posterior(likelihood)
        max_pos_cid = np.argmax(posterior)
        # get the context vector
        if max_pos_cid == 0:
            max_pos_cid, max_pos_cvec = self.add_new_context()
            if verbose >= 1:
                print(f'adding the {self.n_context}-th context!')
            if verbose >= 2:
                print(f'lik: {likelihood}\nposterior: {posterior}')
        elif self.try_reset_h and max_pos_cid == len(likelihood) - 1: # if it is the last index
            max_pos_cid = self.prev_cluster_id
            # max_pos_cvec = self.context[self.prev_cluster_id]
            reset_h = True
            if verbose >= 1:
                print(f'restart the {max_pos_cid}-th context!')
            if verbose >= 2:
                print(f'lik: {likelihood}\nposterior: {posterior}')
        else:
            # max_pos_cvec = self.context[max_pos_cid]
            if verbose >= 2:
                print(f'posterior: {posterior} \t reusing {max_pos_cid}-th context!')
        # update the previous context
        self.prev_cluster_id = max_pos_cid
        return max_pos_cid, reset_h

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
    stickiness = .5
    concentration = 1
    try_reset_h = 0
    sc = SimpleContext(context_dim, stickiness, concentration, try_reset_h)
    sc.init_context()
    sc.to_dict()
    # c_it, ctx_it = sc.add_new_context()
    #
    # pe = np.array([.8, .3])
    #
    # c_it = sc.assign_context(-pe, verbose=True)
    # sc.prev_cluster_id
