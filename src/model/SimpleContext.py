import numpy as np
from copy import deepcopy

class SimpleContext():

    def __init__(self, context_dim, stickiness, concentration, try_reset_h):
        self.context_dim = context_dim
        self.stickiness = stickiness
        self.concentration = concentration
        self.try_reset_h = try_reset_h
        self.prev_cluster_id = None
        self.n_context = None
        self.context = []
        self.frozen = False

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
        self.context = [self.zero_vector()]
        # self.context = [self.random_vector()]
        self.prev_cluster_id = 1
        c_id, c_vec = self.add_new_context()
        return c_id, c_vec

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False


    def add_new_context(self):
        '''
        sample a random vector to represent the k-th context
        this is useful because
        - random vectors are easy to get
        - random vectors are roughly orthogonal
        '''
        # set ctx 0 to be the added context
        self.context.append(self.random_vector())
        # self.context.append(novel_context)
        self.n_context += 1
        return self.n_context, self.context[-1]


    def compute_posterior(self, likelihood, verbose=False):
        assert np.all(likelihood) >= 0, f'likelihood must be non-neg, but received {likelihood}'
        if self.try_reset_h:
            assert len(likelihood) == self.n_context + 2, f'dim(likelihood) must = # contexts, but len(likelihood) = {len(likelihood)}, n_context = {self.n_context}'
        else:
            assert len(likelihood) == self.n_context + 1, f'dim(likelihood) must = # contexts, but len(likelihood) = {len(likelihood)}, n_context = {self.n_context}'

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
        reset_h = False
        posterior = self.compute_posterior(likelihood)
        if self.frozen:
            posterior[0] = -777
        # find the max posterior context id
        max_pos_cid = np.argmax(posterior)
        # if infer a new LC
        if max_pos_cid == 0:
            max_pos_cid, _ = self.add_new_context()
            reset_h = True
            if verbose >= 1:
                print(f'adding the {self.n_context}-th context! RESET H! c = {self.context[max_pos_cid][:3]}!')

        elif self.try_reset_h and max_pos_cid == len(likelihood) - 1: # if it is the last index
            # print(likelihood)
            # print(posterior)
            # print(max_pos_cid, self.prev_cluster_id)
            # assert max_pos_cid == self.prev_cluster_id
            max_pos_cid = self.prev_cluster_id
            reset_h = True
            if verbose >= 1:
                print(f'restart the {max_pos_cid}-th context! RESET H! c = {self.context[max_pos_cid][:3]}!')
        elif max_pos_cid == self.prev_cluster_id:
            if verbose >= 2:
                print(f'keep using the {max_pos_cid}-th context!')
        else:
            reset_h = True
            if verbose >= 1:
                print(f'switch to {max_pos_cid}-th context! RESET H! c = {self.context[max_pos_cid][:3]}!')

        if verbose >= 3:
            print(f'lik: {likelihood}\nposterior: {posterior}')

        # update the previous context
        self.prev_cluster_id = max_pos_cid
        return max_pos_cid, reset_h

    def prev_ctx(self):
        return self.context[self.prev_cluster_id]

    def random_vector(self, loc=0, scale=1):
        return np.random.normal(loc=loc, scale=scale, size=(self.context_dim,))

    def zero_vector(self):
        return np.zeros(size=(self.context_dim,))

    def zero_vector(self):
        return np.zeros(self.context_dim,)

    def __repr__(self):
        repr = f'n context vectors: {self.n_context} \n'
        repr += f'context vectors: \n{self.context}'
        return repr



if __name__ == "__main__":
    '''how to use'''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    sns.set(style='white', palette='colorblind', context='poster')

    context_dim= 16
    stickiness = 8
    concentration = 1
    try_reset_h = 0
    sc = SimpleContext(context_dim, stickiness, concentration, try_reset_h)
    c_id, c_vec = sc.init_context()
    sc.context[0]
    print(sc.prev_cluster_id)


    # pe = np.array([.8, .3])
    # c_it = sc.assign_context(-pe, verbose=True)
    # sc.prev_cluster_id
    # sc.context


    # sc.assign_context
    # sc.n_context
    sc.assign_context([.999, .1], verbose=3)
    sc.assign_context([.1, .1, .1], verbose=3)

    sc.assign_context([.1, .1, .1, .1, .1], verbose=3)


    # sc.context
# sc.context[sc.prev_cluster_id]
