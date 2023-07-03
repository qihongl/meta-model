import numpy as np
from scipy.stats import sem
NULL_RESPONSE = -1

class SimpleTracker():

    def __init__(self, size=256):
        self.buffer = {}
        self.unique = {}
        self.size = size

    def add(self, ctx_id, value, verbose=False):
        # if input is new, create new slots
        if not self.ctx_in_tracker(ctx_id):
            self.reinit_ctx_buffer(ctx_id)
        # add new data point to the buffer
        self.buffer[ctx_id].append(value)
        # if buffer is full
        if self.buffer_full(ctx_id):
            self.buffer[ctx_id].pop(0)
        # else:
        #     print('buffer')
        #     print(self.buffer[ctx_id])
        #     print('unique:')
        #     print(np.unique(self.buffer[ctx_id]))
        #     print('len:')
        #     print(len(np.unique(self.buffer[ctx_id])))
        #     if len(np.unique(self.buffer[ctx_id])) == 1:
        #         self.unique[ctx_id] = True
        #     else:
        #         self.unique[ctx_id] = False

    def is_unique(self, ctx_id):
        if not self.ctx_in_tracker(ctx_id):
            return False
        return self.unique[ctx_id]

    def reinit_ctx_buffer(self, ctx_id):
        self.buffer[ctx_id] = []
        self.unique[ctx_id] = False
        # print(f'context {ctx_id} reinitialized!')

    def buffer_full(self, ctx_id):
        return len(self.buffer[ctx_id]) == self.size

    def get_recent(self, ctx_id, n=1):
        if not self.ctx_in_tracker(ctx_id):
            return None
        return np.mean(self.buffer[ctx_id][-n:])

    def get_mean(self, ctx_id):
        if not self.ctx_in_tracker(ctx_id):
            return None
        return np.mean(self.buffer[ctx_id])

    def get_se(self, ctx_id):
        if not self.ctx_in_tracker(ctx_id):
            return None
        return sem(self.buffer[ctx_id])

    def get_std(self, ctx_id):
        if not self.ctx_in_tracker(ctx_id):
            return None
        return np.std(self.buffer[ctx_id])

    def get_z_stats(self, ctx_id, value):
        if not self.ctx_in_tracker(ctx_id):
            return None
        # compute the z stats
        z = (value - np.mean(self.buffer[ctx_id])) / np.std(self.buffer[ctx_id])
        return z

    def peaked(self, ctx_id, n_std, value):
        assert n_std > 0
        if not self.ctx_in_tracker(ctx_id):
            return None
        return value > self.get_mean(ctx_id) + n_std * self.get_std(ctx_id)

    def ctx_in_tracker(self, ctx_id):
        return ctx_id in self.buffer.keys()

    def list_accuracy(self):
        return [v for k,v in self.buffer.items()]

    def get_sigma(self, ctx_id):
        if not self.ctx_in_tracker(ctx_id):
            return np.ones(30, )
        return map_variance(np.stack(self.buffer[ctx_id]))


def map_variance(samples, nu0=10, var0=.06):
    """
    This estimator assumes an scaled inverse-chi squared prior over the
    variance and a Gaussian likelihood. The parameters d and scale
    of the internal function parameterize the posterior of the variance.
    Taken from Bayesian Data Analysis, ch2 (Gelman)
    samples: N length array or NxD array, where N is the number of
             samples and D is the dimensions
    nu0: prior degrees of freedom
    var0: prior scale parameter
    returns: float or D-length array, mode of the posterior
    ## Calculation ##
    the posterior of the variance is thus (Gelman, 2nd edition, page 50):
        p(var | y) ~ Inv-X^2(nu0 + n, (nu0 * var0 + n * v) / (nu0 + n) )
    where n is the sample size and v is the empirical variance.  The
    mode of this posterior simplifies to:
        mode(var|y) = (nu0 * var0 + n * v) / (v0 + n + 2)
    which is just a weighted average of the two modes
    """
    # get n and v from the data
    n = np.shape(samples)[0]
    v = np.var(samples, axis=0)
    mode = (nu0 * var0 + n * v) / (nu0 + n + 2)
    return mode


if __name__ == "__main__":
    '''how to use'''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    sns.set(style='white', palette='colorblind', context='poster')

    size = 32
    pet = SimpleTracker(size=size)
    print(pet.buffer)
    print()

    [c1, c2] = [0, 1]
    sc_ctx = [c1, c2, c1, c2]
    buffer = [0, 1, 0, 1]

    for sc_ctx_i, pe_i in zip(sc_ctx, buffer):
        pet.add(sc_ctx_i, pe_i)
        print(pet.buffer)

    for i in range(10):
        pet.add(c2, np.random.normal())
        pet.buffer[1]


    print()
    pet.get_mean(1)
    pet.get_std(1)
    pet.get_z_stats(1, 2)

    x = np.unique(pet.buffer[1])
    len(x)
    pet.unique
