import numpy as np
from scipy.stats import sem
NULL_RESPONSE = -1

class SimpleTracker():

    def __init__(self, size=256):
        self.buffer = {}
        self.accurate = {}
        self.size = size

    def add(self, ctx_id, value, verbose=False):
        # if input is new, create new slots
        if not self.ctx_in_tracker(ctx_id):
            self.reinit_ctx_buffer(ctx_id)
        # add new data point to the buffer
        self.buffer[ctx_id].append(value)
        # if buffer is full
        n_items_in_buffer = len(self.buffer[ctx_id])
        if n_items_in_buffer == self.size:
            if np.sum(self.buffer[ctx_id]) == self.size:
                self.accurate[ctx_id] = True
            else:
                self.accurate[ctx_id] = False
        # if buffer overflow
        elif n_items_in_buffer > self.size:
            self.buffer[ctx_id].pop(0)

    def use_shortcut_t(self, ctx_id):
        if not self.ctx_in_tracker(ctx_id):
            return False
        return self.accurate[ctx_id]

    def reinit_ctx_buffer(self, ctx_id):
        self.buffer[ctx_id] = []
        self.accurate[ctx_id] = False

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

    def peaked(self, ctx_id, n_std, value):
        assert n_std > 0
        if not self.ctx_in_tracker(ctx_id):
            return None
        return value > self.get_mean(ctx_id) + n_std * self.get_std(ctx_id)

    def ctx_in_tracker(self, ctx_id):
        return ctx_id in self.buffer.keys()

    def list_accuracy(self):
        return [v for k,v in self.buffer.items()]


if __name__ == "__main__":
    '''how to use'''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    sns.set(style='white', palette='colorblind', context='poster')

    size = 8
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
        pet.add(c2, 1)
        pet.buffer


    print()
    pet.get_mean(1)
    pet.get_std(1)

    pet.buffer[1]
    pet.accurate
