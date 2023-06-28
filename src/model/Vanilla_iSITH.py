import torch
from torch import nn as nn
from deepsith import iSITH

class Vanilla_iSITH(nn.Module):
    def __init__(
            self, ntau = 5, tau_min = 1, tau_max = 40, buff_max = 80,
            k = 10, dt = 1, g = 0, ttype=torch.FloatTensor
        ):
        super(Vanilla_iSITH, self).__init__()
        self.ntau = ntau
        self.iSITH = iSITH(
            tau_min=tau_min, tau_max=tau_max, buff_max=buff_max, k=k, ntau=ntau,
            dt=dt, g=g, ttype=ttype
        )

    def forward(self, x):
        return self.iSITH(x)

    def simple_forward(self, x, unroll=True):
        y = self.iSITH(x)
        assert y.size()[0] == 1
        y = torch.squeeze(y)
        if unroll:
            return y[:,:,-1].view(-1)
        return y[:,:,-1]


if __name__ == "__main__":
    '''how to use'''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    sns.set(style='white', palette='colorblind', context='poster')

    ntau = 5
    isith = Vanilla_iSITH(ntau = ntau)

    in_features = 30
    out_features = 30
    # data
    # x = torch.zeros(1,1,in_features,20)
    # x[:,:,0,:] = 1

    X = torch.eye(in_features)
    seq_len = 20
    x = X[:, :seq_len].view((1, 1, in_features, seq_len))
    np.shape(x)


    y = isith.forward(x)
    y_simple = isith.simple_forward(x, unroll=False)
    y_simple_unrollled = isith.simple_forward(x)


    f, ax = plt.subplots(1,1, figsize=(5,7))
    ax.imshow(torch.squeeze(x).data.T)
    ax.set_title('input')
    ax.set_ylabel('time')
    ax.set_xlabel('feature dim')
    ax.set_yticks(np.arange(5, seq_len+5, 5)-1)
    ax.set_yticklabels(np.flip(-np.arange(0, seq_len, 5)))

    for i in range(ntau):
        f, ax = plt.subplots(1,1, figsize=(5,7))
        ax.imshow(torch.squeeze(y).data[i].T)
        # ax.set_title('output')
        ax.set_ylabel('time')
        ax.set_xlabel('feature dim')
        ax.set_yticks(np.arange(5, seq_len+5, 5)-1)
        ax.set_yticklabels(np.flip(-np.arange(0, seq_len, 5)))

    f, ax = plt.subplots(1,1)
    ax.plot(y_simple.T)
    ax.set_title('iSITH')
    ax.set_ylabel('feature val')
    ax.set_xlabel('feature dim')
    sns.despine()



    '''expo weighted moving average'''
    def erwa(data, decay_factor):
        data = data[:,::-1]
        num_features, num_time_steps = data.shape
        weights = np.power(decay_factor, np.arange(num_time_steps))
        weighted_data = data * weights.reshape(1, -1)
        erwa_result = np.sum(weighted_data, axis=1) / np.sum(weights)
        return erwa_result


    # plt.imshow(torch.squeeze(x).numpy().T)
    f, ax = plt.subplots(1,1)
    for decay_factor in [0, .3, .6, 1]:
        y = erwa(torch.squeeze(x).numpy(), decay_factor)
        ax.plot(y, label = f'{decay_factor}')
    ax.set_title('EWMA')
    ax.set_ylabel('feature val')
    ax.set_xlabel('feature dim')
    ax.legend(title='decay rate')
    sns.despine()
