
from deepsith import DeepSITH
from torch import nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', palette='colorblind', context='poster')


class DeepSITH_Classifier(nn.Module):
    def __init__(self, out_features, layer_params, dropout=0):
        super(DeepSITH_Classifier, self).__init__()
        last_hidden = layer_params[-1]['hidden_size']
        self.hs = DeepSITH(layer_params=layer_params, dropout=dropout)
        # self.to_out = nn.Linear(last_hidden, out_features)

    def forward(self, inp):
        x = self.hs(inp)
        # x = self.to_out(x)
        return x

in_features = 30
out_features = 30
tau_min = 1
# tau_max should be higher for higher layers
tau_max = 30
# k is the peakiness of the peak
# buff_max ~= 1.5 x tau_max

sith_params1 = {"in_features":in_features,
                "tau_min":tau_min, "tau_max":tau_max, 'buff_max':40,
                "k":84, 'dt':1, "ntau":15, 'g':.0,
                "ttype":torch.FloatTensor, 'batch_norm':False,
                "hidden_size":30, "act_func":nn.ReLU()}
# sith_params2 = {"in_features":sith_params1['hidden_size'],
#                 "tau_min":1, "tau_max":100.0, 'buff_max':175,
#                 "k":40, 'dt':1, "ntau":15, 'g':.0,
#                 "ttype":torch.FloatTensor, 'batch_norm':True,
#                 "hidden_size":35, "act_func":nn.ReLU()}
# layer_params = [sith_params1, sith_params2]
layer_params = [sith_params1]

# g
ds = DeepSITH_Classifier(out_features, layer_params)

# (Batch, 1, features, sequence) and returns (Batch, Taustar, features, sequence)
X = torch.eye(in_features)
seq_len = 20
x = X[:, :seq_len].view((1, 1, in_features, seq_len))
y = ds(x)

f, ax = plt.subplots(1,1, figsize=(7,5))
ax.imshow(torch.squeeze(x).data)
f, ax = plt.subplots(1,1, figsize=(7,5))
ax.imshow(torch.squeeze(y).data)
