"""
An implementation of Context dependent GRU.
the gru uses the implementation from:
https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
"""
import math
import torch
import torch.nn as nn
import numpy as np
from utils import to_pth, to_np

class CGRU_v2(nn.Module):

    def __init__(
            self, input_dim, hidden_dim, output_dim, context_dim, ctx_wt=0,
            softmax_beta=None, try_reset_h=False,
            bias=True, sigmoid_output=False, dropout_rate=0, zero_init_state=True,
        ):
        super(CGRU_v2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.bias = bias
        # weights
        self.i2h = nn.Linear(input_dim+context_dim, 3 * hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim+context_dim, 3 * hidden_dim, bias=bias)
        self.h2o = nn.Linear(hidden_dim, output_dim, bias=bias)
        # set dropout rate
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.i2h_dropout = nn.Dropout(dropout_rate)
            self.ci2h_dropout = nn.Dropout(dropout_rate)
            self.h2o_dropout = nn.Dropout(dropout_rate)
        # miscs
        self.sigmoid_output = sigmoid_output
        self.zero_init_state = zero_init_state
        # optimization crit
        self.criterion = nn.MSELoss()
        self.softmax_beta = softmax_beta
        # compute
        self.try_reset_h = try_reset_h
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        # std = 1.0 / self.hidden_dim
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, x, hidden, context_t=None):
        hidden = hidden.view(-1)
        # combine contextual input and x / h_prev
        xc = torch.cat([x, context_t])
        hc = torch.cat([hidden, context_t])
        gate_x = self.i2h(xc)
        gate_h = self.h2h(hc)

        # compute the gates
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 0)
        h_r, h_i, h_n = gate_h.chunk(3, 0)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        # compute h
        h_t = newgate + inputgate * (hidden - newgate)
        if self.dropout_rate > 0:
            h_t = self.h2o_dropout(h_t)
        yhat_t = self.h2o(h_t)
        if self.sigmoid_output:
            yhat_t = yhat_t.sigmoid()

        output = [yhat_t, h_t]
        cache = [resetgate, inputgate, newgate]
        return output, cache


    def forward_nograd(self, x_t, h, context_t=None):
        with torch.no_grad():
            [yhat_t, h_t], _ = self.forward(x_t, h, context_t=context_t)
        return yhat_t

    def try_all_contexts(self, y_t, x_t, h_t, contexts, prev_context_id=None, verbose=True):
        # loop over all ctx ...
        n_contexts = len(contexts)
        pe = torch.zeros(n_contexts, )
        [None] * (n_contexts)
        for k in range(n_contexts):
            yhat_k = self.forward_nograd(x_t, h_t, to_pth(contexts[k]))
            pe[k] = self.criterion(y_t, torch.squeeze(yhat_k))

        # decide if whether to restart the ongoing context
        if prev_context_id is not None and self.try_reset_h:
            yhat_prev_restart = self.forward_nograd(
                x_t, self.get_init_states(), to_pth(contexts[prev_context_id])
            )
            pe_prev_restart = self.criterion(y_t, torch.squeeze(yhat_prev_restart))
            # append the restarting PE at the end of the list 
            pe = torch.cat([pe, pe_prev_restart.view(1)])

        # softmax the pe
        if self.softmax_beta is not None:
            pe = stable_softmax(to_np(pe), self.softmax_beta)
        else:
            pe = to_np(pe)
        return pe


    def get_kaiming_states(self):
        return nn.init.kaiming_uniform_(torch.empty(1, 1, self.hidden_dim))

    def get_init_states(self):
        if self.zero_init_state:
            return self.get_zero_states()
        return self.get_rand_states()

    @torch.no_grad()
    def get_zero_states(self):
        h_0 = torch.zeros(1, 1, self.hidden_dim)
        return h_0

    @torch.no_grad()
    def get_rand_states(self, scale=.1):
        h_0 = torch.randn(1, 1, self.hidden_dim) * scale
        return h_0

def freeze_layer(layer_to_freeze, model, verbose=False):
    # layer_to_freeze = 'ci2h'
    layer_found = False
    for name, param in model.named_parameters():
        if layer_to_freeze in name:
            layer_found = True
            if param.requires_grad:
                param.requires_grad = False
            else:
                print(f'layer already freezed: {name}')
                if verbose:
                    print(f'freeze layer: {name}')
    if not layer_found:
        raise ValueError(f'layer {name} not found')
    return model

def get_weights(layer_name, model, to_np=True):
    weights = {}
    for name, param in model.named_parameters():
        if layer_name in name:
            w_ = param.data.numpy() if to_np else param.data
            weights[name] = w_
    return weights



def stable_softmax(x, beta=1/3, subtract_max=True):
    assert beta > 0
    if subtract_max:
        x -= max(x)
    # apply temperture
    z = x / beta
    return np.exp(z) / (np.sum(np.exp(z)) + 1e-010)
