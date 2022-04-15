# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2022/4/12 14:31

import torch
import numpy as np
import math

from torch.nn import Parameter, ParameterList, ModuleList
from torch.nn import init
from torch.nn import Hardsigmoid


batch_size = 1
n_seq = 6

tt_params = 0
tt_flops = 0

in_tt_shapes = [8, 20, 20, 18]
in_tt_order = len(in_tt_shapes)
out_tt_shapes = [4, 8, 8]
out_tt_order = len(out_tt_shapes)
tt_ranks = [1, 4, 5, 9, 12, 6, 3, 1]
in_features = int(np.prod(in_tt_shapes))
out_features = int(np.prod(out_tt_shapes))
params = 4 * in_features * out_features
flops = batch_size * in_features * out_features

out_tt_shapes[0] *= 4
tt_shapes = out_tt_shapes + in_tt_shapes
tt_cores = [torch.randn(tt_ranks[i], tt_shapes[i], tt_ranks[i+1]) for i in range(len(tt_shapes))]
h2h_weight = torch.randn(out_features*4, out_features)
bias = torch.randn(out_features*4)

hidden_size = out_features

for i in range(len(tt_cores)):
    tt_params += int(np.prod(tt_cores[i].shape))
print('Compression ratio: {}'.format(params/tt_params))


def lstm_step(x, h_tm1, c_tm1):
    out_shape = list(x.shape)
    out_shape[-1] = 4*out_features
    out = x
    for i in range(in_tt_order - 1, -1, -1):
        a = tt_cores[i + out_tt_order].reshape(
            -1, in_tt_shapes[i] * tt_ranks[i + out_tt_order + 1])
        b = out.reshape(-1, in_tt_shapes[i] * tt_ranks[i + out_tt_order + 1]).t()
        out = a.mm(b).t()

    for i in range(out_tt_order - 1, -1, -1):
        a = tt_cores[i].reshape(-1, tt_ranks[i + 1])
        b = out.reshape(-1, tt_ranks[i + 1]).t()
        out = a.mm(b)
        out = out.reshape(tt_ranks[i], -1).t()

    out = out.reshape(4*out_features, -1).t().reshape(out_shape)
    res = torch.add(out, torch.mm(h_tm1, h2h_weight.t()))

    res = torch.add(res, bias)

    z0 = res[:, :hidden_size]
    z1 = res[:, hidden_size:2 * hidden_size]
    z2 = res[:, 2 * hidden_size:3 * hidden_size]
    z3 = res[:, 3 * hidden_size:]

    i = Hardsigmoid()(z0)
    f = Hardsigmoid()(z1)
    c = f * c_tm1 + i * torch.tanh(z2)
    o = Hardsigmoid()(z3)

    h = o * torch.tanh(c)

    return h, c


x = torch.randn(n_seq, batch_size, in_features)

h = torch.Tensor(torch.zeros(x.shape[1], hidden_size))
c = torch.Tensor(torch.zeros(x.shape[1], hidden_size))
for step in range(n_seq):
    h, c = lstm_step(x[step], h, c)
