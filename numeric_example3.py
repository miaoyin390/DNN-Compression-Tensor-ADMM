# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/10/12 21:19

# general TT FC layer

import torch
import numpy as np
import torch.nn.functional as F

torch.manual_seed(20211012)

in_tt_shapes = [8, 20, 20, 18]
out_tt_shapes = [4, 4, 4, 4]
tt_ranks = [1, 4, 4, 4, 1]

out_tt_shapes[0] *= 4

base_params = np.prod(in_tt_shapes) * np.prod(out_tt_shapes)
base_flops = base_params

tt_params = 0
tt_flops = 0
for k in range(len(in_tt_shapes)):
    tt_flops += np.prod(in_tt_shapes[k:]) * np.prod(out_tt_shapes[:k+1]) * tt_ranks[k] * tt_ranks[k+1]
    tt_params += in_tt_shapes[k] * out_tt_shapes[k] * tt_ranks[k] * tt_ranks[k+1]

print('compression ratio: {}'.format(base_params/tt_params))
print('FLOPs reduction: {}'.format(base_flops/tt_flops))

batch_size = 1

tt_params = 0
tt_flops = 0

in_tt_shapes = [8, 20, 20, 18]
in_tt_order = len(in_tt_shapes)
out_tt_shapes = [4, 8, 8]
out_tt_shapes[0] *= 4
out_tt_order = len(out_tt_shapes)
tt_shapes = out_tt_shapes + in_tt_shapes
tt_ranks = [1, 4, 5, 9, 12, 6, 3, 1]
in_features = int(np.prod(in_tt_shapes))
out_features = int(np.prod(out_tt_shapes))
params = in_features * out_features
flops = batch_size * in_features * out_features

tt_cores = [torch.randn(tt_ranks[i], tt_shapes[i], tt_ranks[i+1]) for i in range(len(tt_shapes))]
for i in range(len(tt_cores)):
    tt_params += int(np.prod(tt_cores[i].shape))
print('Compression ratio: {}'.format(params/tt_params))

x = torch.randn(batch_size, in_features)

out_shape = list(x.shape)
out_shape[-1] = out_features
out = x
for i in range(in_tt_order - 1, -1, -1):
    a = tt_cores[i + out_tt_order].reshape(
        -1, in_tt_shapes[i] * tt_ranks[i + out_tt_order + 1])
    b = out.reshape(-1, in_tt_shapes[i] * tt_ranks[i + out_tt_order + 1]).t()
    out = a.mm(b).t()
    tt_flops += a.shape[0] * a.shape[1] * b.shape[1]


for i in range(out_tt_order - 1, -1, -1):
    a = tt_cores[i].reshape(-1, tt_ranks[i + 1])
    b = out.reshape(-1, tt_ranks[i + 1]).t()
    out = a.mm(b)
    out = out.reshape(tt_ranks[i], -1).t()
    tt_flops += a.shape[0] * a.shape[1] * b.shape[1]

print('Speedup: {}'.format(flops/tt_flops))
out = out.reshape(out_features, -1).t().reshape(out_shape)




