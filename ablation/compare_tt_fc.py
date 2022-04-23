# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2022/4/11 22:31

import numpy as np
import torch

# VGG-FC6
# in_tt_shapes = [4, 4, 4, 4, 4, 4]
# out_tt_shapes = [2, 7, 8, 8, 7, 4]
# tt_ranks = [1, 3, 3, 4, 4, 4, 1]

# VGG-FC7
# in_tt_shapes = [4, 4, 4, 4, 4, 4]
# out_tt_shapes = [4, 4, 4, 4, 4, 4]
# tt_ranks = [1, 4, 4, 4, 4, 4, 1]

# in_tt_shapes = [1, 1, 5, 7, 9]
# out_tt_shapes = [2, 2, 4, 4, 8]
# tt_ranks = [1, 2, 2, 2, 2, 1]

# Embedding
# in_tt_shapes = [56, 56, 56, 58]
# out_tt_shapes = [2, 2, 2, 2]
# tt_ranks = [1,3,4,3,1]

# TIMIT projection
in_tt_shapes = [2, 4, 4, 4, 8]
out_tt_shapes = [2, 4, 4, 4, 4]
tt_ranks = [1, 1, 1, 1, 1, 1]

base_params = 0
base_flops = 0

base_params += np.prod(in_tt_shapes) * np.prod(out_tt_shapes)
base_flops += base_params

tt_params = 0
tt_flops = 0
for k in range(len(in_tt_shapes)):
    tt_flops += np.prod(in_tt_shapes[k:]) * np.prod(out_tt_shapes[:k + 1]) * tt_ranks[k] * tt_ranks[k + 1]
    tt_params += in_tt_shapes[k] * out_tt_shapes[k] * tt_ranks[k] * tt_ranks[k + 1]

print('# tt_params: {}'.format(tt_params))
print('# tt_flops: {}'.format(tt_flops))
print('compression ratio: {}'.format(base_params / tt_params))
print('FLOPs reduction: {}'.format(base_flops / tt_flops))

batch_size = 1

tt_params = 0
tt_flops = 0

# VGG-FC6
# in_tt_shapes = [16, 14, 8, 14]
# out_tt_shapes = [16, 16, 16]
# tt_ranks = [1, 4, 4, 8, 4, 4, 4, 1]

# VGG-FC7
# in_tt_shapes = [16, 16, 16]
# out_tt_shapes = [16, 16, 16]
# tt_ranks = [1, 4, 4, 8, 4, 2, 1]

# Embedding
# in_tt_shapes = [25, 25, 25, 25, 26]
# out_tt_shapes = [16]
# tt_ranks = [1, 4, 6, 8, 4, 4, 1]

# TIMIT projection
in_tt_shapes = [8, 8, 16]
out_tt_shapes = [8, 8, 8]
tt_ranks = [1, 1, 1, 2, 1, 1, 1]

in_tt_order = len(in_tt_shapes)
out_tt_order = len(out_tt_shapes)
tt_shapes = out_tt_shapes + in_tt_shapes
in_features = int(np.prod(in_tt_shapes))
out_features = int(np.prod(out_tt_shapes))
params = in_features * out_features
flops = batch_size * in_features * out_features

tt_cores = [torch.randn(tt_ranks[i], tt_shapes[i], tt_ranks[i + 1]) for i in range(len(tt_shapes))]
for i in range(len(tt_cores)):
    tt_params += int(np.prod(tt_cores[i].shape))
print('# tt_params: {}'.format(tt_params))
print('Compression ratio: {}'.format(params / tt_params))

x = torch.randn(batch_size, in_features)

out_shape = list(x.shape)
out_shape[-1] = out_features
out = x
for i in range(in_tt_order - 1, -1, -1):
    a = tt_cores[i + out_tt_order].reshape(
        -1, in_tt_shapes[i] * tt_ranks[i + out_tt_order + 1])
    b = out.reshape(-1, in_tt_shapes[i] * tt_ranks[i + out_tt_order + 1]).t()
    print(b.shape[0])
    out = a.mm(b).t()
    tt_flops += a.shape[0] * a.shape[1] * b.shape[1]

for i in range(out_tt_order - 1, -1, -1):
    a = tt_cores[i].reshape(-1, tt_ranks[i + 1])
    b = out.reshape(-1, tt_ranks[i + 1]).t()
    out = a.mm(b)
    print(b.shape[0])
    out = out.reshape(tt_ranks[i], -1).t()
    tt_flops += a.shape[0] * a.shape[1] * b.shape[1]

print('# tt_flops: {}'.format(tt_flops))
print('Speedup: {}'.format(flops / tt_flops))
out = out.reshape(out_features, -1).t().reshape(out_shape)
