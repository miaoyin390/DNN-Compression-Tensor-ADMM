# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/10/12 21:19

import torch
import numpy as np
import torch.nn.functional as F

torch.manual_seed(20211012)

batch_size = 64
in_features = 192
out_features = 768

in_tt_shapes = [12, 16]
in_tt_order = len(in_tt_shapes)
out_tt_shapes = [32, 24]
out_tt_order = len(out_tt_shapes)
tt_shapes = out_tt_shapes + in_tt_shapes
tt_ranks = [1, 28, 128, 14, 1]

tt_cores = [torch.randn(tt_ranks[i], tt_shapes[i], tt_ranks[i+1]) for i in range(len(tt_shapes))]

x = torch.randn(batch_size, in_features)

out_shape = list(x.shape)
out_shape[-1] = out_features
out = x
for i in range(in_tt_order - 1, -1, -1):
    out = tt_cores[i + out_tt_order].reshape(
        -1, in_tt_shapes[i] * tt_ranks[i + out_tt_order + 1]).mm(
        out.reshape(-1, in_tt_shapes[i] * tt_ranks[i + out_tt_order + 1]).t()).t()

for i in range(out_tt_order - 1, -1, -1):
    out = tt_cores[i].reshape(-1, tt_ranks[i + 1]).mm(out.reshape(-1, tt_ranks[i + 1]).t())
    out = out.reshape(tt_ranks[i], -1).t()

out = out.reshape(out_features, -1).t().reshape(out_shape)
print(out.shape)




