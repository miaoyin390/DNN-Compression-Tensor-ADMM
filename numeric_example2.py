# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/15 17:24

# This is a general TT convolution example for any tensor order

import torch
import torch.nn.functional as F

torch.manual_seed(20210915)

batch_size = 12

in_channels = 16
height = 38
width = 38

# original convolutional kernel is of size 32 * 16 * 3 * 3
out_channels = 32
stride = 1
padding = 1
kernel_size = 3

# hyper-parameters settings
order = 2
out_tt_shapes = [8, 4]
in_tt_shapes = [4, 4]
out_tt_ranks = [1, 8, 30]
in_tt_ranks = [14, 4, 1]

# input data
x = torch.randn([batch_size, in_channels, height, width])

# trainable parameters in a single layer
in_tt_cores = [torch.randn([in_tt_ranks[i], in_tt_shapes[i], in_tt_ranks[i + 1]]) for i in range(order)]
core_conv_weight = torch.randn([out_tt_ranks[-1], in_tt_ranks[0], kernel_size, kernel_size])
out_tt_cores = [torch.randn([out_tt_ranks[i], out_tt_shapes[i], out_tt_ranks[i + 1]]) for i in range(order)]

# first computation, where the number of sub-computations equals order
h = x.permute([0, 2, 3, 1])
for i in range(order - 1, -1, -1):
    h = torch.mm(in_tt_cores[i].reshape([in_tt_ranks[i], in_tt_shapes[i] * in_tt_ranks[i + 1]]),
                 h.reshape([-1, in_tt_shapes[i] * in_tt_ranks[i + 1]]).t())
    h = h.t()
h = h.reshape([batch_size, height, width, in_tt_ranks[0]]).permute([0, 3, 1, 2])

# second computation which is a convolution
h = F.conv2d(h, weight=core_conv_weight, stride=stride, padding=padding)
_, _, height_, width_ = h.shape

# third computation, where the number of sub-computations equals order
h = h.permute([0, 2, 3, 1])
for i in range(order - 1, -1, -1):
    h = torch.mm(out_tt_cores[i].reshape(out_tt_ranks[i] * out_tt_shapes[i], out_tt_ranks[i + 1]),
                 h.reshape([-1, out_tt_ranks[i + 1]]).t())
    h = h.reshape([out_tt_ranks[i], -1]).t()

# output data
y = h.reshape([out_channels, batch_size, height_, width_]).permute([1, 0, 2, 3])
print(y.shape)
