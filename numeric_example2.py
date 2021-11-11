# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/15 17:24

# This is a general TT convolution example for any tensor order

import torch
import torch.nn.functional as F
import numpy as np

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

params = in_channels * out_channels * kernel_size * kernel_size
flops = batch_size * in_channels * out_channels * kernel_size * kernel_size * height * width
tt_params = 0
tt_flops = 0

# hyper-parameters settings
order = 2
out_tt_shapes = [8, 4]
in_tt_shapes = [4, 4]
out_tt_ranks = [1, 8, 16]
in_tt_ranks = [10, 4, 1]

# input data
x = torch.randn([batch_size, in_channels, height, width])

# trainable parameters in a single layer
in_tt_cores = [torch.randn([in_tt_ranks[i], in_tt_shapes[i], in_tt_ranks[i + 1]]) for i in range(order)]
for i in range(len(in_tt_cores)):
    tt_params += int(np.prod(in_tt_cores[i].shape))
core_conv_weight = torch.randn([out_tt_ranks[-1], in_tt_ranks[0], kernel_size, kernel_size])
tt_params += int(np.prod(core_conv_weight.shape))
out_tt_cores = [torch.randn([out_tt_ranks[i], out_tt_shapes[i], out_tt_ranks[i + 1]]) for i in range(order)]
for i in range(len(out_tt_cores)):
    tt_params += int(np.prod(out_tt_cores[i].shape))

print('Compression ratio: {}'.format(params/tt_params))

# first computation, where the number of sub-computations equals order
h = x.permute([0, 2, 3, 1])
for i in range(order - 1, -1, -1):
    a = in_tt_cores[i].reshape([in_tt_ranks[i], in_tt_shapes[i] * in_tt_ranks[i + 1]])
    b = h.reshape([-1, in_tt_shapes[i] * in_tt_ranks[i + 1]]).t()
    h = a.mm(b).t()
    tt_flops += a.shape[0] * a.shape[1] * b.shape[1]
h = h.reshape([batch_size, height, width, in_tt_ranks[0]]).permute([0, 3, 1, 2])

# second computation which is a convolution
h = F.conv2d(h, weight=core_conv_weight, stride=stride, padding=padding)
_, _, height_, width_ = h.shape
tt_flops += batch_size * height_ * width_ * kernel_size * kernel_size * in_tt_ranks[0] * out_tt_ranks[-1]

# third computation, where the number of sub-computations equals order
h = h.permute([0, 2, 3, 1])
for i in range(order - 1, -1, -1):
    a = out_tt_cores[i].reshape(out_tt_ranks[i] * out_tt_shapes[i], out_tt_ranks[i + 1])
    b = h.reshape([-1, out_tt_ranks[i + 1]]).t()
    h = a.mm(b)
    h = h.reshape([out_tt_ranks[i], -1]).t()
    tt_flops += a.shape[0] * a.shape[1] * b.shape[1]

# output data
y = h.reshape([out_channels, batch_size, height_, width_]).permute([1, 0, 2, 3])
# print(y.shape)
print('Speedup: {}'.format(flops/tt_flops))

print(y.size()[1:].numel())
# print(torch.prod(y.shape[1:]))
