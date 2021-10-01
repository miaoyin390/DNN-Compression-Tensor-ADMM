# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/15 14:08

# This is a specific example for general TT convolution when the order is set as 1

import numpy as np

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

# hyper-parameters
rank1 = 10
rank2 = 24

# In this case, there are only three computation in total

# input data
x = torch.randn([batch_size, in_channels, height, width])

# trainable parameters in a single layer
in_tt_core = torch.randn([rank1, in_channels, 1])
core_conv_weight = torch.randn([rank2, rank1, kernel_size, kernel_size])
out_tt_core = torch.randn([1, out_channels, rank2])

# the first computation
h = torch.mm(in_tt_core.reshape([rank1, in_channels]), x.permute([0, 2, 3, 1]).reshape([-1, in_channels]).t())
h = h.reshape([rank1, batch_size, height, width]).permute([1, 0, 2, 3])
# the second computation
h = F.conv2d(h, weight=core_conv_weight, stride=stride, padding=padding)
(_, _, height_, width_) = h.shape
# the third computation
h = torch.mm(out_tt_core.reshape([out_channels, rank2]), h.permute([1, 0, 2, 3]).reshape([rank2, -1]))

# output data
y = h.reshape([out_channels, batch_size, height_, width_]).permute([1, 0, 2, 3])
print(y.shape)








