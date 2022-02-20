# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2022/2/19 23:42


import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import math
import torch.nn as nn

from torch import Tensor
from torch.nn import Parameter, ParameterList
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple
from torch.nn import Conv2d
import geoopt

tl.set_backend('pytorch')


class StfTKConv2dC(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 hp_dict: Optional = None,
                 name: str = None,
                 dense_w: Tensor = None,
                 dense_b: Tensor = None,
                 ):
        if groups != 1:
            raise ValueError("groups must be 1 in this mode")
        if padding_mode != 'zeros':
            raise ValueError("padding_mode must be zero in this mode")
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(StfTKConv2dC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ranks = hp_dict.ranks[name]
        self.in_rank = self.ranks[1]
        self.out_rank = self.ranks[0]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.padding_mode = padding_mode

        self.first_kernel = geoopt.ManifoldParameter(
            torch.Tensor(self.in_channels, self.in_rank), manifold=geoopt.Stiefel())
        self.core_kernel = Parameter(Tensor(self.out_rank, self.in_rank, *kernel_size))
        self.last_kernel = geoopt.ManifoldParameter(
            torch.Tensor(self.out_channels, self.out_rank), manifold=geoopt.Stiefel())

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        if dense_w is not None:
            core_tensor, [last_factor, first_factor] = partial_tucker(dense_w, modes=[0, 1],
                                                                      rank=self.ranks, init='svd')
            self.first_kernel.data = first_factor.data
            self.last_kernel.data = last_factor.data
            self.core_kernel.data = core_tensor.data

        else:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.first_kernel)
        init.xavier_uniform_(self.core_kernel)
        init.xavier_uniform_(self.last_kernel)

    def forward(self, x):
        out = F.conv2d(x, self.first_kernel.t().unsqueeze(-1).unsqueeze(-1))
        out = F.conv2d(out, self.core_kernel, None, self.stride,
                       self.padding, self.dilation, self.groups)
        out = F.conv2d(out, self.last_kernel.unsqueeze(-1).unsqueeze(-1), self.bias)
        return out



