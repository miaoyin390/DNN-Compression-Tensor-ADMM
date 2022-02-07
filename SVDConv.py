# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2022/2/6 22:02

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


class SVDConv2dR(Module):
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
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = list(hp_dict.ranks[name])
        if kernel_size != 1:
            raise ValueError('kernel_size must be 1')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.left_factor = Parameter(torch.Tensor(self.rank, self.in_channels))
        self.right_factor = Parameter(torch.Tensor(self.out_channels, self.rank))

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        if dense_w is not None:
            u, s, v = np.linalg.svd(dense_w.detach().cpu().numpy(), full_matrices=False)
            self.left_factor.data = torch.from_numpy(u)
            self.right_factor.data = torch.from_numpy(s@v)

        else:
            self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.left_factor)
        init.xavier_uniform_(self.right_factor)
        weight = self._recover_weight()
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _recover_weight(self):
        return self.left_factor.mm(self.right_factor)

    def _conv_forward(self, x, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, x: Tensor) -> Tensor:
        return self._conv_forward(x, self._recover_weight())



