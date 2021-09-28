# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/16 23:19

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

from ttd import ten2tt


class TTConv2dM(Module):
    def __init__(self, in_channels, out_channels, tt_shapes, tt_ranks,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 from_dense=False, dense_w=None, dense_b=None):
        # kernel_size = _pair(kernel_size)
        # stride = _pair(stride)
        # padding = _pair(padding)
        # dilation = _pair(dilation)
        super().__init__()

        self.tt_shapes = list(tt_shapes)
        self.tt_order = len(self.tt_shapes)
        channels = 1
        for i in range(len(self.tt_shapes)):
            channels *= self.tt_shapes[i]
            if channels == out_channels:
                self.out_tt_order = i + 1
                self.in_tt_order = self.tt_order - self.out_tt_order - 1
                break
        self.out_tt_shapes = self.tt_shapes[:self.out_tt_order]
        self.in_tt_shapes = self.tt_shapes[self.out_tt_order+1:]

        assert in_channels == int(np.prod(self.in_tt_shapes))
        assert out_channels == int(np.prod(self.out_tt_shapes))

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.tt_ranks = list(tt_ranks)
        self.out_tt_ranks = self.tt_ranks[:self.out_tt_order+1]
        self.in_tt_ranks = self.tt_ranks[self.out_tt_order+1:]

        self.in_tt_cores = ParameterList([Parameter(torch.Tensor(
                    self.in_tt_ranks[i], self.in_tt_shapes[i], self.in_tt_ranks[i+1]))
                    for i in range(self.in_tt_order)])

        self.core_conv = torch.nn.Conv2d(in_channels=self.in_tt_ranks[0],
                                         out_channels=self.out_tt_ranks[-1], kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=False, padding_mode=padding_mode)

        self.out_tt_cores = ParameterList([Parameter(torch.Tensor(
            self.out_tt_ranks[i], self.out_tt_shape[i], self.out_tt_ranks[i+1]))
            for i in range(self.out_tt_order)])

        if bias:
            self.bias = Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter('bias', None)

        if from_dense:
            w = dense_w.detach().cpu().numpy()
            w = np.reshape(w, [self.out_channels, self.in_channels, -1])
            tt_shapes = self.out_tt_shapes + [w.shape[-1]] + self.in_tt_shapes
            tt_cores = ten2tt(w, tt_shapes, self.tt_ranks)

            for i in range(len(tt_cores)):
                if i < self.out_tt_order:
                    self.out_tt_cores[i].data = torch.from_numpy(tt_cores[i])
                elif i == self.out_tt_order:
                    self.core_conv.weight.data = torch.from_numpy(tt_cores[i]).permute(0, 2, 1).reshape(
                        self.out_tt_ranks[-1], self.in_tt_ranks[0], w.shape[2], w.shape[3])
                else:
                    self.in_tt_cores[i-self.out_tt_order-1].data = torch.from_numpy(tt_cores[i])

            if bias:
                self.bias.data = dense_b

        else:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.out_tt_order):
            init.xavier_uniform_(self.out_tt_cores[i])
        for i in range(self.in_tt_order):
            init.xavier_uniform_(self.in_tt_cores[i])
        self.core_conv.reset_parameters()
        # init.kaiming_uniform_(self.cores[i], a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)

    def get_ranks(self):
        str_ranks = [str(r) for r in self.tt_ranks]
        return ', '.join(str_ranks)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out = x.permute(0, 2, 3, 1)
        for i in range(self.in_tt_order-1, -1, -1):
            out = torch.mm(self.in_tt_cores[i].reshape(self.in_tt_ranks[i], self.in_tt_shapes[i]*self.in_tt_ranks[i+1]),
                           out.reshape(-1, self.in_tt_shapes[i] * self.in_tt_ranks[i+1]).t()).t()
        out = out.reshape(batch_size, height, width, self.in_tt_ranks[0]).permute(0, 3, 1, 2)

        out = self.core_conv(out)
        _, _, height_, width_ = out.shape

        out = out.permute(0, 2, 3, 1)
        for i in range(self.out_tt_order-1, -1, -1):
            out = torch.mm(self.out_tt_cores[i].reshape(self.out_tt_ranks[i]*self.out_tt_shapes[i], self.out_tt_ranks[i+1]),
                           out.reshape(-1, self.out_tt_ranks[i+1]).t())
            out = out.reshape(self.out_tt_ranks[i], -1).t()

        out = out.reshape(self.out_channels, batch_size, height_, width_).permute(1, 0, 2, 3)
        if self.bias is not None:
            out += self.bias

        return out


class TTConv2dR(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 tt_shapes: list,
                 tt_ranks: list,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 from_dense: bool = False,
                 dense_w: Tensor = None,
                 dense_b: Tensor = None,
                 ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(TTConv2dR, self).__init__()

        self.tt_shapes = list(tt_shapes)
        self.tt_order = len(self.tt_shapes)
        channels = 1
        for i in range(len(self.tt_shapes)):
            channels *= self.tt_shapes[i]
            if channels == out_channels:
                self.out_tt_order = i + 1
                self.in_tt_order = self.tt_order - self.out_tt_order - 1
                break
        self.out_tt_shapes = self.tt_shapes[:self.out_tt_order]
        self.in_tt_shapes = self.tt_shapes[self.out_tt_order+1:]

        # output channels are in front of input channels in the original kernel
        self.tt_ranks = list(tt_ranks)
        self.out_tt_ranks = self.tt_ranks[:self.out_tt_order+1]
        self.in_tt_ranks = self.tt_ranks[self.out_tt_order+1:]

        assert in_channels == int(np.prod(self.in_tt_shapes))
        assert out_channels == int(np.prod(self.out_tt_shapes))

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
        self.kernel_shape = [out_channels, in_channels // groups, *kernel_size]

        self.filter_dim = int(self.kernel_shape[2] * self.kernel_shape[3])

        self.out_tt_cores = ParameterList([Parameter(torch.Tensor(
            self.out_tt_ranks[i], self.out_tt_shape[i], self.out_tt_ranks[i+1]))
            for i in range(self.out_tt_order)])

        self.conv_core = Parameter(torch.Tensor(self.out_tt_ranks[-1], self.filter_dim, self.in_tt_ranks[0]))

        self.in_tt_cores = ParameterList([Parameter(torch.Tensor(
                    self.in_tt_ranks[i], self.in_tt_shapes[i], self.in_tt_ranks[i+1]))
                    for i in range(self.in_tt_order)])

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        if from_dense:
            w = dense_w.detach().cpu().numpy()
            w = np.reshape(w, [self.out_channels, self.in_channels, -1])
            tt_shapes = self.out_tt_shapes + [w.shape[-1]] + self.in_tt_shapes
            tt_cores = ten2tt(w, tt_shapes, self.tt_ranks)

            for i in range(len(tt_cores)):
                if i < self.out_tt_order:
                    self.out_tt_cores[i].data = torch.from_numpy(tt_cores[i])
                elif i == self.out_tt_order:
                    self.core_conv.weight.data = torch.from_numpy(tt_cores[i])
                else:
                    self.in_tt_cores[i-self.out_tt_order-1].data = torch.from_numpy(tt_cores[i])

            if bias:
                self.bias.data = dense_b

        else:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.out_tt_order):
            init.xavier_uniform_(self.out_tt_cores[i])
        init.xavier_uniform_(self.conv_core)
        for i in range(self.in_tt_order):
            init.xavier_uniform_(self.in_tt_cores[i])
        weight = self._recover_weight()
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _recover_weight(self):
        w = self.out_tt_cores[0]
        for i in range(1, self.out_tt_order):
            w = torch.mm(w.reshape(-1, self.out_tt_ranks[i]), self.out_tt_cores[i].reshape(self.out_tt_ranks[i], -1))
        w = torch.mm(w.reshape(-1, self.out_tt_ranks[-1]), self.conv_core.reshape(self.out_tt_ranks[-1], -1))
        for i in range(0, self.in_tt_order):
            w = torch.mm(w.reshape(-1, self.in_tt_ranks[i]), self.in_tt_cores[i].reshape(self.in_tt_ranks[i], -1))

        w = w.reshape(self.out_channels, self.filter_dim, self.in_channels).permute(0, 1, 2).reshape(self.kernel_shape)
        return w

    def _conv_forward(self, x, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, x: Tensor) -> Tensor:
        return self._conv_forward(x, self._recover_weight())
