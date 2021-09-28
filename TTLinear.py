# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/23 21:17

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


class TTLinearM(Module):
    def __init__(self, in_features: int, out_features: int, tt_shapes: list,
                 tt_ranks: list, bias: bool = True,
                 from_dense: bool = False, dense_w: Tensor = None, dense_b: Tensor = None):
        super().__init__()

        self.tt_shapes = list(tt_shapes)
        self.tt_order = len(self.tt_shapes)
        channels = 1
        for i in range(len(self.tt_shapes)):
            channels *= self.tt_shapes[i]
            if channels == out_features:
                self.out_tt_order = i + 1
                self.in_tt_order = self.tt_order - self.out_tt_order
                break
        self.out_tt_shapes = self.tt_shapes[:self.out_tt_order]
        self.in_tt_shapes = self.tt_shapes[self.out_tt_order:]

        assert in_features == int(np.prod(self.in_tt_shapes))
        assert out_features == int(np.prod(self.out_tt_shapes))

        self.in_features = in_features
        self.out_features = out_features

        self.tt_ranks = tt_ranks

        self.tt_cores = ParameterList([Parameter(torch.Tensor(
            self.tt_ranks[i], self.tt_shapes[i], self.tt_ranks[i+1])) for i in range(self.tt_order)])

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        if from_dense:
            w = dense_w.detach().cpu().numpy()
            tt_cores = ten2tt(w, self.tt_shapes, self.tt_ranks)

            for i in range(len(tt_cores)):
                self.tt_cores[i].data = torch.from_numpy(tt_cores[i])

            if bias:
                self.bias.data = dense_b

        else:
            self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.tt_order):
            init.xavier_uniform_(self.tt_cores[i])

    def forward(self, x):
        out_shape = x.shape
        out_shape[-1] = self.out_features
        out = x
        for i in range(self.in_tt_order-1, -1, -1):
            out = self.tt_cores[i+self.out_tt_order].reshape(
                -1, self.in_tt_shapes[i]*self.tt_ranks[i+self.out_tt_order+1]).mm(
                out.reshape(-1, self.in_tt_shapes[i]*self.tt_ranks[i+self.out_tt_order+1]).t()).t()

        for i in range(self.out_tt_order-1, -1, -1):
            out = self.tt_cores[i].reshape(-1, self.tt_ranks[i+1]).mm(out.reshape(-1, self.tt_ranks[i+1]).t())
            out = out.reshape(self.tt_ranks[i], -1).t()

        out = out.reshape(self.out_features, -1).t().reshape(out_shape)

        if self.bias is not None:
            out += self.bias

        return out


class TTLinearR(Module):
    def __init__(self, in_features: int, out_features: int, tt_shapes: list,
                 tt_ranks: list, bias: bool = True,
                 from_dense: bool = False, dense_w: Tensor = None, dense_b: Tensor = None):
        super().__init__()

        self.tt_shapes = list(tt_shapes)
        self.tt_order = len(self.tt_shapes)
        channels = 1
        for i in range(len(self.tt_shapes)):
            channels *= self.tt_shapes[i]
            if channels == out_features:
                self.out_tt_order = i + 1
                self.in_tt_order = self.tt_order - self.out_tt_order
                break
        self.out_tt_shapes = self.tt_shapes[:self.out_tt_order]
        self.in_tt_shapes = self.tt_shapes[self.out_tt_order:]

        assert in_features == int(np.prod(self.in_tt_shapes))
        assert out_features == int(np.prod(self.out_tt_shapes))

        self.in_features = in_features
        self.out_features = out_features

        self.tt_ranks = tt_ranks

        self.tt_cores = ParameterList([Parameter(torch.Tensor(
            self.tt_ranks[i], self.tt_shapes[i], self.tt_ranks[i+1])) for i in range(self.tt_order)])

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        if from_dense:
            w = dense_w.detach().cpu().numpy()
            tt_cores = ten2tt(w, self.tt_shapes, self.tt_ranks)

            for i in range(len(tt_cores)):
                self.tt_cores[i].data = torch.from_numpy(tt_cores[i])

            if bias:
                self.bias.data = dense_b

        else:
            self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.tt_order):
            init.xavier_uniform_(self.tt_cores[i])
        weight = self._recover_weight()
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _recover_weight(self):
        w = self.tt_cores[0]
        for i in range(1, self.tt_order):
            w = w.reshape(-1, self.tt_ranks[i]).mm(self.tt_cores[i].reshape(self.tt_ranks[i], -1))
        w = w.reshape(self.out_features, self.in_features)

        return w

    def forward(self, x):
        return F.linear(x, self._recover_weight(), self.bias)
