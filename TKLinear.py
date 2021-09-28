# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/16 23:18

import numpy as np
import torch
import math
import torch.nn as nn
import tensorly as tl

from torch.nn import Parameter, ParameterList
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor
from torch.nn.modules import Module
from tensorly.decomposition import parafac, partial_tucker
from typing import List, Tuple

tl.set_backend('pytorch')


class TKLinearM(Module):
    def __init__(self, in_features: int, out_features: int,
                 ranks: list, bias: bool = True,
                 from_dense: bool = False, dense_w: Tensor = None, dense_b: Tensor = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        (out_rank, in_rank) = ranks
        self.in_rank = ranks[1]
        self.out_rank = ranks[0]

        self.first_factor = Parameter(torch.Tensor(in_rank, in_features))
        self.core_tensor = Parameter(torch.Tensor(out_rank, in_rank))
        self.last_factor = Parameter(torch.Tensor(out_features, out_rank))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        if from_dense:
            core_tensor, (last_factor, first_factor) = partial_tucker(dense_w, modes=[0, 1],
                                                                      rank=[out_rank, in_rank], init='svd')
            self.first_factor.data = torch.transpose(first_factor, 1, 0)
            self.last_factor.data = last_factor
            self.core_tensor = core_tensor

            if bias:
                self.bias.data = dense_b
        else:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.first_factor, a=math.sqrt(5))
        init.kaiming_uniform_(self.core_tensor, a=math.sqrt(5))
        init.kaiming_uniform_(self.last_factor, a=math.sqrt(5))
        weight = tl.tucker_to_tensor((self.core_tensor, (self.last_factor, self.first_factor.t())))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.first_factor)
        out = F.linear(out, self.core_tensor)
        out = F.linear(out, self.last_factor, self.bias)

        return out


class TKLinearR(Module):
    def __init__(self, in_features: int, out_features: int,
                 ranks: list, bias: bool = True,
                 from_dense: bool = False, dense_w: Tensor = None, dense_b: Tensor = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        (out_rank, in_rank) = ranks
        self.in_rank = ranks[1]
        self.out_rank = ranks[0]

        self.first_factor = Parameter(torch.Tensor(in_rank, in_features))
        self.core_tensor = Parameter(torch.Tensor(out_rank, in_rank))
        self.last_factor = Parameter(torch.Tensor(out_features, out_rank))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        if from_dense:
            core_tensor, (last_factor, first_factor) = partial_tucker(dense_w, modes=[0, 1],
                                                                      rank=[out_rank, in_rank], init='svd')
            self.first_factor.data = torch.transpose(first_factor, 1, 0)
            self.last_factor.data = last_factor
            self.core_tensor = core_tensor

            if bias:
                self.bias.data = dense_b
        else:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.first_factor, a=math.sqrt(5))
        init.kaiming_uniform_(self.core_tensor, a=math.sqrt(5))
        init.kaiming_uniform_(self.last_factor, a=math.sqrt(5))
        weight = tl.tucker_to_tensor((self.core_tensor, (self.last_factor, self.first_factor.t())))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _recover_weight(self):
        w = tl.tucker_to_tensor((self.core_tensor, (self.last_factor, self.first_factor.t())))
        return w

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self._recover_weight(), self.bias)
