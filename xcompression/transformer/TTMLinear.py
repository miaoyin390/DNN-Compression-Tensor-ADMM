# -*- coding:utf-8 -*-
# 
# Author: JINQI XIAO
# Time: Mar/01/2022

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn import Parameter
from torch.nn import ParameterList
from torch.nn import init


class TTMLinear(Module):
    __constants__ = ['input_tt_shape', 'output_tt_shape', 'tt_ranks']

    def __init__(self, input_tt_shape, output_tt_shape, tt_ranks,
                 bias=True):
        super(TTMLinear, self).__init__()
        self.input_tt_shape = input_tt_shape
        self.output_tt_shape = output_tt_shape
        self.n_dim = len(input_tt_shape)
        self.output_size = int(np.prod(self.output_tt_shape))
        self.input_size = int(np.prod(self.input_tt_shape))

        self.intermediate_shape = []
        for i in range(self.n_dim):
            self.intermediate_shape.append(self.input_tt_shape[i])
            self.intermediate_shape.append(self.output_tt_shape[i])
        self.transpose = []
        for i in range(0, 2 * self.n_dim, 2):
            self.transpose.append(i)
        for i in range(1, 2 * self.n_dim, 2):
            self.transpose.append(i)

        self.tt_ranks = tt_ranks
        self.cores = ParameterList([Parameter(
            torch.Tensor(self.tt_ranks[i], self.input_tt_shape[i], self.output_tt_shape[i], self.tt_ranks[i + 1])) for i
            in range(self.n_dim)])
        if bias:
            self.bias = Parameter(torch.randn([self.output_size]))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def get_tt_ranks(self):
        str_tt_ranks = [str(r) for r in self.tt_ranks]
        return ', '.join(str_tt_ranks)

    def get_core_size(self):
        size = 0
        for i in range(self.n_dim):
            size += self.tt_ranks[i] * self.input_tt_shape[i] * self.output_tt_shape[i] * self.tt_ranks[i + 1]
        return size

    def reset_parameters(self):
        for i in range(self.n_dim):
            init.xavier_uniform_(self.cores[i])

    def forward(self, input):
        res = self.cores[0]
        for i in range(1, self.n_dim):
            res = torch.reshape(res, (-1, self.tt_ranks[i]))
            curr_core = torch.reshape(self.cores[i], (self.tt_ranks[i], -1))
            res = torch.matmul(res, curr_core)

        res = torch.reshape(res, self.intermediate_shape)
        res = torch.Tensor.permute(res, self.transpose)
        weight = torch.reshape(res, [self.input_size, self.output_size]).to(input.device)
        return F.linear(input, weight.T, self.bias)

    def extra_repr(self):
        return 'input_tt_shape={}, output_tt_shape={}, tt_ranks={}, bias={}'.format(
            self.input_tt_shape, self.output_tt_shape, self.tt_ranks, self.bias is not None
        )
