# -*- coding:utf-8 -*-
#
# Author: MIAO YIN
# Time: 2021/9/23 21:17
import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter, ParameterList
from torch.nn import init
from torch.nn.modules import Module
from typing import Optional
from numpy.linalg import svd


def get_factors(n):
    factors = []
    for k in range(2, n + 1):
        while n != k:
            if n % k == 0:
                factors.append(k)
                # print(k, end="*")
                n = n / k
            else:
                break
    factors.append(n)
    return factors


def split_to_factors(feature_size, dim):
    # print(feature_size, dim)
    factors = get_factors(feature_size)
    dim_factors = [1] * dim
    avg = int(np.power(feature_size, 1.0 / dim))
    end = len(factors) - 1
    start = 0
    i = 0
    while end >= start:
        if factors[end] >= avg:
            dim_factors[i] = factors[end]
            end -= 1
            i += 1
            if i >= dim:
                break
            avg = int(np.power(feature_size / np.prod(dim_factors), 1.0 / (dim - i)))
        else:
            dim_factors[i] = factors[end] * factors[start]
            while dim_factors[i] < avg and end > start:
                start += 1
                t = factors[start] * dim_factors[i]
                if (t - avg) > (avg - dim_factors[i]):
                    start -= 1
                    break
                else:
                    dim_factors[i] = t
            end -= 1
            i += 1
            if i >= dim:
                break
            avg = int(np.power(feature_size / np.prod(dim_factors), 1.0 / (dim - i)))
    dim_factors = list(map(int, np.sort(dim_factors)[::-1]))
    return list(dim_factors)


def ten2tt(x, tt_shapes, tt_ranks):
    # tt_ranks_ = tt_ranks.copy()
    d = len(tt_shapes)
    t = x
    tt_cores = []
    for i in range(d - 1):
        # print(t.shape)
        t = np.reshape(t, [tt_ranks[i] * tt_shapes[i], -1])
        # print(t.shape)
        u, s, v = svd(t, full_matrices=False)
        if s.shape[0] < tt_ranks[i + 1]:
            tt_ranks[i + 1] = s.shape[0]

        u = u[:, :tt_ranks[i + 1]]
        s = s[:tt_ranks[i + 1]]
        v = v[:tt_ranks[i + 1], :]

        tt_cores.append(np.reshape(u, [tt_ranks[i], tt_shapes[i], tt_ranks[i + 1]]))
        t = np.dot(np.diag(s), v)
    # print(t.shape)
    t = np.reshape(t, [tt_ranks[d - 1], tt_shapes[d - 1], tt_ranks[d]])
    # print(t.shape)
    tt_cores.append(t)
    # print(tt_ranks)

    return tt_cores


def tt2ten(tt_cores, tt_shapes):
    d = len(tt_cores)
    t = tt_cores[0]
    for i in range(1, d):
        rank = tt_cores[i].shape[0]
        t = np.reshape(t, [-1, rank])
        t = np.dot(t, np.reshape(tt_cores[i], [rank, -1]))

    t = np.reshape(t, tt_shapes)
    return t


def compute_ranks_tt(tt_shapes, ratio):
    param = np.prod(tt_shapes)
    d = len(tt_shapes)
    c = param / ratio
    if d == 2:
        r = int(param / (ratio * sum(tt_shapes)))
        # print(r)
        return [1, r, 1]
    b = tt_shapes[0] + tt_shapes[-1]
    a = sum(tt_shapes[1:-1])
    r_avg = int((math.sqrt(b * b + 4 * a * c) - b) / (2 * a))
    cur_params = a * r_avg * r_avg + b * r_avg
    tt_ranks = [r_avg] * (d - 1)

    # for i in range(d-1):
    #     if i == 0:
    #         t = tt_shapes[i] + tt_shapes[i+1] * tt_ranks[i+1]
    #     elif i == d-2:
    #         t = tt_shapes[-1] + tt_shapes[i] * tt_ranks[i-1]
    #     else:
    #         t =  tt_shapes[i] * tt_ranks[i-1] + tt_shapes[i+1] * tt_ranks[i+1]
    #     if (cur_params + t) <= c:
    #         cur_params += t
    #         if tt_ranks[i] < tt_shapes[i - 1]:
    #             tt_ranks[i] += 1

    tt_ranks = [1] + tt_ranks + [1]
    # print(param, c)
    # print(tt_shapes, tt_ranks, param, c, ratio)
    return list(tt_ranks)


class TTLinear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, ranks=None,
                 in_shapes=None, out_shapes=None, dense_w: Tensor = None, dense_b: Tensor = None,
                 compression_ratio=None, dim=3):
        super().__init__()
        if in_shapes is not None and out_shapes is not None:
            self.tt_shapes = out_shapes + in_shapes
        else:
            in_shapes = split_to_factors(in_features, dim)
            out_shapes = split_to_factors(out_features, dim)
            self.tt_shapes = out_shapes + in_shapes
            # print(self.tt_shapes)
        self.tt_order = len(self.tt_shapes)

        self.out_tt_shapes = out_shapes
        self.in_tt_shapes = in_shapes
        self.in_tt_order = len(in_shapes)
        self.out_tt_order = len(out_shapes)

        assert in_features == int(np.prod(self.in_tt_shapes))
        assert out_features == int(np.prod(self.out_tt_shapes))

        self.in_features = in_features
        self.out_features = out_features

        if compression_ratio is not None:
            self.tt_ranks = compute_ranks_tt(self.tt_shapes, compression_ratio)
            # print(self.tt_ranks)
        else:
            assert ranks is not None
            self.tt_ranks = ranks

        self.tt_cores = ParameterList([Parameter(torch.Tensor(
            self.tt_ranks[i], self.tt_shapes[i], self.tt_ranks[i + 1])) for i in range(self.tt_order)])



        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        if dense_w is not None:
            w = dense_w.detach().cpu().numpy()
            tt_cores = ten2tt(w, self.tt_shapes, self.tt_ranks)

            for i in range(len(tt_cores)):
                self.tt_cores[i].data = torch.from_numpy(tt_cores[i])
        else:
            w = torch.randn([out_features, in_features]).detach().cpu().numpy()
            tt_cores = ten2tt(w, self.tt_shapes, self.tt_ranks)

            for i in range(len(tt_cores)):
                self.tt_cores[i].data = torch.from_numpy(tt_cores[i])
            self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.tt_order):
            init.xavier_uniform_(self.tt_cores[i])

    def get_core_size(self):
        size = 0
        for core in self.tt_cores:
            size += core.shape.numel()
        return size

    def forward(self, x):
        out_shape = list(x.shape)
        out_shape[-1] = self.out_features
        out = x
        for i in range(self.in_tt_order - 1, -1, -1):
            out = self.tt_cores[i + self.out_tt_order].reshape(
                -1, self.in_tt_shapes[i] * self.tt_ranks[i + self.out_tt_order + 1]).mm(
                out.reshape(-1, self.in_tt_shapes[i] * self.tt_ranks[i + self.out_tt_order + 1]).t()).t()

        for i in range(self.out_tt_order - 1, -1, -1):
            out = self.tt_cores[i].reshape(-1, self.tt_ranks[i + 1]).mm(out.reshape(-1, self.tt_ranks[i + 1]).t())
            out = out.reshape(self.tt_ranks[i], -1).t()

        out = out.reshape(self.out_features, -1).t().reshape(out_shape)

        if self.bias is not None:
            out += self.bias
        return out


if __name__ == '__main__':
    a = TTLinear(in_features=768, out_features=768,
             # in_shapes=[32, 24], out_shapes=[12, 8, 8],
             in_shapes=[32, 24], out_shapes=[32, 24],
             compression_ratio=140)
    print(a)
    a = TTLinear(in_features=768*4, out_features=768,
             in_shapes=[64, 48], out_shapes=[32, 24],
             # in_shapes=[8, 8, 8, 6], out_shapes=[12, 8, 8],
             compression_ratio=140)
    print(a)
    a = TTLinear(in_features=768, out_features=768*4,
             in_shapes=[32, 24], out_shapes=[64, 48],
             # in_shapes=[12, 8, 8], out_shapes=[8, 8, 8, 6],
             compression_ratio=140)
    print(a)
    # t = TTLinear(in_features=768, out_features=768*4, in_shapes=[12, 8, 8], out_shapes=[8, 8, 8, 6], compression_ratio=1678)
    # print(t.get_core_size())
    # print(t.tt_ranks)
    # w = torch.randn([768, 768*4])
    # b = torch.zeros(768)
    #
    # t = TTLinear(in_features=768*4, out_features=768, in_shapes=[64, 48], out_shapes=[32, 24], compression_ratio=2, dense_w=w, dense_b=b)
    # x = torch.randn([16, 768*4])
    # lin = torch.nn.Linear(in_features=768*4, out_features=768)
    # lin.weight.data = w
    # lin.bias.data = b
    # l = torch.nn.MSELoss()
    # print('TT, in_shapes=[64, 48], out_shapes=[32, 24]', l(t(x), lin(x)))
    #
    # t = TTLinear(in_features=768 * 4, out_features=768, in_shapes=[64*48], out_shapes=[32, 24], compression_ratio=2,
    #              dense_w=w, dense_b=b)
    # print('TT, in_shapes=[3072], out_shapes=[32, 24]', l(t(x), lin(x)))
    #
    # from SVDLinear import SVDLinear
    # t = SVDLinear(in_features=768 * 4, out_features=768,dense_w=w, dense_b=b, compression_ratio=2)
    # print('SVD', l(t(x), lin(x)))
