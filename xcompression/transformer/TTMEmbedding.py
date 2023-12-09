# -*- coding:utf-8 -*-
# 
# Author: Jinqi Xiao
# Time: Feb/11/2022
import math
from functools import reduce

import numpy as np
import torch
from torch.nn import Module, ParameterList, Parameter
from torch.nn import init
from torch.utils.benchmark import timer


class TTMEmbedding(Module):
    __constants__ = ['input_tt_shape', 'output_tt_shape', 'tt_ranks']

    def __init__(self, input_tt_shape, output_tt_shape, tt_ranks):
        super(TTMEmbedding, self).__init__()
        self.input_tt_shape = input_tt_shape
        self.output_tt_shape = output_tt_shape
        self.n_dim = len(input_tt_shape)
        self.output_size = reduce(lambda x, y: x * y, self.output_tt_shape)
        # calculating index
        self.tt_index_factor = [input_tt_shape[-1]]
        for i in range(len(input_tt_shape) - 2, 0, -1):
            self.tt_index_factor.append(input_tt_shape[i] * self.tt_index_factor[len(input_tt_shape) - 2 - i])
        self.tt_index_factor.reverse()
        self.tt_ranks = tt_ranks
        self.cores = ParameterList([Parameter(
            torch.Tensor(self.tt_ranks[i], self.input_tt_shape[i], self.output_tt_shape[i], self.tt_ranks[i + 1])) for i
            in range(self.n_dim)])
        self.reset_parameters()

    def get_core_size(self):
        size = 0
        for i in range(self.n_dim):
            size += self.tt_ranks[i] * self.input_tt_shape[i] * self.output_tt_shape[i] * self.tt_ranks[i + 1]
        return size

    def get_tt_ranks(self):
        str_tt_ranks = [str(r) for r in self.tt_ranks]
        return ', '.join(str_tt_ranks)

    def reset_parameters(self):
        for i in range(self.n_dim):
            init.xavier_uniform_(self.cores[i])

    def initialize(self, embeddings, steps=1000):
        if type(embeddings) != torch.Tensor:
            embeddings = torch.Tensor(embeddings)
        embeddings.cuda()
        self.cores.cuda()
        optimizer = torch.optim.SGD(self.parameters(), lr=1.0)
        for i in range(steps):
            loss = torch.nn.MSELoss(self.cores.data, embeddings)
            loss.backward(retain_graph=True)
            print("compressing word embeddings with loss {}".format(loss.item()))
            optimizer.step()

    def compute_ranks_ttm(self, ratio):
        input_shape = self.input_tt_shape
        output_shape = self.output_tt_shape
        param = np.prod(input_shape) * np.prod(output_shape)
        d = len(input_shape)
        p = param / (input_shape[0] * output_shape[0])
        maxr = [int(min(input_shape[0] * output_shape[0], p))]
        for i in range(1, d - 1):
            p = p / (input_shape[i] * output_shape[i])
            maxr.append(int(min(maxr[i - 1] * input_shape[i] * output_shape[i], p)))
        # print(maxr)
        c = input_shape[0] * output_shape[0] * maxr[0] + input_shape[-1] * output_shape[-1] * maxr[-1] - param / ratio
        b = input_shape[1] * output_shape[1] * maxr[1] * maxr[0] + input_shape[-2] * output_shape[-2] * maxr[-1] * maxr[
            -2]
        a = 0
        for i in range(2, d - 2):
            a += maxr[i - 1] * maxr[i] * input_shape[i] * output_shape[i]
        if a == 0:
            x = -c / b
        else:
            x = (-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        assert 0 < x < 1
        for i in range(1, d - 1):
            maxr[i] = int(maxr[i] * x)
        # print(maxr)
        tt_ranks = [1] + maxr + [1]
        features = np.prod(input_shape) * np.prod(output_shape)
        for i in range(d - 1):
            f = tt_ranks[i] * input_shape[i] * output_shape[i]
            s = features / f
            min_r = min(f, s)
            if min_r < tt_ranks[i + 1]:
                tt_ranks[i + 1] = int(min_r)
        return list(tt_ranks)

    def forward(self, input):
        # get tensorized index
        input = input.view(-1)
        rem = input
        index = []
        for factor in self.tt_index_factor:
            val = torch.div(rem, factor, rounding_mode='floor')
            rem = torch.fmod(rem, factor)
            index.append(val)
        index.append(rem)
        index = torch.stack(index).T

        batch_size = int(index.shape[0])
        for k, core in enumerate(self.cores):
            i = index[:, k]
            cur_slice = torch.index_select(core, 1, i)
            # r x B x M x r
            if k == 0:
                res = cur_slice.transpose(0, 1)
                # B x r x M x r
            else:
                res = res.contiguous().view(batch_size, -1, self.tt_ranks[k])
                # B x rM x r
                curr_core = cur_slice.view(self.tt_ranks[k], batch_size, -1)
                # r x B x Mr
                res = torch.einsum('oqb,bow->oqw', (res, curr_core))
        res = torch.einsum('i...i->...', res.view(batch_size, self.tt_ranks[0], res.shape[1] // self.tt_ranks[0], -1,
                                                  self.tt_ranks[0]).transpose(0, 1))

        res = res.reshape(-1, self.output_size)
        res = res.view(input.shape[0], -1)
        res_shape = list(input.shape) + [self.output_size, ]
        res = res.view(*res_shape)
        return res.to(input.device)



if __name__ == '__main__':
    nbatches = 256
    times = 2000
    tt_inputs = [[200, 220, 250], [125, 130, 136], [200, 200, 209], [166, 175, 188], [200, 200, 200], [53, 72, 75],
                 [50, 52, 55]]
    for input_tt_shape in tt_inputs:
        tt = TTMEmbedding(input_tt_shape=input_tt_shape, output_tt_shape=[2, 2, 4], tt_ranks=[1, 16, 16, 1])
        a = torch.tensor(np.random.randint(4000, size=nbatches))
        for i in range(50):
            tt(a)
        torch.cuda.synchronize()
        tic = timer()
        for i in range(times):
            tt(a)
        torch.cuda.synchronize()
        toc = timer()
        latency = (toc - tic) * 1000 / times
        print(input_tt_shape)
        print("latency:", latency)
