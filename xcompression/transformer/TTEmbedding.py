import math
from functools import reduce

import numpy as np
import tensorly as tl
from numpy.linalg import svd

tl.set_backend("pytorch")
import torch
from torch.nn import Module, ParameterList, Parameter
from torch.nn import init


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

    # for i in range(d - 1):
    #     if i == 0:
    #         t = tt_shapes[i] + tt_shapes[i + 1] * tt_ranks[i + 1]
    #     elif i == d - 2:
    #         t = tt_shapes[-1] + tt_shapes[i] * tt_ranks[i - 1]
    #     else:
    #         t = tt_shapes[i] * tt_ranks[i - 1] + tt_shapes[i + 1] * tt_ranks[i + 1]
    #     if (cur_params + t) <= c:
    #         cur_params += t
    #         if tt_ranks[i] < tt_shapes[i-1]:
    #             tt_ranks[i] += 1

    tt_ranks = [1] + tt_ranks + [1]
    # print(param, c, cur_params)
    # print(tt_shapes, tt_ranks, param, c, param / cur_params)
    return list(tt_ranks)


class TTEmbedding(Module):
    __constants__ = ['input_tt_shape', 'output_tt_shape', 'tt_ranks']

    def __init__(self, input_tt_shape, output_tt_shape, tt_ranks=None, compression_ratio=None):
        super(TTEmbedding, self).__init__()
        if compression_ratio is not None:
            self.tt_ranks = compute_ranks_tt(input_tt_shape + output_tt_shape, compression_ratio)
        else:
            assert tt_ranks is not None
            self.tt_ranks = tt_ranks

        self.input_tt_shape = input_tt_shape
        self.output_tt_shape = output_tt_shape
        self.output_tt_shape = output_tt_shape
        self.output_size = reduce(lambda x, y: x * y, self.output_tt_shape)
        # calculating index
        self.tt_index_factor = [input_tt_shape[-1]]
        for i in range(len(input_tt_shape) - 2, 0, -1):
            self.tt_index_factor.append(input_tt_shape[i] * self.tt_index_factor[len(input_tt_shape) - 2 - i])
        self.tt_index_factor.reverse()

        self.tt_shapes = self.input_tt_shape + self.output_tt_shape
        self.cores = ParameterList(
            [Parameter(torch.Tensor(self.tt_ranks[i], self.tt_shapes[i], self.tt_ranks[i + 1])) for i in
             range(len(self.tt_shapes))])

        self.reset_parameters()

    def get_core_size(self):
        size = 0
        for i in range(len(self.cores)):
            size += int(self.cores[i].data.numel())
        return size

    def get_tt_ranks(self):
        str_tt_ranks = [str(r) for r in self.tt_ranks]
        return ', '.join(str_tt_ranks)

    def reset_parameters(self):
        for i in range(len(self.cores)):
            init.xavier_uniform_(self.cores[i])

    def tt_reduce_fun(self, x, y):
        return torch.bmm(x, y.permute([1, 0, 2]))

    def forward(self, input):
        # get tensorized index
        input_shape = input.shape
        input = input.view(-1)
        rem = input
        index = []
        for factor in self.tt_index_factor:
            val = torch.div(rem, factor, rounding_mode='floor')
            rem = torch.fmod(rem, factor)
            index.append(val)
        index.append(rem)
        index = torch.stack(index).T

        tmp_cores = []
        for i, col in enumerate(index.unbind(1)):
            tmp_cores.append(self.cores[i][:, col, :])
            # print(tmp_cores[i].shape)
        tmp_cores[0] = tmp_cores[0].permute([1, 0, 2])
        reduced = reduce(self.tt_reduce_fun, tmp_cores)
        tmp_factors = [reduced.permute([1, 0, 2])]
        for core in self.cores[-len(self.output_tt_shape):]:
            tmp_factors.append(core)
        res = tl.tt_to_tensor(tmp_factors).view(-1, self.output_size)

        res = res.view(input.shape[0], -1)
        res_shape = list(input_shape) + [self.output_size, ]
        res = res.view(*res_shape)
        return res.to(input.device)

    def forward2(self, input):
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

        tmp_cores = []
        for i, col in enumerate(index.unbind(1)):
            tmp_cores.append(self.cores[i][:, col, :])

        res = tmp_cores[0].reshape(-1, self.tt_ranks[1])
        for i in range(1, len(tmp_cores)):
            curr_core = torch.reshape(tmp_cores[i], (self.tt_ranks[i], -1))
            res = torch.matmul(res, curr_core)
        res = torch.matmul(res, self.cores[-1].reshape(-1, self.tt_ranks[-2]))
        return res

    def init_pretrained_emb(self, emb):
        tt_ranks = self.tt_ranks.copy()
        d = len(self.tt_shapes)
        t = emb.detach().numpy()
        tt_cores = []
        for i in range(d - 1):
            # print(t.shape)
            t = np.reshape(t, [self.tt_ranks[i] * self.tt_shapes[i], -1])
            # print(t.shape)
            u, s, v = svd(t, full_matrices=False)
            if s.shape[0] < tt_ranks[i + 1]:
                tt_ranks[i + 1] = s.shape[0]

            u = u[:, :tt_ranks[i + 1]]
            s = s[:tt_ranks[i + 1]]
            v = v[:tt_ranks[i + 1], :]

            tt_cores.append(np.reshape(u, [tt_ranks[i], self.tt_shapes[i], tt_ranks[i + 1]]))
            t = np.dot(np.diag(s), v)
        # print(t.shape)
        t = np.reshape(t, [tt_ranks[d - 1], self.tt_shapes[d - 1], tt_ranks[d]])
        # print(t.shape)
        tt_cores.append(t)
        # print(tt_ranks)
        for i in range(len(tt_cores)):
            self.cores[i].data = torch.from_numpy(tt_cores[i])

    def restore_weights(self):
        d = len(self.cores)
        t = self.cores[0]
        for i in range(1, d):
            rank = self.cores[i].shape[0]
            t = t.view(-1, rank)
            t = t @ self.cores[i].reshape(rank, -1)
        t = t.view(-1, 16)
        return t

    def tt2ten(self):
        d = len(self.cores)
        t = self.cores[0].detach().numpy()
        for i in range(1, d):
            rank = self.cores[i].shape[0]
            t = np.reshape(t, [-1, rank])
            t = np.dot(t, np.reshape(self.cores[i].detach().numpy(), [rank, -1]))
        t = np.reshape(t, self.tt_shapes)
        return torch.from_numpy(t)


# def test_latency():
#     nbatches = 256
#     times = 2000
#     tt_rank = 16
#     tt_inputs = [[200, 220, 250], [125, 130, 136], [200, 200, 209], [166, 175, 188], [200, 200, 200], [53, 72, 75],
#                  [50, 52, 55]]
#     for input_tt_shape in tt_inputs:
#         # input_tt_shape = [250, 220, 200]
#         output_tt_shape = [2, 2, 4]
#         tt_ranks = [1] + [tt_rank] * (len(input_tt_shape) + len(output_tt_shape) - 1) + [1]
#         tt = TTEmbedding(input_tt_shape=input_tt_shape, output_tt_shape=output_tt_shape, tt_ranks=tt_ranks)
#         a = torch.tensor([1360])
#         for i in range(50):
#             tt(a)
#         torch.cuda.synchronize()
#         tic = timer()
#         for i in range(times):
#             tt(a)
#         torch.cuda.synchronize()
#         toc = timer()
#         latency = (toc - tic) * 1000 / times
#         print(input_tt_shape)
#         print("latency:", latency)


if __name__ == '__main__':
    # test_latency()
    # input_tt_shape = [50, 52, 55]
    # ot = torch.randn([50 * 52 * 55, 16])
    # output_tt_shape = [2, 2, 4]
    # tt_ranks = compute_ranks_tt(input_tt_shape + output_tt_shape, 3)
    # print("ranks:", tt_ranks)
    # # tt_ranks = [1] + [tt_rank] * (len(input_tt_shape) + len(output_tt_shape) - 1) + [1]
    # tt = TTEmbedding(input_tt_shape=input_tt_shape, output_tt_shape=output_tt_shape, tt_ranks=tt_ranks)
    # print("compressed:", ot.numel() / tt.get_core_size())
    # print(torch.sum((tt.restore_weights() - ot).abs()))
    # tt.init_pretrained_emb(ot)
    # tt.initialize(ot)
    a = TTEmbedding(input_tt_shape=[30522], output_tt_shape=[768], compression_ratio=5)
    print(a)
    # print(a(torch.LongTensor([3, 6, 10, 12])))
