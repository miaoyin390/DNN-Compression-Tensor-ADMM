import numpy as np
import torch

batch_size = 1


def tt_format(tt_inputs, out_tt_shapes, tt_ranks):
    print("tt-foramt:")
    for in_tt_shapes in tt_inputs:
        print(in_tt_shapes)
        tt_params = 0
        tt_flops = 0
        # TIMIT projection
        # in_tt_shapes = [200, 220, 250]
        # out_tt_shapes = [16]
        # tt_rank = 16
        # tt_ranks = [1] + [tt_rank] * (len(in_tt_shapes) + len(out_tt_shapes) - 1) + [1]

        in_tt_order = len(in_tt_shapes)
        out_tt_order = len(out_tt_shapes)
        tt_shapes = out_tt_shapes + in_tt_shapes
        in_features = int(np.prod(in_tt_shapes))
        out_features = int(np.prod(out_tt_shapes))
        params = in_features * out_features
        flops = batch_size * in_features * out_features

        tt_cores = [torch.randn(tt_ranks[i], tt_shapes[i], tt_ranks[i + 1]) for i in range(len(tt_shapes))]
        a = []
        for i in range(len(tt_cores)):
            a.append([tt_ranks[i], tt_shapes[i], tt_ranks[i + 1]])
            tt_params += int(np.prod(tt_cores[i].shape))
        print(a)
        print('# tt_params: {}'.format(tt_params))
        print('Compression ratio: {}'.format(params / tt_params))

        x = torch.randn(batch_size, in_features)

        out_shape = list(x.shape)
        out_shape[-1] = out_features
        out = x
        for i in range(in_tt_order - 1, -1, -1):
            a = tt_cores[i + out_tt_order].reshape(
                -1, in_tt_shapes[i] * tt_ranks[i + out_tt_order + 1])
            b = out.reshape(-1, in_tt_shapes[i] * tt_ranks[i + out_tt_order + 1]).t()
            out = a.mm(b).t()
            # print(a.shape, b.shape)
            tt_flops += a.shape[0] * a.shape[1] * b.shape[1]

        for i in range(out_tt_order - 1, -1, -1):
            a = tt_cores[i].reshape(-1, tt_ranks[i + 1])
            b = out.reshape(-1, tt_ranks[i + 1]).t()
            out = a.mm(b)
            # print(b.shape[0])
            out = out.reshape(tt_ranks[i], -1).t()
            tt_flops += a.shape[0] * a.shape[1] * b.shape[1]
            # print(a.shape, b.shape)

        print('# tt_flops: {}'.format(tt_flops))
        print('Speedup: {}'.format(flops / tt_flops))
    return flops, params, tt_flops, tt_params


def ttm_format(tt_inputs, out_tt_shapes, tt_ranks):
    print("ttm-foramt:")
    for in_tt_shapes in tt_inputs:
        # print(in_tt_shapes)
        tt_params = 0
        tt_flops = 0
        # in_tt_shapes = [200, 220, 250]
        # out_tt_shapes = [2, 2, 4]
        # tt_rank = 16
        # tt_ranks = [1] + [tt_rank] * (len(in_tt_shapes) - 1) + [1]
        in_features = int(np.prod(in_tt_shapes))
        out_features = int(np.prod(out_tt_shapes))
        params = in_features * out_features
        flops = batch_size * in_features * out_features

        ttm_cores = [(torch.Tensor(tt_ranks[i] * out_tt_shapes[i], in_tt_shapes[i] * tt_ranks[i + 1])) for i
                     in range(len(in_tt_shapes))]
        for c in ttm_cores:
            print(c.shape)
        for i in range(len(ttm_cores)):
            tt_params += int(np.prod(ttm_cores[i].shape))
        print('# ttm_params: {}'.format(tt_params))
        print('Compression ratio: {}'.format(params / tt_params))

        x = torch.randn(batch_size, in_features)

        out_shape = list(x.shape)
        out_shape[-1] = out_features
        res = x
        for k in range(len(in_tt_shapes) - 1, -1, -1):
            res = torch.reshape(res, [-1, in_tt_shapes[k]])
            res = torch.transpose(res, 0, 1)
            res = torch.reshape(res, [tt_ranks[k + 1] * in_tt_shapes[k], -1])
            tt_flops += ttm_cores[k].shape[0] * ttm_cores[k].shape[1] * res.shape[1]
            res = torch.matmul(ttm_cores[k], res)
        print('# ttm_flops: {}'.format(tt_flops))
        print('Speedup: {}'.format(flops / tt_flops))
    return flops, params, tt_flops, tt_params


def test_ttm56_d2():
    tt_rank = 7
    tt_input = [[32, 24]]
    tt_output = [32, 24]
    tt_ranks = [1, tt_rank, 1]
    a, b, c, d = ttm_format(tt_input, tt_output, tt_ranks)
    a *= 4
    b *= 4
    c *= 4
    d *= 4

    tt_rank = 12
    tt_input = [[64, 48]]
    tt_output = [32, 24]
    tt_ranks = [1, tt_rank, 1]
    a1, b1, c1, d1 = ttm_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    tt_rank = 13
    tt_input = [[32, 24]]
    tt_output = [64, 48]
    tt_ranks = [1, tt_rank, 1]
    a1, b1, c1, d1 = ttm_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    print('flops, params', a / c, b / d)


def test_ttm56_d3():
    tt_rank = 10
    tt_input = [[12, 8, 8]]
    tt_output = [12, 8, 8]
    tt_ranks = [1, tt_rank, tt_rank, 1]
    a, b, c, d = ttm_format(tt_input, tt_output, tt_ranks)
    a *= 4
    b *= 4
    c *= 4
    d *= 4

    tt_rank = 18
    tt_input = [[16, 16, 12]]
    tt_output = [12, 8, 8]
    tt_ranks = [1, tt_rank, tt_rank, 1]
    a1, b1, c1, d1 = ttm_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    tt_rank = 18
    tt_input = [[12, 8, 8]]
    tt_output = [16, 16, 12]
    tt_ranks = [1, tt_rank, tt_rank - 1, 1]
    a1, b1, c1, d1 = ttm_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    print('flops, params', a / c, b / d)

def test_tt48_d3():
    tt_rank = 18
    tt_input = [[24, 32]]
    tt_output = [768]
    tt_ranks = [1, tt_rank, tt_rank, 1]
    a, b, c, d = tt_format(tt_input, tt_output, tt_ranks)
    a *= 4
    b *= 4
    c *= 4
    d *= 4

    tt_rank = 17
    tt_input = [[48, 64]]
    tt_output = [768]
    tt_ranks = [1, tt_rank, tt_rank, 1]
    a1, b1, c1, d1 = tt_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    tt_rank = 18
    tt_input = [[768]]
    tt_output = [64, 48]
    tt_ranks = [1, tt_rank, tt_rank, 1]
    a1, b1, c1, d1 = tt_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    print('flops, params', a / c, b / d)

def test_tt56_d3():
    tt_rank = 9
    tt_input = [[32, 24]]
    tt_output = [768]
    tt_ranks = [1, tt_rank, tt_rank, 1]
    a, b, c, d = tt_format(tt_input, tt_output, tt_ranks)
    a *= 4
    b *= 4
    c *= 4
    d *= 4

    tt_rank = 20
    tt_input = [[64, 48]]
    tt_output = [768]
    tt_ranks = [1, tt_rank, tt_rank, 1]
    a1, b1, c1, d1 = tt_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    tt_rank = 23
    tt_input = [[768]]
    tt_output = [64, 48]
    tt_ranks = [1, tt_rank, tt_rank, 1]
    a1, b1, c1, d1 = tt_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    print('flops, params', a / c, b / d)


def test_tt56_d4():
    tt_rank = 13
    tt_input = [[32, 24]]
    tt_output = [32, 24]
    tt_ranks = [1, tt_rank, tt_rank, tt_rank, 1]
    a, b, c, d = tt_format(tt_input, tt_output, tt_ranks)
    a *= 4
    b *= 4
    c *= 4
    d *= 4

    tt_rank = 21
    tt_input = [[64, 48]]
    tt_output = [32, 24]
    tt_ranks = [1, tt_rank, tt_rank, tt_rank, 1]
    a1, b1, c1, d1 = tt_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    tt_rank = 22
    tt_input = [[32, 24]]
    tt_output = [64, 48]
    tt_ranks = [1, tt_rank, tt_rank, tt_rank, 1]
    a1, b1, c1, d1 = tt_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    print('flops, params', a / c, b / d)


def test_tt137_d4():
    tt_rank = 8
    tt_input = [[32, 24]]
    tt_output = [32, 24]
    tt_ranks = [1, tt_rank, tt_rank, tt_rank, 1]
    a, b, c, d = tt_format(tt_input, tt_output, tt_ranks)
    a *= 4
    b *= 4
    c *= 4
    d *= 4

    tt_rank = 13
    tt_input = [[64, 48]]
    tt_output = [32, 24]
    tt_ranks = [1, tt_rank, tt_rank, tt_rank, 1]
    a1, b1, c1, d1 = tt_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    tt_rank = 13
    tt_input = [[32, 24]]
    tt_output = [64, 48]
    tt_ranks = [1, tt_rank, tt_rank, tt_rank, 1]
    a1, b1, c1, d1 = tt_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    print('flops, params', a / c, b / d)

def test_tt1678_d5():
    tt_rank = 2
    tt_input = [[12, 8, 8]]
    tt_output = [12, 8, 8]
    tt_ranks = [1, tt_rank, tt_rank, tt_rank, tt_rank, tt_rank, 1]
    a, b, c, d = tt_format(tt_input, tt_output, tt_ranks)
    a *= 4
    b *= 4
    c *= 4
    d *= 4

    tt_rank = 5
    tt_input = [[8, 8, 8, 6]]
    tt_output = [12, 8, 8]
    tt_ranks = [1, tt_rank, tt_rank, tt_rank, tt_rank, tt_rank, tt_rank, 1]
    a1, b1, c1, d1 = tt_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    tt_rank = 5
    tt_input = [[12, 8, 8]]
    tt_output = [8, 8, 8, 6]
    tt_ranks = [1, tt_rank, tt_rank, tt_rank, tt_rank, tt_rank, tt_rank, 1]
    a1, b1, c1, d1 = tt_format(tt_input, tt_output, tt_ranks)
    a += a1
    b += b1
    c += c1
    d += d1

    print('flops, params', a / c, b / d)


if __name__ == '__main__':
    # test_tt1678_d5()
    test_tt48_d3()