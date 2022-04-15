# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2022/4/11 22:31

import numpy as np


in_tt_shapes = [8, 8, 8, 8]
out_tt_shapes = [4, 4, 4, 4]
tt_ranks = [1, 5, 5, 5, 1]

base_params = np.prod(in_tt_shapes) * np.prod(out_tt_shapes)
base_flops = base_params

tt_params = 0
tt_flops = 0
for k in range(len(in_tt_shapes)):
    tt_flops += np.prod(in_tt_shapes[k:]) * np.prod(out_tt_shapes[:k+1]) * tt_ranks[k] * tt_ranks[k+1]
    tt_params += in_tt_shapes[k] * out_tt_shapes[k] * tt_ranks[k] * tt_ranks[k+1]

print('compression ratio: {}'.format(base_params/tt_params))
print('FLOPs reduction: {}'.format(base_flops/tt_flops))

