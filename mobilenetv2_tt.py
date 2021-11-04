# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/11/3 14:51

import timm
import torch
import numpy as np


if __name__ == '__main__':
    model = timm.create_model('mobilenetv2_100')
    from hp_dicts.tk_mobilenetv2 import HyperParamsDictRatio2x as hp_dict
    params = 0
    tk_params = 0
    for n, p in model.named_parameters():
        params += int(np.prod(p.shape))
        if n in hp_dict.ranks.keys():
            ranks = hp_dict.ranks[n]
            tk_params += p.shape[0] * ranks[0] + p.shape[1] * ranks[1] + p.shape[2] * p.shape[3] * ranks[0] * ranks[1]
        else:
            tk_params += int(np.prod(p.shape))

    print(params)
    print(tk_params)