# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2022/3/5 22:37

class HyperParamsDictRatio2x:
    kernel_shapes = {
        'features.0.weight':     [64, 3, 3, 3],
        'features.2.weight':     [64, 64, 3, 3],
        'features.5.weight':     [128, 64, 3, 3],
        'features.7.weight':     [128, 128, 3, 3],
        'features.10.weight':    [256, 128, 3, 3],
        'features.12.weight':    [256, 256, 3, 3],
        'features.14.weight':    [256, 256, 3, 3],
        'features.17.weight':    [512, 256, 3, 3],
        'features.19.weight':    [512, 512, 3, 3],
        'features.21.weight':    [512, 512, 3, 3],
        'features.24.weight':    [512, 512, 3, 3],
        'features.26.weight':    [512, 512, 3, 3],
        'features.28.weight':    [512, 512, 3, 3],
        'pre_logits.fc1.weight': [4096, 512, 7, 7],
        'pre_logits.fc2.weight': [4096, 4096, 1, 1],
    }

    ranks = {
        'features.2.weight':  [32, 64],
        'features.5.weight':  [64, 64],
        'features.7.weight':  [64, 64],
        'features.10.weight': [128, 128],
        'features.12.weight': [96, 128],
        'features.14.weight': [128, 160],
        'features.17.weight': [160, 160],
        'features.19.weight': [160, 196],
        'features.21.weight': [160, 196],
        'features.24.weight': [160, 196],
        'features.26.weight': [160, 196],
        'features.28.weight': [160, 196],
        'pre_logits.fc1.weight': [256, 288],
        # 'pre_logits.fc2.weight': [512],
    }


class HyperParamsDictRatio10x:
    ranks = {
        'features.2.weight':  [32, 32],
        'features.5.weight':  [32, 32],
        'features.7.weight':  [32, 32],
        'features.10.weight': [32, 64],
        'features.12.weight': [32, 64],
        'features.14.weight': [64, 64],
        'features.17.weight': [64, 64],
        'features.19.weight': [64, 96],
        'features.21.weight': [64, 96],
        'features.24.weight': [64, 96],
        'features.26.weight': [64, 96],
        'features.28.weight': [64, 96],
        'pre_logits.fc1.weight': [96, 96],
        # 'pre_logits.fc2.weight': [512],
    }
