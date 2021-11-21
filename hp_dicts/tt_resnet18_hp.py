# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/10/18 17:09

# in_order = len(in_tt_shapes)
# out_order = len(out_tt_shapes)

class HyperParamsDictSpecialRatio2x:
    tt_shapes = {
        'layer1.0.conv1.weight': [64, 9, 64],
        'layer1.0.conv2.weight': [64, 9, 64],
        'layer1.1.conv1.weight': [64, 9, 64],
        'layer1.1.conv2.weight': [64, 9, 64],
        'layer2.0.conv1.weight': [128, 9, 64],
        'layer2.0.conv2.weight': [128, 9, 128],
        'layer2.1.conv1.weight': [128, 9, 128],
        'layer2.1.conv2.weight': [128, 9, 128],
        'layer3.0.conv1.weight': [256, 9, 128],
        'layer3.0.conv2.weight': [256, 9, 256],
        'layer3.1.conv1.weight': [256, 9, 256],
        'layer3.1.conv2.weight': [256, 9, 256],
        'layer4.0.conv1.weight': [512, 9, 256],
        'layer4.0.conv2.weight': [512, 9, 512],
        'layer4.1.conv1.weight': [512, 9, 512],
        'layer4.1.conv2.weight': [512, 9, 512],
    }

    ranks = {
        'layer1.0.conv1.weight': [1, 45, 45, 1],
        'layer1.0.conv2.weight': [1, 36, 36, 1],
        'layer1.1.conv1.weight': [1, 32, 32, 1],
        'layer1.1.conv2.weight': [1, 32, 32, 1],
        'layer2.0.conv1.weight': [1, 80, 40, 1],
        'layer2.0.conv2.weight': [1, 70, 70, 1],
        'layer2.1.conv1.weight': [1, 65, 65, 1],
        'layer2.1.conv2.weight': [1, 64, 64, 1],
        'layer3.0.conv1.weight': [1, 130, 65, 1],
        'layer3.0.conv2.weight': [1, 120, 120, 1],
        'layer3.1.conv1.weight': [1, 120, 120, 1],
        'layer3.1.conv2.weight': [1, 115, 115, 1],
        'layer4.0.conv1.weight': [1, 250, 122, 1],
        'layer4.0.conv2.weight': [1, 240, 240, 1],
        'layer4.1.conv1.weight': [1, 230, 230, 1],
        'layer4.1.conv2.weight': [1, 220, 220, 1],
    }


class HyperParamsDictGeneralRatio2x:
    tt_shapes = {
        'layer1.0.conv1.weight': [8, 8, 9, 8, 8],
        'layer1.0.conv2.weight': [8, 8, 9, 8, 8],
        'layer1.1.conv1.weight': [8, 8, 9, 8, 8],
        'layer1.1.conv2.weight': [8, 8, 9, 8, 8],
        'layer2.0.conv1.weight': [16, 8, 9, 8, 8],
        'layer2.0.conv2.weight': [16, 8, 9, 8, 16],
        'layer2.1.conv1.weight': [16, 8, 9, 8, 16],
        'layer2.1.conv2.weight': [16, 8, 9, 8, 16],
        'layer3.0.conv1.weight': [16, 16, 9, 8, 16],
        'layer3.0.conv2.weight': [16, 16, 9, 16, 16],
        'layer3.1.conv1.weight': [16, 16, 9, 16, 16],
        'layer3.1.conv2.weight': [16, 16, 9, 16, 16],
        'layer4.0.conv1.weight': [32, 16, 9, 16, 16],
        'layer4.0.conv2.weight': [32, 16, 9, 16, 32],
        'layer4.1.conv1.weight': [32, 16, 9, 16, 32],
        'layer4.1.conv2.weight': [32, 16, 9, 16, 32],
    }

    in_tt_shapes = {
        'layer1.0.conv1.weight': [8, 8],
        'layer1.0.conv2.weight': [8, 8],
        'layer1.1.conv1.weight': [8, 8],
        'layer1.1.conv2.weight': [8, 8],
        'layer2.0.conv1.weight': [8, 8],
        'layer2.0.conv2.weight': [8, 16],
        'layer2.1.conv1.weight': [8, 16],
        'layer2.1.conv2.weight': [8, 16],
        'layer3.0.conv1.weight': [8, 16],
        'layer3.0.conv2.weight': [16, 16],
        'layer3.1.conv1.weight': [16, 16],
        'layer3.1.conv2.weight': [16, 16],
        'layer4.0.conv1.weight': [16, 16],
        'layer4.0.conv2.weight': [16, 32],
        'layer4.1.conv1.weight': [16, 32],
        'layer4.1.conv2.weight': [16, 32],
    }

    out_tt_shapes = {
        'layer1.0.conv1.weight': [8, 8],
        'layer1.0.conv2.weight': [8, 8],
        'layer1.1.conv1.weight': [8, 8],
        'layer1.1.conv2.weight': [8, 8],
        'layer2.0.conv1.weight': [16, 8],
        'layer2.0.conv2.weight': [16, 8],
        'layer2.1.conv1.weight': [16, 8],
        'layer2.1.conv2.weight': [16, 8],
        'layer3.0.conv1.weight': [16, 16],
        'layer3.0.conv2.weight': [16, 16],
        'layer3.1.conv1.weight': [16, 16],
        'layer3.1.conv2.weight': [16, 16],
        'layer4.0.conv1.weight': [32, 16],
        'layer4.0.conv2.weight': [32, 16],
        'layer4.1.conv1.weight': [32, 16],
        'layer4.1.conv2.weight': [32, 16],
    }

    ranks = {
        'layer1.0.conv1.weight': [1, 8, 45, 45, 8, 1],
        'layer1.0.conv2.weight': [1, 7, 34, 34, 7, 1],
        'layer1.1.conv1.weight': [1, 7, 32, 32, 7, 1],
        'layer1.1.conv2.weight': [1, 7, 32, 32, 7, 1],
        'layer2.0.conv1.weight': [1, 15, 75, 40, 8, 1],
        'layer2.0.conv2.weight': [1, 15, 64, 64, 15, 1],
        'layer2.1.conv1.weight': [1, 15, 64, 64, 15, 1],
        'layer2.1.conv2.weight': [1, 15, 60, 60, 15, 1],
        'layer3.0.conv1.weight': [1, 15, 140, 70, 15, 1],
        'layer3.0.conv2.weight': [1, 15, 120, 120, 15, 1],
        'layer3.1.conv1.weight': [1, 15, 120, 120, 15, 1],
        'layer3.1.conv2.weight': [1, 15, 110, 110, 15, 1],
        'layer4.0.conv1.weight': [1, 30, 236, 120, 15, 1],
        'layer4.0.conv2.weight': [1, 30, 230, 230, 30, 1],
        'layer4.1.conv1.weight': [1, 30, 220, 220, 30, 1],
        'layer4.1.conv2.weight': [1, 30, 210, 210, 30, 1],
    }

    in_ranks = {
        'layer1.0.conv1.weight': [64, 8, 1],
        'layer1.0.conv2.weight': [60, 8, 1],
        'layer1.1.conv1.weight': [60, 8, 1],
        'layer1.1.conv2.weight': [60, 8, 1],
        'layer2.0.conv1.weight': [64, 8, 1],
        'layer2.0.conv2.weight': [100, 15, 1],
        'layer2.1.conv1.weight': [100, 15, 1],
        'layer2.1.conv2.weight': [100, 15, 1],
        'layer3.0.conv1.weight': [100, 15, 1],
        'layer3.0.conv2.weight': [160, 15, 1],
        'layer3.1.conv1.weight': [160, 15, 1],
        'layer3.1.conv2.weight': [160, 15, 1],
        'layer4.0.conv1.weight': [128, 15, 1],
        'layer4.0.conv2.weight': [180, 30, 1],
        'layer4.1.conv1.weight': [180, 30, 1],
        'layer4.1.conv2.weight': [180, 30, 1],
    }

    out_ranks = {
        'layer1.0.conv1.weight': [1, 8, 64],
        'layer1.0.conv2.weight': [1, 8, 60],
        'layer1.1.conv1.weight': [1, 8, 60],
        'layer1.1.conv2.weight': [1, 8, 60],
        'layer2.0.conv1.weight': [1, 15, 128],
        'layer2.0.conv2.weight': [1, 15, 100],
        'layer2.1.conv1.weight': [1, 15, 100],
        'layer2.1.conv2.weight': [1, 15, 100],
        'layer3.0.conv1.weight': [1, 15, 200],
        'layer3.0.conv2.weight': [1, 15, 160],
        'layer3.1.conv1.weight': [1, 15, 160],
        'layer3.1.conv2.weight': [1, 15, 160],
        'layer4.0.conv1.weight': [1, 30, 256],
        'layer4.0.conv2.weight': [1, 30, 180],
        'layer4.1.conv1.weight': [1, 30, 180],
        'layer4.1.conv2.weight': [1, 30, 180],
    }
