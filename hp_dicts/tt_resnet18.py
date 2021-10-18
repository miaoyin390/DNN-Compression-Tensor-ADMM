# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/10/18 17:09


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
        'layer1.0.conv1.weight': [1, 64, 64, 1],
        'layer1.0.conv2.weight': [1, 64, 64, 1],
        'layer1.1.conv1.weight': [1, 64, 64, 1],
        'layer1.1.conv2.weight': [1, 64, 64, 1],
        'layer2.0.conv1.weight': [1, 120, 60, 1],
        'layer2.0.conv2.weight': [1, 100, 100, 1],
        'layer2.1.conv1.weight': [1, 100, 100, 1],
        'layer2.1.conv2.weight': [1, 100, 100, 1],
        'layer3.0.conv1.weight': [1, 200, 150, 1],
        'layer3.0.conv2.weight': [1, 135, 135, 1],
        'layer3.1.conv1.weight': [1, 135, 135, 1],
        'layer3.1.conv2.weight': [1, 135, 135, 1],
        'layer4.0.conv1.weight': [1, 320, 200, 1],
        'layer4.0.conv2.weight': [1, 170, 170, 1],
        'layer4.1.conv1.weight': [1, 170, 170, 1],
        'layer4.1.conv2.weight': [1, 170, 170, 1],
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

    ranks = {
        'layer1.0.conv1.weight': [1, 8, 64, 64, 8, 1],
        'layer1.0.conv2.weight': [1, 8, 60, 60, 8, 1],
        'layer1.1.conv1.weight': [1, 8, 60, 60, 8, 1],
        'layer1.1.conv2.weight': [1, 8, 60, 60, 8, 1],
        'layer2.0.conv1.weight': [1, 16, 128, 64, 8, 1],
        'layer2.0.conv2.weight': [1, 16, 100, 100, 8, 1],
        'layer2.1.conv1.weight': [1, 16, 100, 100, 8, 1],
        'layer2.1.conv2.weight': [1, 16, 100, 100, 8, 1],
        'layer3.0.conv1.weight': [1, 16, 200, 100, 16, 1],
        'layer3.0.conv2.weight': [1, 16, 160, 160, 16, 1],
        'layer3.1.conv1.weight': [1, 16, 160, 160, 16, 1],
        'layer3.1.conv2.weight': [1, 16, 160, 160, 16, 1],
        'layer4.0.conv1.weight': [1, 32, 256, 128, 16, 1],
        'layer4.0.conv2.weight': [1, 32, 180, 180, 32, 1],
        'layer4.1.conv1.weight': [1, 32, 180, 180, 32, 1],
        'layer4.1.conv2.weight': [1, 32, 180, 180, 32, 1],
    }