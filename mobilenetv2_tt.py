# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2022/2/10 23:41

import timm
import math
import numpy as np
from timm.models import register_model
from torch import nn
import torch
import pickle
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional, Tuple

import utils
from TTConv import TTConv2dM, TTConv2dR
from TKConv import TKConv2dR, TKConv2dM, TKConv2dC
from SVDConv import SVDConv2dC, SVDConv2dR, SVDConv2dM
from mobilenetv2 import mobilenetv2
from timm.models.registry import register_model


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def tt_conv_1x1_bn(inp, oup, conv, hp_dict, name, dense_w):
    return nn.Sequential(
        conv(inp, oup, 1, 1, 0, bias=False, hp_dict=hp_dict, name=name, dense_w=dense_w),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class TTInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, id=None,
                 conv=Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR],
                 hp_dict=None, dense_dict=None):
        super(TTInvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        layers = []
        if expand_ratio == 1:
            # dw
            w_name = 'features.' + str(id) + '.conv.0.weight'
            if w_name in hp_dict.ranks:
                layers.append(conv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False,
                                   hp_dict=hp_dict, name=w_name,
                                   dense_w=None if dense_dict is None else dense_dict[w_name]))
            else:
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
            # pw-linear
            w_name = 'features.' + str(id) + '.conv.3.weight'
            if w_name in hp_dict.ranks:
                layers.append(conv(hidden_dim, oup, 1, 1, 0, bias=False, hp_dict=hp_dict, name=w_name,
                                   dense_w=None if dense_dict is None else dense_dict[w_name]))
            else:
                layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(oup))
        else:
            # pw
            w_name = 'features.' + str(id) + '.conv.0.weight'
            if w_name in hp_dict.ranks:
                layers.append(conv(inp, hidden_dim, 1, 1, 0, bias=False, hp_dict=hp_dict, name=w_name,
                                   dense_w=None if dense_dict is None else dense_dict[w_name]))
            else:
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
            # dw
            w_name = 'features.' + str(id) + '.conv.3.weight'
            if w_name in hp_dict.ranks:
                layers.append(conv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False,
                                   hp_dict=hp_dict, name=w_name,
                                   dense_w=None if dense_dict is None else dense_dict[w_name]))
            else:
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
            # pw-linear
            w_name = 'features.' + str(id) + '.conv.6.weight'
            if w_name in hp_dict.ranks:
                layers.append(conv(hidden_dim, oup, 1, 1, 0, bias=False, hp_dict=hp_dict, name=w_name,
                                   dense_w=None if dense_dict is None else dense_dict[w_name]))
            else:
                layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TTMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., cfgs=None,
                 conv=Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR],
                 hp_dict=None,
                 dense_dict=None):
        super(TTMobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ] if cfgs is None else cfgs

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = TTInvertedResidual
        id = 1
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t,
                                    id=id, conv=conv, hp_dict=hp_dict, dense_dict=dense_dict))
                input_channel = output_channel
                id += 1
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        w_name = 'conv.0.weight'
        if w_name in hp_dict.ranks:
            self.conv = tt_conv_1x1_bn(input_channel, output_channel, conv=conv, hp_dict=hp_dict, name=w_name,
                                       dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def _tt_mobilenetv2(conv, hp_dict, dense_dict=None, **kwargs):
    model = TTMobileNetV2(conv=conv, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if dense_dict is not None:
        tt_dict = model.state_dict()
        for key in tt_dict.keys():
            if key in dense_dict.keys():
                tt_dict[key] = dense_dict[key]
        model.load_state_dict(tt_dict)

    return model


@register_model
def ttr_mobilenetv2(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict_ = timm.create_model('mobilenetv2_100', pretrained=True).state_dict()
            dense_dict = mobilenetv2().state_dict()
            for k1, k2 in zip(dense_dict.keys(), dense_dict_.keys()):
                print(k1, k2)
                dense_dict[k1].data = dense_dict_[k2]
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_mobilenetv2(conv=TTConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


@register_model
def tkc_mobilenetv2(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict_ = timm.create_model('mobilenetv2_100', pretrained=True).state_dict()
            dense_dict = mobilenetv2().state_dict()
            for k1, k2 in zip(dense_dict.keys(), dense_dict_.keys()):
                print(k1, k2)
                dense_dict[k1].data = dense_dict_[k2]
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_mobilenetv2(conv=TKConv2dC, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


@register_model
def svdc_mobilenetv2(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict_ = timm.create_model('mobilenetv2_100', pretrained=True).state_dict()
            dense_dict = mobilenetv2().state_dict()
            for k1, k2 in zip(dense_dict.keys(), dense_dict_.keys()):
                print(k1, k2)
                dense_dict[k1].data = dense_dict_[k2]
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_mobilenetv2(conv=SVDConv2dC, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    model_name = 'svdc_mobilenetv2'
    hp_dict = utils.get_hp_dict(model_name, ratio='2')
    model = timm.create_model(model_name, hp_dict=hp_dict, decompose=None)
    n_params = 0
    for name, p in model.named_parameters():
        # if 'conv' in name or 'linear' in name:
        if p.requires_grad:
            print(name, p.shape)
            n_params += int(np.prod(p.shape))
    print('Total # parameters: {}'.format(n_params))