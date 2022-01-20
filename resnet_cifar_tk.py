# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/20 21:07


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import timm

from timm.models.registry import register_model

from TKConv import TKConv2dC, TKConv2dM, TKConv2dR
from typing import Type, Any, Callable, Union, List, Optional, Tuple
import utils
import resnet_cifar


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class TKBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A',
                 conv=Union[TKConv2dR, TKConv2dM, TKConv2dC],
                 stage=None, id=None, hp_dict=None, dense_dict=None):
        super().__init__()
        layer = 'layer' + str(stage) + '.' + str(id) + '.conv1'
        w_name = layer + '.weight'
        if dense_dict is None:
            self.conv1 = conv(in_planes, planes, hp_dict.ranks[w_name], 3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = conv(in_planes, planes, hp_dict.ranks[w_name], 3, stride=stride, padding=1, bias=False,
                              from_dense=True, dense_w=dense_dict[w_name])

        self.bn1 = nn.BatchNorm2d(planes)
        layer = 'layer' + str(stage) + '.' + str(id) + '.conv2'
        w_name = layer + '.weight'
        if dense_dict is None:
            self.conv2 = conv(planes, planes, hp_dict.ranks[w_name], 3, stride=1, padding=1, bias=False)
        else:
            self.conv2 = conv(planes, planes, hp_dict.ranks[w_name], 3, stride=1, padding=1, bias=False,
                              from_dense=True, dense_w=dense_dict[w_name])
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def forward_flops(self, x, name):
        base_flops = 0
        compr_flops = 0

        print('>{}:\t'.format(name + 'conv1'), end='', flush=True)
        if isinstance(self.conv1, (TKConv2dC, TKConv2dM, TKConv2dR)):
            out, flops1, flops2 = self.conv1.forward_flops(x)
            base_flops += flops1
            compr_flops += flops2
        else:
            out = self.conv1(x)
            base_flops += out.shape[2] * out.shape[3] * self.conv1.weight.numel() / 1000 / 1000
            compr_flops += out.shape[2] * out.shape[3] * self.conv2.weight.numel() / 1000 / 1000
        out = F.relu(self.bn1(out))

        print('>{}:\t'.format(name + 'conv2'), end='', flush=True)
        if isinstance(self.conv2, (TKConv2dC, TKConv2dM, TKConv2dR)):
            out, flops1, flops2 = self.conv2.forward_flops(out)
            base_flops += flops1
            compr_flops += flops2
        else:
            out = self.conv2(x)
            base_flops += out.shape[2] * out.shape[3] * self.conv2.weight.numel() / 1000 / 1000
            compr_flops += out.shape[2] * out.shape[3] * self.conv2.weight.numel() / 1000 / 1000
        out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)

        return out, base_flops, compr_flops


class TKResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, conv=Union[TKConv2dR, TKConv2dM, TKConv2dC],
                 hp_dict=None, dense_dict=None):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(TKBasicBlock, 16, num_blocks[0], stride=1, stage=1,
                                       conv=conv, hp_dict=hp_dict, dense_dict=dense_dict)
        self.layer2 = self._make_layer(TKBasicBlock, 32, num_blocks[1], stride=2, stage=2,
                                       conv=conv, hp_dict=hp_dict, dense_dict=dense_dict)
        self.layer3 = self._make_layer(TKBasicBlock, 64, num_blocks[2], stride=2, stage=3,
                                       conv=conv, hp_dict=hp_dict, dense_dict=dense_dict)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, stage, conv, hp_dict=None, dense_dict=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for id, stride in enumerate(strides):
            layers.append(
                block(self.in_planes, planes, stride, stage=stage, id=id, conv=conv,
                      hp_dict=hp_dict, dense_dict=dense_dict))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_flops(self, x):
        base_flops = 0
        compr_flops = 0
        out = F.relu(self.bn1(self.conv1(x)))
        for i, layer in enumerate(self.layer1):
            name = 'layer1.{}.'.format(str(i))
            out, flops1, flops2 = layer.forward_flops(out, name)
            base_flops += flops1
            compr_flops += flops2
        for i, layer in enumerate(self.layer2):
            name = 'layer2.{}.'.format(str(i))
            out, flops1, flops2 = layer.forward_flops(out, name)
            base_flops += flops1
            compr_flops += flops2
        for i, layer in enumerate(self.layer3):
            name = 'layer3.{}.'.format(str(i))
            out, flops1, flops2 = layer.forward_flops(out, name)
            base_flops += flops1
            compr_flops += flops2
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, base_flops, compr_flops


def _tk_resnet(num_blocks, num_classes=10, conv=Union[TKConv2dR, TKConv2dM, TKConv2dC], hp_dict=None,
               dense_dict=None, **kwargs):
    if 'num_classes' in kwargs.keys():
        num_classes = kwargs.get('num_classes')
    model = TKResNet(num_blocks, num_classes=num_classes, conv=conv, hp_dict=hp_dict, dense_dict=dense_dict)
    if dense_dict is not None:
        tk_dict = model.state_dict()
        for key in tk_dict.keys():
            if key in dense_dict.keys():
                tk_dict[key] = dense_dict[key]
        model.load_state_dict(tk_dict)

    return model


@register_model
def tkr_resnet32(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tk_resnet([5, 5, 5], conv=TKConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained:
        state_dict = torch.load(path, map_location='cpu')
        # test influence of Gaussian noise
        # for key in state_dict.keys():
        #     if 'conv' in key:
        #         shape = state_dict[key].shape
        #         watt = torch.mean(state_dict[key] ** 2)
        #         state_dict[key].add_(torch.from_numpy(np.random.normal(0, np.sqrt(0.01*watt), shape)))
        model.load_state_dict(state_dict)
    return model


@register_model
def tkr_resnet20(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tk_resnet([3, 3, 3], conv=TKConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkr_resnet56(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tk_resnet([9, 9, 9], conv=TKConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkm_resnet32(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tk_resnet([5, 5, 5], conv=TKConv2dM, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkm_resnet20(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tk_resnet([3, 3, 3], conv=TKConv2dM, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkc_resnet32(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tk_resnet([5, 5, 5], conv=TKConv2dC, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkc_resnet20(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tk_resnet([3, 3, 3], conv=TKConv2dC, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    baseline = 'resnet32'
    model_name = 'tkc_' + baseline
    hp_dict = utils.get_hp_dict(model_name, ratio='3')
    model = timm.create_model(model_name, hp_dict=hp_dict, decompose=None)
    compr_params = 0
    for name, p in model.named_parameters():
        # if 'conv' in name or 'fc' in name:
            # print(name, p.shape)
        if p.requires_grad:
            compr_params += int(np.prod(p.shape))

    x = torch.randn(1, 3, 32, 32)
    _, base_flops, compr_flops = model.forward_flops(x)
    base_params = 0
    model = timm.create_model(baseline)
    for name, p in model.named_parameters():
        # if 'conv' in name or 'fc' in name:
            # print(name, p.shape)
        if p.requires_grad:
            base_params += int(np.prod(p.shape))
    print('Baseline # parameters: {}'.format(base_params))
    print('Compressed # parameters: {}'.format(compr_params))
    print('Compression ratio: {:.3f}'.format(base_params/compr_params))
    print('Baseline # FLOPs: {:.2f}M'.format(base_flops))
    print('Compressed # FLOPs: {:.2f}M'.format(compr_flops))
    print('FLOPs ratio: {:.3f}'.format(base_flops/compr_flops))
