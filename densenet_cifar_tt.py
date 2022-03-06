import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import timm
from timm.models import register_model

from TKConv import TKConv2dC, TKConv2dM, TKConv2dR
from TTConv import TTConv2dM, TTConv2dR
from typing import Type, Any, Callable, Union, List, Optional, Tuple
import utils
import densenet_cifar


class TenBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0,
                 conv=Union[TKConv2dR, TKConv2dM, TKConv2dC, TTConv2dM, TTConv2dR],
                 stage=None, id=None, hp_dict=None, dense_dict=None):
        super(TenBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        w_name = 'block' + str(stage) + '.layer.' + str(id) + \
                 '.conv1.weight'
        if w_name in hp_dict.ranks:
            self.conv1 = conv(in_planes, out_planes, kernel_size=3,
                              stride=1, padding=1, bias=False,
                              hp_dict=hp_dict, name=w_name,
                              dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class TenBottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0,
                 conv=Union[TKConv2dR, TKConv2dM, TKConv2dC, TTConv2dM, TTConv2dR],
                 stage=None, id=None, hp_dict=None, dense_dict=None):
        super(TenBottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        w_name = 'block' + str(stage) + '.layer.' + str(id) + \
                 '.conv1.weight'
        if w_name in hp_dict.ranks:
            self.conv1 = conv(in_planes, inter_planes, kernel_size=1,
                              stride=1, padding=0, bias=False,
                              hp_dict=hp_dict, name=w_name,
                              dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        w_name = 'block' + str(stage) + '.layer.' + str(id) + \
                 '.conv2.weight'
        if w_name in hp_dict.ranks:
            self.conv2 = conv(inter_planes, out_planes, kernel_size=3,
                              stride=1, padding=1, bias=False,
                              hp_dict=hp_dict, name=w_name,
                              dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TenTransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0,
                 conv=Union[TKConv2dR, TKConv2dM, TKConv2dC, TTConv2dM, TTConv2dR],
                 stage=None, hp_dict=None, dense_dict=None):
        super(TenTransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        w_name = 'trans' + str(stage) + '.conv1.weight'
        if w_name in hp_dict.ranks:
            self.conv1 = conv(in_planes, out_planes,
                              kernel_size=1, stride=1, padding=0, bias=False,
                              hp_dict=hp_dict, name=w_name,
                              dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class TenDenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0,
                 conv=Union[TKConv2dR, TKConv2dM, TKConv2dC, TTConv2dM, TTConv2dR],
                 stage=None, hp_dict=None, dense_dict=None):
        super(TenDenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate, conv, stage, hp_dict,
                                      dense_dict)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate, conv, stage, hp_dict, dense_dict):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(in_planes + i * growth_rate, growth_rate, dropRate, conv, stage, i, hp_dict, dense_dict))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class TenDenseNet(nn.Module):
    def __init__(self, depth, num_classes=10, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0,
                 conv=Union[TKConv2dR, TKConv2dM, TKConv2dC, TTConv2dM, TTConv2dR],
                 hp_dict=None, dense_dict=None):
        super(TenDenseNet, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n / 2
            block = TenBottleneckBlock
        else:
            block = TenBasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = TenDenseBlock(n, in_planes, growth_rate, block, dropRate,
                                   conv, 1, hp_dict, dense_dict)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TenTransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate,
                                        conv, 1, hp_dict, dense_dict)
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.block2 = TenDenseBlock(n, in_planes, growth_rate, block, dropRate,
                                   conv, 2, hp_dict, dense_dict)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TenTransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate,
                                        conv, 2, hp_dict, dense_dict)
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.block3 = TenDenseBlock(n, in_planes, growth_rate, block, dropRate,
                                   conv, 3, hp_dict, dense_dict)
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


def _ten_densenet(num_layers, grow_rate=16, num_classes=10, reduction=0.5,
                  bottleneck=False, conv=Union[TKConv2dR, TKConv2dM, TKConv2dC],
                  hp_dict=None, dense_dict=None, **kwargs):
    if 'num_classes' in kwargs.keys():
        num_classes = kwargs.get('num_classes')
    model = TenDenseNet(num_layers, num_classes, grow_rate, reduction, bottleneck,
                        conv=conv, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if dense_dict is not None:
        ten_dict = model.state_dict()
        for key in ten_dict.keys():
            if key in dense_dict.keys():
                ten_dict[key] = dense_dict[key]
        model.load_state_dict(ten_dict)
    return model


@register_model
def tkr_densenet40(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _ten_densenet(40, bottleneck=False, conv=TKConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    baseline = 'densenet40'
    model_name = 'tkr_' + baseline
    hp_dict = utils.get_hp_dict(model_name, ratio='2')
    model = timm.create_model(model_name, hp_dict=hp_dict, decompose=None)
    x = torch.randn([1, 3, 32, 32])
    y = model(x)
    n_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad and 'conv' in name:
            print('\'{}\': {},'.format(name, list(p.shape)))
        n_params += p.numel()
    print('Total # parameters: {}'.format(n_params))
