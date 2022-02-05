import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from timm.models.registry import register_model

from TKConv import TKConv2dC, TKConv2dM, TKConv2dR
from TTConv import TTConv2dR, TTConv2dM
from typing import Type, Any, Callable, Union, List, Optional, Tuple
import utils
import mobilenetv2_cifar


class TTBaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t=6, downsample=False,
                 conv=Union[TKConv2dR, TKConv2dM, TKConv2dC, TTConv2dR, TTConv2dM],
                 id=None, hp_dict=None, dense_dict=None):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """
        super(TTBaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel)

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)

        # for main path:
        c = t * input_channel
        # 1x1   point wise conv
        w_name = 'bottlenecks.' + str(id) + '.conv1.weight'
        if w_name in hp_dict.ranks:
            self.conv1 = conv(input_channel, c, kernel_size=1, bias=False,
                              hp_dict=hp_dict, name=w_name,
                              dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv1 = nn.Conv2d(input_channel, c, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        # 3x3   depth wise conv
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=self.stride, padding=1, groups=c, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        # 1x1   point wise conv
        w_name = 'bottlenecks.' + str(id) + '.conv3.weight'
        if w_name in hp_dict.ranks:
            self.conv3 = conv(c, output_channel, kernel_size=1, bias=False,
                              hp_dict=hp_dict, name=w_name,
                              dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv3 = nn.Conv2d(c, output_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, inputs):
        # main path
        x = F.relu6(self.bn1(self.conv1(inputs)), inplace=True)
        x = F.relu6(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))

        # shortcut path
        x = x + inputs if self.shortcut else x

        return x


class TTMobileNetV2(nn.Module):
    def __init__(self, output_size=10, alpha=1, conv=Union[TKConv2dR, TKConv2dM, TKConv2dC, TTConv2dR, TTConv2dM],
                 hp_dict=None, dense_dict=None):
        super(TTMobileNetV2, self).__init__()
        self.output_size = output_size

        # first conv layer
        self.conv0 = nn.Conv2d(3, int(32 * alpha), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(int(32 * alpha))

        # build bottlenecks
        TTBaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            TTBaseBlock(32, 16, t=1, downsample=False, conv=conv, id=0, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(16, 24, downsample=False, conv=conv, id=1, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(24, 24, conv=conv, id=2, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(24, 32, downsample=False, conv=conv, id=3, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(32, 32, conv=conv, id=4, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(32, 32, conv=conv, id=5, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(32, 64, downsample=True, conv=conv, id=6, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(64, 64, conv=conv, id=7, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(64, 64, conv=conv, id=8, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(64, 64, conv=conv, id=9, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(64, 96, downsample=False, conv=conv, id=10, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(96, 96, conv=conv, id=11, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(96, 96, conv=conv, id=12, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(96, 160, downsample=True, conv=conv, id=13, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(160, 160, conv=conv, id=14, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(160, 160, conv=conv, id=15, hp_dict=hp_dict, dense_dict=dense_dict),
            TTBaseBlock(160, 320, downsample=False, conv=conv, id=16, hp_dict=hp_dict, dense_dict=dense_dict))

        # last conv layers and fc layer
        w_name = 'conv1.weight'
        if w_name in hp_dict.ranks:
            self.conv1 = conv(int(320*alpha), 1280, kernel_size=1, bias=False,
                              hp_dict=hp_dict, name=w_name,
                              dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv1 = nn.Conv2d(int(320 * alpha), 1280, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, output_size)

        # weights init
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):

        # first conv layer
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace=True)
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        # assert x.shape[1:] == torch.Size([1280,8,8])

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def _tt_mobilenetv2_cifar(num_classes=10, conv=Union[TKConv2dR, TKConv2dM, TKConv2dC],
                          hp_dict=None, dense_dict=None, **kwargs):
    if 'num_classes' in kwargs.keys():
        num_classes = kwargs.get('num_classes')
    model = TTMobileNetV2(output_size=num_classes, conv=conv, hp_dict=hp_dict, dense_dict=dense_dict)
    if dense_dict is not None:
        tk_dict = model.state_dict()
        for key in tk_dict.keys():
            if key in dense_dict.keys():
                tk_dict[key] = dense_dict[key]
        model.load_state_dict(tk_dict)

    return model


@register_model
def tkr_mobilenetv2_cifar(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_mobilenetv2_cifar(conv=TKConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    baseline = 'mobilenetv2_cifar'
    model_name = 'tkr_' + baseline
    hp_dict = utils.get_hp_dict(model_name, ratio='2')
    model = timm.create_model(model_name, num_classes=10, hp_dict=hp_dict, decompose=None)
    x = torch.randn([1, 3, 32, 32])
    y = model(x)
    n_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad and 'conv' in name:
            print('\'{}\': {},'.format(name, list(p.shape)))
        n_params += p.numel()
    print('Total # parameters: {}'.format(n_params))