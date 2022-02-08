# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/11/3 14:51

import timm
import numpy as np
from timm.models import register_model
from torch import nn
from torchvision.models.utils import load_state_dict_from_url
import torch
import pickle
from torch import Tensor
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional, Tuple

from TTConv import TTConv2dM, TTConv2dR
from TKConv import TKConv2dR, TKConv2dM, TKConv2dC
from SVDConv import SVDConv2dC, SVDConv2dR, SVDConv2dM
from mobilenetv2 import mobilenetv2
from timm.models.registry import register_model

__all__ = ['TTMobileNetV2', 'ttr_mobilenetv2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class TTConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None,
                 conv=Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR],
                 hp_dict=None, name=None, dense_w=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(TTConvBNReLU, self).__init__(
            conv(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False,
                 hp_dict=hp_dict, name=name, dense_w=dense_w),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class TTInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None, id=None,
                 conv=Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR],
                 hp_dict=None, dense_dict=None):
        super(TTInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        w0_name = 'features.' + str(id) + '.conv.0.0.weight'
        # dw layer too compact to compress
        # w1_name = 'features.' + str(id) + '.conv.1.0.weight'
        w2_name = 'features.' + str(id) + '.conv.2.weight'
        if expand_ratio != 1:
            # pw
            if w0_name in hp_dict.ranks:
                layers.append(TTConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                           conv=conv, hp_dict=hp_dict, name=w0_name,
                                           dense_w=None if dense_dict is None else dense_dict[w0_name]))
            else:
                layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            conv(hidden_dim, oup, 1, 1, 0, bias=False, hp_dict=hp_dict, name=w2_name,
                 dense_w=None if dense_dict is None else dense_dict[w2_name]) if w2_name in hp_dict.ranks
            else nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TTMobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 conv=Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR],
                 hp_dict=None,
                 dense_dict=None,
                 ):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(TTMobileNetV2, self).__init__()

        if block is None:
            block = TTInvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        id = 1
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer,
                                      id=id, conv=conv, hp_dict=hp_dict, dense_dict=dense_dict))
                id += 1
                input_channel = output_channel
        # building last several layers
        w_name = 'features.18.0.weight'
        if w_name in hp_dict.ranks:
            features.append(TTConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer,
                                         conv=conv, hp_dict=hp_dict, name=w_name,
                                         dense_w=None if dense_dict is None else dense_dict[w_name]))
        else:
            features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


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


if __name__ == '__main__':
    from hp_dicts.tt_mobilenetv2_hp import HyperParamsDictRatio2x as hp_dict
    model = timm.create_model('ttr_mobilenetv2', hp_dict=hp_dict, decompose=None)
    n_params = 0
    for name, p in model.named_parameters():
        # if 'conv' in name or 'linear' in name:
        if p.requires_grad:
            print(name, p.shape)
            n_params += int(np.prod(p.shape))
    print('Total # parameters: {}'.format(n_params))