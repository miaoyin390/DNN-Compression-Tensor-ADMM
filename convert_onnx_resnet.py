# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 12/6/23 1:23 PM

import torch
import timm
import os
import numpy as np
import torchvision.models

import utils
from engines import train, eval, eval_runtime
from parse_args import parse_args
from resnet_cifar_tt import TTConv2dM

import mobilenetv2
import mobilenetv2_tt
import vit_tt
import resnet_cifar
import resnet_cifar_tt
import resnet_inet_tt
import mobilenetv2_cifar
import mobilenetv2_cifar_tt
import densenet_cifar
import densenet_cifar_tt
import densenet_inet_tt
import vgg_tt

if __name__ == '__main__':
    baseline = 'resnet18'
    model_name = baseline
    # model_name = 'ttm_' + baseline
    # hp_dict = utils.get_hp_dict(model_name, ratio='2', tt_type='general')
    # model = timm.create_model(model_name, hp_dict=hp_dict, decompose=True, pretrained=True)
    model = timm.create_model(baseline)
    compr_params = 0
    for name, p in model.named_parameters():
        # if 'conv' in name or 'fc' in name:
        print(name, p.shape)
        if p.requires_grad:
            compr_params += int(np.prod(p.shape))

    x = torch.randn(1, 3, 224, 224)
    _ = model(x)
    print(compr_params)
    # _, base_flops, compr_flops = model.forward_flops(x)

    export_module = {
        TTConv2dM
    }

    # Export can work with named args but the dict containing named args has to be the last element of the args
    # tuple.
    # onnx_export(
    #     model,
    #     (dummy_inputs,),
    #     f=output.as_posix(),
    #     input_names=input_names,
    #     output_names=output_names,
    #     dynamic_axes=dict(chain(inputs.items(), config.outputs.items())),
    #     do_constant_folding=True,
    #     opset_version=opset,
    #     export_modules_as_functions=export_module
    # )

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

    # Export the model
    torch.onnx.export(model,         # model being run
                      dummy_input,       # model input (or a tuple for multiple inputs)
                      "./onnx/{}_nofunc.onnx".format(model_name),
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=17,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['modelInput'],   # the model's input names
                      output_names = ['modelOutput'], # the model's output names
                      # export_modules_as_functions=export_module,
                      )
    base_params = 0
    model = timm.create_model(baseline)

    for name, p in model.named_parameters():
        # if 'conv' in name or 'fc' in name:
        # print(name, p.shape)
        if p.requires_grad:
            base_params += int(np.prod(p.shape))
    print('Baseline # parameters: {}'.format(base_params))
    print('Compressed # parameters: {}'.format(compr_params))
    print('Compression ratio: {:.3f}'.format(base_params / compr_params))
    # print('Baseline # FLOPs: {:.2f}M'.format(base_flops))
    # print('Compressed # FLOPs: {:.2f}M'.format(compr_flops))
    # print('FLOPs ratio: {:.3f}'.format(base_flops / compr_flops))
