# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/17 0:40

import torch
import timm
import numpy as np

import utils
from engines import train, eval, eval_runtime
from parse_args import parse_args

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


def main(args):
    print(f"Creating model: {args.model}")

    if args.admm and (args.decompose or 'tt' in args.model or 'tk' in args.model):
        raise Exception('ERROR: ADMM mode does not support decomposed model!')

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.input_size = 32
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.input_size = 32
    elif args.dataset == 'imagenet':
        args.num_classes = 1000

    hp_dict = utils.get_hp_dict(args.model, args.ratio, tt_type=args.tt_type)

    if 'deit' in args.model or 'vit' in args.model:
        model_dict = dict(drop_rate=args.drop,
                          drop_path_rate=args.drop_path,
                          drop_block_rate=None
                          )
        args.input_size = 224
    else:
        model_dict = dict()

    if args.device == 'cuda':
        torch.cuda.set_device(f'cuda:{list(args.gpus)[0]}')

    model = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        decompose=args.decompose,
        path=args.model_path,
        hp_dict=hp_dict,
        **model_dict
    )

    # Evaluation
    if args.runtime:
        print('===== Runtime Mode =====')
        eval_runtime(model, args)
        return

    if args.eval:
        print('===== Evaluation Mode =====')
        eval(model, args)
        return

    print('===== Training Mode =====')
    train(model, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
