# -*- coding:utf-8 -*-
# 
# Author: MIAO YIN
# Time: 2021/9/17 0:40

import torch
import timm
import numpy as np

import utils
from engines import train, eval
from parse_args import parse_args

import vit_tk
import resnet_cifar
import resnet_cifar_tk
import resnet_inet_tk
import resnet_inet_tt


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

    hp_dict = utils.get_hp_dict(args.model, args.ratio)

    if 'deit' in args.model:
        model_dict = dict(drop_rate=args.drop,
                          drop_path_rate=args.drop_path,
                          drop_block_rate=None
                          )
        args.input_size = 224
    else:
        model_dict = dict()

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
    if args.eval:
        print('===== Evaluation Mode =====')
        eval(model, args)
        return

    print('===== Training Mode =====')
    train(model, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
