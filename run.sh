#!/bin/bash

#python -m torch.distributed.launch --nproc_per_node 4 main.py --model resnet18 --ratio 2 --admm --epochs 180 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 55 --fp16 --num-workers 8 --smoothing 0.1 --distributed --save-log --save-model