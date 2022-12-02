#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node 2 main.py --model resnet18 --ratio 2 --admm --epochs 180 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 55  --fp16 --num-workers 8 --distributed --save-model --save-log > rn18-tk-admm-0.log &

#CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node 2 main.py --model tkc_resnet18 --ratio 2 --epochs 160 --model-path ./saved_models/resnet18_imagenet_admm_tk_0224-175336_model.pt --decompose --lr 0.001 --sched step --decay-epochs 45  --fp16 --num-workers 8 --distributed --save-model --save-log --distillation-type hard --teacher-model resnet34 > rn18-tk-ft-0.log &

#CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch --nproc_per_node 2  main.py  --model resnet50 --ratio sc --admm --epochs 180 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 55 --save-model --save-log --fp16 --num-workers 8 --distributed > rn50-tk-admm-0.log &

#CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node 2 python main.py  --model tkc_resnet50 --decompose --model-path ./saved_models/resnet50_imagenet_admm_tt_1028-215851_model.pt --ratio 3 --sched step --decay-epochs 45 --decay-rate 0.2 --lr 0.001 --epochs 160 --save-log --save-model --fp16 --num-workers 8 --distillation-type hard --teacher-model resnet50 > r50-tt-general-ft-1.log &

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 main.py --model resnet18 --ratio 2 --admm --epochs 180 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 55 --fp16 --num-workers 8 --distributed

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_resnet18 --ratio 2 --decompose --epochs 180 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 55 --fp16 --num-workers 8 --distributed

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 main.py --model resnet50 --ratio 3 --admm --epochs 180 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 55 --fp16 --num-workers 8 --distributed

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_resnet50 --ratio 3 --decompose --epochs 180 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 55 --fp16 --num-workers 8 --distributed

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_resnet18 --ratio 3 --epochs 100 --model-path resnet18_imagenet_admm_tk_0222-022350_model.pt --decompose --lr 0.001 --sched step --decay-epochs 45  --fp16 --num-workers 8 --distributed --save-model --save-log
