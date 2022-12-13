#!/bin/bash

#SBATCH -N 1

##SBATCH -p GPU
##SBATCH --gpus=v100-32:4

#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:4

#SBATCH -t 48:00:00 # Run time (HH:MM:SS)
#SBATCH -o out.%j  # Standard out and error in filename


echo 'use bridges-2'
RC=1
n=0

# /ocean/projects/asc200010p/czhang82/imagenet_tar (distributed file system)
# $LOCAL/imagenet (local disk) /local/imagenet

sourceDir=/ocean/projects/asc200010p/czhang82/imagenet_tar/
destDir=$LOCAL/imagenet
while [[ $RC -ne 0 && $n -lt 20 ]]; do
    echo 'copy dataset to' $destDir
    rsync -avP $sourceDir/* $destDir
    RC=$?
    n=$(( $n + 1 ))
    # let n = n + 1
    sleep 10
done
workDir=$(pwd)
cd $destDir
tar -xf train.tar
tar -xf val.tar
cd $workDir
##source $PROJECT/anaconda3/etc/profile.d/conda.sh
##conda activate miaoyin
module load cuda/11.1.1
module load cudnn

## command for temporarily apply for GPUs
## interact -p GPU-shared --gres=gpu:v100-32:2 -t 58:00

#python -m torch.distributed.launch --nproc_per_node 4 main.py --model resnet18 --ratio sc --admm --epochs 180 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 55 --fp16 --num-workers 6 --smoothing 0.1 --distributed --save-log --save-model --distillation-type hard --teacher-model resnet34
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_resnet18 --decompose --model-path ./saved_models/resnet18_imagenet_admm_tk_0222-022350_model.pt --ratio 2 --format tk --sched step --decay-epochs 40 --decay-rate 0.2 --lr 0.001 --epochs 150 --save-log --save-model --fp16 --num-workers 6 --distributed
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_resnet18 --decompose --model-path ./saved_models/resnet18_imagenet_admm_tk_0225-161144_model.pt --ratio sc --format tk --sched step --decay-epochs 45 --decay-rate 0.2 --lr 0.001 --epochs 160 --save-log --save-model --fp16 --num-workers 6 --distributed
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_resnet18 --decompose --model-path ./saved_models/resnet18_imagenet_admm_tk_0225-161144_model.pt --ratio sc --format tk --sched step --decay-epochs 45 --decay-rate 0.2 --lr 0.001 --epochs 160 --save-log --save-model --fp16 --num-workers 6 --distributed --distillation-type hard --teacher-model resnet34
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model resnet50 --ratio 3 --admm --epochs 200 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 60 --fp16 --num-workers 8 --distributed --save-log --save-model --distillation-type hard --teacher-model resnet50
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model resnet50 --ratio sc --admm --epochs 200 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 60 --fp16 --num-workers 8--distributed --save-log --save-model --distillation-type hard --teacher-model resnet50
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_resnet50 --decompose --pretrained --ratio 3 --lr 0.01 --epochs 180 --save-log --save-model --fp16 --num-workers 8 --distributed --distillation-type hard --teacher-model resnet50 
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_resnet50 --decompose --pretrained --ratio sc --lr 0.008 --epochs 180 --save-log --save-model --fp16 --num-workers 8 --distributed --distillation-type hard --teacher-model resnet50
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model vgg16_bn --ratio 10 --admm --epochs 160 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 50 --fp16 --num-workers 8 --distributed --save-log --save-model --distillation-type hard --teacher-model resnet34
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model densenet121 --ratio 2 --admm --epochs 200 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 60 --fp16 --num-workers 8 --distributed --save-log --save-model --distillation-type hard --teacher-model densenet169
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_densenet121 --decompose --model-path ./saved_models/densenet121_imagenet_admm_tk_0307-171848_model.pt --ratio 2 --format tk --lr 0.001 --epochs 180 --save-log --save-model --fp16 --num-workers 6 --distributed --distillation-type hard --teacher-model densenet169
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_vgg16_bn --ratio 2 --epochs 120 --format tk --decompose --model-path ./saved_models/vgg16_bn_imagenet_admm_tk_0308-025139_model.pt --lr 0.001 --fp16 --num-workers 8 --distributed --save-log --save-model --distillation-type hard --teacher-model resnet34
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_vgg16_bn --ratio 10 --epochs 150 --format tk --decompose --model-path ./saved_models/vgg16_bn_imagenet_admm_tk_0311-152539_model.pt --lr 0.001 --fp16 --num-workers 8 --distributed --save-log --save-model --distillation-type hard --teacher-model resnet34

#python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_resnet18 --decompose --model-path ./saved_models/resnet18_imagenet_admm_tk_0222-022350_model.pt --ratio 3 --format tk --sched step --decay-epochs 30 --decay-rate 0.2 --lr 0.001 --epochs 100 --save-log --save-model --fp16 --num-workers 6 --distributed
#python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_resnet18 --decompose --model-path ./saved_models/resnet18_imagenet_admm_tk_0222-022350_model.pt --ratio 4 --format tk --sched step --decay-epochs 30 --decay-rate 0.2 --lr 0.001 --epochs 100 --save-log --save-model --fp16 --num-workers 6 --distributed
python -m torch.distributed.launch --nproc_per_node 4 main.py --model tkc_resnet18 --decompose --model-path ./saved_models/resnet18_imagenet_admm_tk_0222-022350_model.pt --ratio 5 --format tk --sched step --decay-epochs 30 --decay-rate 0.2 --lr 0.001 --epochs 100 --save-log --save-model --fp16 --num-workers 6 --distributed
