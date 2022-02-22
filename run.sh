#!/bin/bash

#SBATCH -N 1

##SBATCH -p GPU
##SBATCH --gpus=v100-32:4

#SBATCH -p GPU-shared
#SBATCH --gpus=v100-16:4

#SBATCH -t 36:30:00 # Run time (HH:MM:SS)
#SBATCH -o out.%j  # Standard out and error in filename


echo 'use bridges-2'
RC=1
n=0

# /ocean/projects/asc200010p/czhang82/imagenet_tar (distributed file system)
# $LOCAL/imagenet (local disk) /local/imagenet

sourceDir=/ocean/projects/asc200010p/czhang82/imagenet_tar
destDir=$LOCAL/imagenet
while [[ $RC -ne 0 && $n -lt 20 ]]; do
    echo 'copy dataset to' $destDir
    rsync -avP $sourceDir $destDir
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
module load cuda/11.1.1
module load cudnn
conda activate miaoyin

#python -m torch.distributed.launch --nproc_per_node 4 main.py --model resnet18 --ratio 2 --admm --epochs 180 --format tk --pretrained --lr 0.01 --sched step --decay-epochs 55 --fp16 --num-workers 8 --smoothing 0.1 --distributed --save-log --save-model