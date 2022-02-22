#!/bin/bash

#SBATCH -N 1

##SBATCH -p GPU
##SBATCH --gpus=v100-32:8

#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1

#SBATCH -t 00:30:00 # Run time (HH:MM:SS)
#SBATCH -o out.%j  # Standard out and error in filename


echo 'use bridges-2'
RC=1
n=0

# /ocean/projects/asc200010p/czhang82/imagenet_tar (distributed file system)
# $LOCAL/imagenet_tar (local disk) /local/imagenet_tar

sourcedir=/ocean/projects/asc200010p/czhang82/imagenet_tar
while [[ $RC -ne 0 && $n -lt 20 ]]; do
    echo 'copy dataset to'$LOCAL/
    rsync -avP $sourcedir $LOCAL/
    RC=$?
    n=$(( $n + 1 ))
    # let n = n + 1
    sleep 10
done
workDir=$(pwd)
cd $LOCAL/imagenet_tar
tar -xf train.tar
tar -xf val.tar
cd $workDir
module load cuda/11.1.1
module load cudnn
conda activate miaoyin