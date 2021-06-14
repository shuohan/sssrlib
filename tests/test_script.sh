#!/usr/bin/env bash

dir=$(realpath $(dirname $0)/..)
improc3d_dir=~/Code/shuo/utils/improc3d
datadir=/data
n=10000

docker run --gpus device=1 --rm -v $dir:$dir \
    -v $improc3d_dir:$improc3d_dir \
    -v $datadir:$datadir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir:$improc3d_dir -w $dir/tests -t \
    pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime ../scripts/sample_patches.py \
    -i $datadir/smore_simu/orig_data/sub-OAS30108_ses-d0168_T2w_initnorm.nii.gz \
    -o /data/compare_patches/uniform_32-32-1_${n}_run0 -a -n $n -p 32 32 1

docker run --gpus device=1 --rm -v $dir:$dir \
    -v $improc3d_dir:$improc3d_dir \
    -v $datadir:$datadir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir:$improc3d_dir -w $dir/tests -t \
    pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime ../scripts/sample_patches.py \
    -i $datadir/smore_simu/orig_data/sub-OAS30108_ses-d0168_T2w_initnorm.nii.gz \
    -o /data/compare_patches/uniform_32-32-1_${n}_run1 -a -n $n -p 32 32 1

docker run --gpus device=1 --rm -v $dir:$dir \
    -v $improc3d_dir:$improc3d_dir \
    -v $datadir:$datadir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir:$improc3d_dir -w $dir/tests -t \
    pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime ../scripts/sample_patches.py \
    -i $datadir/smore_simu/orig_data/sub-OAS30108_ses-d0168_T2w_initnorm.nii.gz \
    -o /data/compare_patches/uniform_1-32-32_${n} -a -n $n -p 1 32 32

docker run --gpus device=1 --rm -v $dir:$dir \
    -v $improc3d_dir:$improc3d_dir \
    -v $datadir:$datadir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir:$improc3d_dir -w $dir/tests -t \
    pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime ../scripts/sample_patches.py \
    -i $datadir/smore_simu/orig_data/sub-OAS30108_ses-d0168_T2w_initnorm.nii.gz \
    -o /data/compare_patches/uniform_32-1-32_${n} -a -n $n -p 32 1 32
