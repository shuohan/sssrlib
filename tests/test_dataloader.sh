#!/usr/bin/env bash

dir=$(realpath $(dirname $0)/..)
im_proc_3d_dir=~/Code/shuo/utils/image-processing-3d
data_dir=/data

docker run --gpus device=1 --rm -v $dir:$dir \
    -v $data_dir:$data_dir \
    -v $im_proc_3d_dir:$im_proc_3d_dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir:$im_proc_3d_dir -w $dir/tests -t \
    pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime ./test_dataloader.py
