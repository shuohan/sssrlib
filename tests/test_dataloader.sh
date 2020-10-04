#!/usr/bin/env bash

dir=$(realpath $(dirname $0)/..)
im_proc_3d_dir=~/Code/shuo/utils/image-processing-3d

docker run --rm -v $dir:$dir \
    -v $im_proc_3d_dir:$im_proc_3d_dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir:$im_proc_3d_dir -w $dir/tests -t \
    pytorch-shan:1.6.0-cuda10.1-cudnn7-runtime ./test_dataloader.py
