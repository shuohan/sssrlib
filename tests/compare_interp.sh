#!/usr/bin/env bash

dir=$(realpath $(dirname $0)/..)

docker run --gpus device=1 --rm -v $dir:$dir \
    -v $im_proc_3d_dir:$im_proc_3d_dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir:$im_proc_3d_dir -w $dir/tests -t \
    pytorch-shan:1.6.0-cuda10.1-cudnn7-runtime ./compare_interp.py
