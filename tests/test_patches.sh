#!/usr/bin/env bash

dir=$(realpath $(dirname $0)/..)
im_proc_3d_dir=~/Code/shuo/utils/image-processing-3d

docker run --gpus device=1 --rm -v $dir:$dir \
    -v $im_proc_3d_dir:$im_proc_3d_dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir:$im_proc_3d_dir -w $dir/tests -t \
    psf-est ./test_patches.py
