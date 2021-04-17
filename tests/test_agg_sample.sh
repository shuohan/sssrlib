#!/usr/bin/env bash

dir=$(realpath $(dirname $0)/..)
improc3d_dir=~/Code/shuo/utils/improc3d

docker run --gpus device=1 --rm -v $dir:$dir \
    -v $improc3d_dir:$improc3d_dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir:$improc3d_dir -w $dir/tests -t \
    pytorch-shan:1.7.0-cuda11.0-cudnn8-runtime ./test_agg_sample.py
