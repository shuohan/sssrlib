#!/usr/bin/env python

import torch
import os
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from improc3d import transform_to_axial
import time

from sssrlib.patches import Patches, PatchesCollection, TransformedPatches
from sssrlib.sample import SamplerCollection, Sampler, ImageGradients
from sssrlib.sample import SuppressWeights, GradSampleWeights
from sssrlib.transform import Flip
from sssrlib.utils import calc_avg_kernel


def test_patches():
    ps2d = (1, 128, 128)
    ps2d_t = (128, 1, 128)
    fn = '/data/smore_simu/orig_data/sub-OAS30108_ses-d0168_T2w_initnorm.nii.gz'
    obj = nib.load(fn)
    image = obj.get_fdata(dtype=np.float32)
    image = transform_to_axial(image, obj.affine, coarse=True).copy()

    print('Create patches')
    patches0 = Patches(ps2d, image, x=2, y=1, z=0)
    patches1 = Patches(ps2d, image, x=2, y=0, z=1)
    patches2 = Patches(ps2d_t, patches=patches1)
    patches3 = Patches(ps2d_t, patches=patches2)
    patches3 = TransformedPatches(patches3, Flip((-2, )))
    patches = PatchesCollection(patches0, patches1, patches2, patches3)
    
    assert len(patches2) == len(patches3)
    assert len(patches) == len(patches0) + len(patches1) + len(patches2) + len(patches3)
    assert torch.equal(patches[2, 20].data, patches2[20].data)
    assert patches[2, 10].name.startswith('p2')

    dirname = Path('results_patches-collection')
    os.system('rm -rf %s' % dirname)
    dirname.mkdir(exist_ok=True)
    patches.save_figures(Path(dirname, 'images'), d3=False)

    print('Uniform')
    sampler_uniform = Sampler(patches0)
    indices = sampler_uniform.sample_indices(50)
    batch = sampler_uniform.get_patches(indices)
    Path(dirname, 'uniform').mkdir(exist_ok=True)
    for i in range(batch.data.shape[0]):
        data = batch.data[i, 0, ...]
        name = batch.name[i]
        plt.imsave(Path(dirname, 'uniform', name + '.jpg'), data.squeeze(), cmap='gray')

    agg_kernel_yz = calc_avg_kernel(ps2d)
    agg_kernel_xz = calc_avg_kernel(ps2d_t)
    kernel_size = (4, 4, 1)
    stride = (2, 2, 1)

    print('Create gradients')
    start_time = time.time()
    grads0 = ImageGradients(patches0, sigma=3)
    grads1 = ImageGradients(patches1, sigma=3)
    grads0.save_figures(Path(dirname, 'grads', '0'), d3=False)
    grads1.save_figures(Path(dirname, 'grads', '1'), d3=False)
    print('Time', time.time() - start_time)

    print('Create weights')
    start_time = time.time()
    weights0 = GradSampleWeights(patches0, (grads0.gradients[1], ), agg_kernel=agg_kernel_yz)
    weights0 = SuppressWeights(weights0, kernel_size=kernel_size, stride=stride)

    weights11 = GradSampleWeights(patches1, (grads1.gradients[1], ), agg_kernel=agg_kernel_yz)
    weights11 = SuppressWeights(weights11, kernel_size=kernel_size, stride=stride)
    weights12 = GradSampleWeights(patches1, (grads1.gradients[2], ), agg_kernel=agg_kernel_yz)
    weights12 = SuppressWeights(weights12, kernel_size=kernel_size, stride=stride)

    weights20 = GradSampleWeights(patches2, (grads1.gradients[0], ), agg_kernel=agg_kernel_xz)
    weights20 = SuppressWeights(weights20, kernel_size=kernel_size, stride=stride)
    weights22 = GradSampleWeights(patches2, (grads1.gradients[2], ), agg_kernel=agg_kernel_xz)
    weights22 = SuppressWeights(weights22, kernel_size=kernel_size, stride=stride)
    print('Time', time.time() - start_time)

    weights0.save_figures(Path(dirname, 'weights', '0'), d3=False)
    weights11.save_figures(Path(dirname, 'weights', '11'), d3=False)
    weights12.save_figures(Path(dirname, 'weights', '12'), d3=False)
    weights20.save_figures(Path(dirname, 'weights', '20'), d3=False)
    weights22.save_figures(Path(dirname, 'weights', '22'), d3=False)

    print('Sample')
    start_time = time.time()
    sampler0 = Sampler(patches0, weights0.weights_flat, weights0.weights_mapping)
    sampler11 = Sampler(patches1, weights11.weights_flat, weights11.weights_mapping)
    sampler12 = Sampler(patches1, weights12.weights_flat, weights12.weights_mapping)
    sampler20 = Sampler(patches2, weights20.weights_flat, weights20.weights_mapping)
    sampler22 = Sampler(patches2, weights22.weights_flat, weights22.weights_mapping)
    sampler30 = Sampler(patches3, weights20.weights_flat, weights20.weights_mapping)
    sampler32 = Sampler(patches3, weights22.weights_flat, weights22.weights_mapping)
    sampler = SamplerCollection(sampler0, sampler11, sampler12, sampler20,
                                sampler22, sampler30, sampler32)

    indices = sampler.sample_indices(50)
    batch = sampler.get_patches(indices)
    assert list(batch.data.shape) == [50, 1, 128, 128]
    print('Time', time.time() - start_time)

    print('Save')
    Path(dirname, 'patches').mkdir(exist_ok=True)
    for i in range(batch.data.shape[0]):
        data = batch.data[i, 0, ...]
        name = batch.name[i]
        plt.imsave(Path(dirname, 'patches', name + '.jpg'), data.squeeze(), cmap='gray')

    print('successful')


if __name__ == '__main__':
    test_patches()
