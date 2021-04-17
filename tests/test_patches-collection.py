#!/usr/bin/env python

import torch
import os
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from improc3d import transform_to_axial

from sssrlib.patches import Patches, PatchesCollection, TransformedPatches
from sssrlib.patches import NamedData
from sssrlib.sample import SamplerCollection, Sampler
from sssrlib.sample import SuppressWeights, GradSampleWeights
from sssrlib.transform import Flip
from sssrlib.utils import calc_avg_kernel, save_sample_weights_figures


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
    patches2 = Patches(ps2d_t, image, x=2, y=0, z=1)
    patches3 = Patches(ps2d_t, patches=patches2)
    patches3 = TransformedPatches(patches3, Flip((-2, )))
    patches = PatchesCollection(patches0, patches1, patches2)
    
    assert len(patches2) == len(patches3)
    assert len(patches) == len(patches0) + len(patches1) + len(patches2)
    assert torch.equal(patches[2, 20].data, patches2[20].data)
    assert patches[2, 10].name.startswith('p2')

    dirname = Path('results_patches-collection')
    os.system('rm -rf %s' % dirname)
    dirname.mkdir(exist_ok=True)

    agg_kernel_yz = calc_avg_kernel(ps2d)
    agg_kernel_xz = calc_avg_kernel(ps2d_t)
    kernel_size = (4, 4, 1)
    stride = (2, 2, 1)

    print('Create weights')
    weights0 = GradSampleWeights(patches0, sigma=3, use_grads=[False, True, False],
                                 agg_kernel=agg_kernel_yz)
    weights0 = SuppressWeights(weights0, kernel_size=kernel_size, stride=stride)

    weights11 = GradSampleWeights(patches1, sigma=3, use_grads=[False, True, False],
                                  agg_kernel=agg_kernel_yz)
    weights11 = SuppressWeights(weights11, kernel_size=kernel_size, stride=stride)
    weights12 = GradSampleWeights(patches1, sigma=3, use_grads=[False, False, True],
                                  agg_kernel=agg_kernel_yz)
    weights12 = SuppressWeights(weights12, kernel_size=kernel_size, stride=stride)

    weights20 = GradSampleWeights(patches2, sigma=3, use_grads=[True, False, False],
                                  agg_kernel=agg_kernel_xz)
    weights20 = SuppressWeights(weights20, kernel_size=kernel_size, stride=stride)
    weights22 = GradSampleWeights(patches2, sigma=3, use_grads=[False, False, True],
                                  agg_kernel=agg_kernel_xz)
    weights22 = SuppressWeights(weights22, kernel_size=kernel_size, stride=stride)

    save_sample_weights_figures([weights0, weights11, weights12, weights20,
                                 weights22], dirname.joinpath('3d'), d3=True)
    save_sample_weights_figures([weights0, weights11, weights12, weights20,
                                 weights22], dirname.joinpath('2d'), d3=False)

    print('Sample')
    sampler0 = Sampler(patches0, weights0.weights_flat)
    sampler11 = Sampler(patches1, weights11.weights_flat)
    sampler12 = Sampler(patches1, weights12.weights_flat)
    sampler20 = Sampler(patches2, weights20.weights_flat)
    sampler22 = Sampler(patches2, weights22.weights_flat)
    sampler30 = Sampler(patches3, weights20.weights_flat)
    sampler32 = Sampler(patches3, weights22.weights_flat)
    sampler = SamplerCollection(sampler0, sampler11, sampler12, sampler20,
                                sampler22, sampler30, sampler32)

    indices = sampler.sample_indices(30)
    batch = sampler.get_patches(indices)
    assert list(batch.data.shape) == [30, 1, 128, 128]

    print('Save')
    for i in range(batch.data.shape[0]):
        data = batch.data[i, 0, ...]
        name = batch.name[i]
        fig = plt.figure()
        plt.imshow(data)
        fig.savefig(Path(dirname, name + '.png'))

    print('successful')


if __name__ == '__main__':
    test_patches()
