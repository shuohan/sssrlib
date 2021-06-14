#!/usr/bin/env python

import os
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from improc3d import transform_to_axial

from sssrlib.patches import Patches
from sssrlib.sample import Sampler, ImageGradients
from sssrlib.sample import SuppressWeights, SampleWeights, Aggregate
from sssrlib.utils import calc_foreground_mask, calc_avg_kernel, save_fig_2d


def test_headmask():
    ps2d = (1, 128, 128)
    kernel_size = (1, 4, 4)
    stride = (1, 2, 2)
    fn = '/data/smore_simu/orig_data/sub-OAS30108_ses-d0168_T2w_initnorm.nii.gz'
    dirname = Path('results_headmask')
    os.system('rm -rf %s' % dirname)
    dirname.mkdir(exist_ok=True)

    obj = nib.load(fn)
    image = obj.get_fdata(dtype=np.float32)
    image = transform_to_axial(image, obj.affine, coarse=True).copy()

    patches = Patches(ps2d, image, x=2, y=1, z=0)
    mask = calc_foreground_mask(patches.image)
    patches.save_figures(Path(dirname, 'images'), d3=False)
    save_fig_2d(Path(dirname, 'images'), mask, 'mask', cmap='gray')

    agg_kernel = calc_avg_kernel(ps2d)
    agg = Aggregate(agg_kernel, (mask, ))
    weights = SampleWeights(patches, agg.agg_images)
    weights = SuppressWeights(weights, kernel_size=kernel_size, stride=stride)
    sampler = Sampler(patches, weights.weights_flat, weights.weights_mapping)

    agg.save_figures(Path(dirname, 'agg'), d3=False)
    weights.save_figures(Path(dirname, 'weights'), d3=False)

    indices = sampler.sample_indices(50)
    batch = sampler.get_patches(indices)
    assert list(batch.data.shape) == [50, 1, 128, 128]

    Path(dirname, 'patches').mkdir(exist_ok=True)
    for i in range(batch.data.shape[0]):
        data = batch.data[i, 0, ...]
        name = batch.name[i]
        plt.imsave(Path(dirname, 'patches', name + '.jpg'), data.squeeze().T, cmap='gray')

    print('successful')

    sampler = Sampler(patches)


if __name__ == '__main__':
    test_headmask()
