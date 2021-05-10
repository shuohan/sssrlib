#!/usr/bin/env python

import os
import torch
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path

from sssrlib.patches import Patches
from sssrlib.sample import Aggregate, SampleWeights, Sampler, ImageGradients
from sssrlib.utils import calc_avg_kernel


def test_agg_sample():
    dirname = 'results_agg_sample/avg'
    os.system('rm -rf %s' % dirname)

    image = np.load('shepp3d.npy')
    image = zoom(image, (1, 1, 0.4))
    patches = Patches((32, 1, 10), image, named=False, voxel_size=(1, 1, 2.5))
    grads = ImageGradients(patches, sigma=3)
    grads.save_figures(dirname, d3=False)

    kernel_size = patches.patch_size
    agg_size = [int(np.ceil(ks / 2)) for ks in kernel_size]
    agg_kernel = calc_avg_kernel(agg_size, kernel_size)
    agg = Aggregate(agg_kernel, grads.gradients)
    agg.save_figures(dirname, d3=False)

    sample_weights = SampleWeights(patches, agg.agg_images)
    sample_weights.save_figures(dirname, d3=False)

    sampler = Sampler(patches, sample_weights.weights_flat)
    indices = sampler.sample_indices(3)
    batch = sampler.get_patches(indices)
    for i in range(batch.shape[0]):
        fig = plt.figure()
        plt.imshow(batch[i, ...].squeeze(), cmap='gray')
        fig.savefig(Path(dirname, '%s.png' % i))

    dirname = 'results_agg_sample/delta'
    os.system('rm -rf %s' % dirname)

    shape = patches.patch_size
    agg_kernel = np.zeros(shape)
    agg_kernel[(shape[0] - 1) // 2, (shape[1] - 1) // 2, (shape[2] - 1) // 2] = 1
    agg = Aggregate(agg_kernel, grads.gradients)
    agg.save_figures(dirname, d3=False)

    sample_weights1 = SampleWeights(patches, grads.gradients)
    sample_weights2 = SampleWeights(patches, agg.agg_images)
    sample_weights2.save_figures(dirname)

    assert torch.allclose(sample_weights1.weights_flat, sample_weights2.weights_flat)

    print('successful')

if __name__ == '__main__':
    test_agg_sample()
