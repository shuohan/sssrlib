#!/usr/bin/env python

import os
import torch
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path

from sssrlib.patches import Patches
from sssrlib.sample import GradSampleWeights, Sampler


def test_agg_sample():
    image = np.load('shepp3d.npy')
    image = zoom(image, (1, 1, 0.4))
    patches = Patches((32, 1, 10), image, named=False, voxel_size=(1, 1, 2.5))

    dirname = 'results_agg_sample/avg'
    os.system('rm -rf %s' % dirname)
    sample_weights = GradSampleWeights(patches)
    sample_weights.save_figures(dirname)

    sampler = Sampler(patches, sample_weights.weights_flat)
    indices = sampler.sample_indices(3)
    batch = sampler.get_patches(indices)
    for i in range(batch.shape[0]):
        fig = plt.figure()
        plt.imshow(batch[i, ...].squeeze())
        fig.savefig(Path(dirname, '%s.png' % i))

    shape = patches.patch_size
    agg_kernel = np.zeros(shape)
    agg_kernel[(shape[0] - 1) // 2, (shape[1] - 1) // 2, (shape[2] - 1) // 2] = 1
    sample_weights = GradSampleWeights(patches, 3, agg_kernel=agg_kernel)

    dirname = 'results_agg_sample/delta'
    os.system('rm -rf %s' % dirname)
    sample_weights.save_figures(dirname)

    sample_weights2 = GradSampleWeights(patches, 3)
    assert torch.allclose(sample_weights.weights_flat, sample_weights2.weights_flat)

    print('successful')

if __name__ == '__main__':
    test_agg_sample()
