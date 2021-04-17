#!/usr/bin/env python

import os
import nibabel as nib
import torch
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path

from sssrlib.patches import Patches
from sssrlib.sample import GradSampleWeights, SuppressWeights, Sampler


def test_sup_sample():
    image = np.load('shepp3d.npy')
    image = zoom(image, (1, 1, 0.4))
    patches = Patches((32, 1, 10), image, named=False, voxel_size=(1, 1, 2.5))

    dirname = 'results_sup_sample'
    os.system('rm -rf %s' % dirname)
    weights = GradSampleWeights(patches)
    weights = SuppressWeights(weights, stride=[16, 16, 1], kernel_size=[32, 32, 1])
    weights.save_figures(dirname)

    sampler = Sampler(patches, weights.weights_flat, weights.weights_mapping)
    indices = sampler.sample_indices(3)
    batch = sampler.get_patches(indices)
    for i in range(batch.shape[0]):
        fig = plt.figure()
        plt.imshow(batch[i, ...].squeeze())
        fig.savefig(Path(dirname, '%s.png' % i))

    assert torch.equal(weights.weights_flat,
                       weights.sample_weights.weights_flat[weights.weights_mapping])

    print('successful')

if __name__ == '__main__':
    test_sup_sample()
