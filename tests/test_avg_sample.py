#!/usr/bin/env python

import os
import torch
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path

from sssrlib.patches import Patches
from sssrlib.sample import AvgGradSample, GradSample


def test_avg_sample():
    image = np.load('shepp3d.npy')
    image = zoom(image, (1, 1, 0.4))
    patches = Patches((32, 1, 10), image, named=False, voxel_size=(1, 1, 2.5))

    dirname = 'results_avg_sample/avg'
    os.system('rm -rf %s' % dirname)
    sample = AvgGradSample(patches)
    sample.save_figures(dirname)
    indices = sample.sample_indices(3)
    batch = sample.get_patches(indices)
    for i in range(batch.shape[0]):
        fig = plt.figure()
        plt.imshow(batch[i, ...].squeeze())
        fig.savefig(Path(dirname, '%s.png' % i))

    shape = patches.patch_size
    avg_kernel = np.zeros(shape)
    avg_kernel[(shape[0] - 1) // 2, (shape[1] - 1) // 2, (shape[2] - 1) // 2] = 1
    sample = AvgGradSample(patches, 3, avg_kernel=avg_kernel)

    dirname = 'results_avg_sample/delta'
    os.system('rm -rf %s' % dirname)
    sample.save_figures(dirname)
    sample.calc_sample_weights()

    sample2 = GradSample(patches, 3)
    sample2.calc_sample_weights()
    assert torch.allclose(sample._weights_flat, sample2._weights_flat)

    print('successful')

if __name__ == '__main__':
    test_avg_sample()
