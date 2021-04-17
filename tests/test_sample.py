#!/usr/bin/env python

import os
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path

from sssrlib.patches import Patches
from sssrlib.sample import GradSampleWeights, Sampler


def test_sample():
    image = np.load('shepp3d.npy')
    image = zoom(image, (1, 1, 0.4))
    patches = Patches((32, 1, 10), image, voxel_size=(1, 1, 2.5), named=False)

    dirname = 'results_sample/and'
    os.system('rm -rf %s' % dirname)

    sample_weights = GradSampleWeights(patches, weights_op='and')
    sample_weights.save_figures(dirname)

    sampler = Sampler(patches, sample_weights.weights_flat)
    indices = sampler.sample_indices(3)
    batch = sampler.get_patches(indices)

    for i in range(batch.shape[0]):
        fig = plt.figure()
        plt.imshow(batch[i, ...].squeeze())
        fig.savefig(Path(dirname, '%s.png' % i))

    assert len(sampler.weights_flat) == len(patches)

    grads = [sample_weights._gradients[0], sample_weights._gradients[2]]
    starts = [(ps - 1) // 2 for ps in patches.patch_size]
    sum0 = grads[0][starts[0] : starts[0]+patches.xnum,
                    starts[1] : starts[1]+patches.ynum,
                    starts[2] : starts[2]+patches.znum].sum()
    sum1 = grads[1][starts[0] : starts[0]+patches.xnum,
                    starts[1] : starts[1]+patches.ynum,
                    starts[2] : starts[2]+patches.znum].sum()
    grads[0] = grads[0] / sum0
    grads[1] = grads[1] / sum1
    grad = grads[0] * grads[1]
    for i in range(100):
        ind = int(np.random.uniform(0, len(sampler.weights_flat)))
        x, y, z = np.unravel_index(ind, [patches.xnum, patches.ynum, patches.znum])
        assert sampler.weights_flat[ind] == grad[x + (patches.patch_size[0] - 1)//2,
                                                 y + (patches.patch_size[1] - 1)//2,
                                                 z + (patches.patch_size[2] - 1)//2]

    sample = GradSampleWeights(patches, sigma=3, weights_op='or')
    dirname = 'results_sample/or'
    os.system('rm -rf %s' % dirname)
    sample.save_figures(dirname)

    image = zoom(image, (1, 1, 0.4))
    patches = Patches((32, 32, 1), image, named=False, voxel_size=(1, 1, 2.5))

    dirname = 'results_sample/x_grad'
    os.system('rm -rf %s' % dirname)
    sample_weights = GradSampleWeights(patches, use_grads=[True, False, False])
    sample_weights.save_figures(dirname)
    sampler = Sampler(patches, sample_weights.weights_flat)
    indices = sampler.sample_indices(3)

    for i in range(batch.shape[0]):
        fig = plt.figure()
        plt.imshow(batch[i, ...].squeeze())
        fig.savefig(Path(dirname, '%s.png' % i))

    print('successful')

if __name__ == '__main__':
    test_sample()
