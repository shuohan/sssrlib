#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sssrlib.patches import Patches


def test_grad():
    dirname = Path('results_grad')
    dirname.mkdir(exist_ok=True)

    image = np.load('shepp3d.npy')
    patch_size = [64, 64, 1]
    patches = Patches(image, patch_size).cuda()
    kernel = patches._get_gaussian_kernel().cpu().numpy().squeeze()
    sum = np.sum(kernel)
    assert np.isclose(sum, 1)
    grad = patches._calc_image_grad().cpu().numpy()

    plt.figure()
    plt.imshow(kernel)
    plt.gcf().savefig(dirname.joinpath('kernel.png'))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(grad[:, :, grad.shape[2]//2], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(image[:, :, grad.shape[2]//2], cmap='gray')
    plt.gcf().savefig(dirname.joinpath('grad.png'))

    weights = patches._get_sample_weights()
    assert len(weights) == len(patches) / len(patches.transforms)
    shape = [s - ps + 1 for s, ps in zip(image.shape, patch_size)]

    for i in range(100):
        ind = int(np.random.uniform(0, len(weights)))
        x, y, z = np.unravel_index(ind, [65, 65, 128])
        assert weights[ind] == grad[x + (patch_size[0] - 1)//2,
                                    y + (patch_size[1] - 1)//2,
                                    z + (patch_size[2] - 1)//2]

    print('successful')

if __name__ == '__main__':
    test_grad()
