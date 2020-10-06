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
    avg_grad = patches._calc_image_grad().cpu().numpy()

    plt.figure()
    plt.imshow(kernel)
    plt.gcf().savefig(dirname.joinpath('kernel.png'))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(avg_grad[:, :, avg_grad.shape[2]//2], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(image[:, :, avg_grad.shape[2]//2], cmap='gray')
    plt.gcf().savefig(dirname.joinpath('grad.png'))


if __name__ == '__main__':
    test_grad()
