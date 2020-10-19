#!/usr/bin/env python

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sssrlib.patches import Patches


def test_xz():
    orig_image = np.load('shepp3d.npy')
    image = orig_image[:, :, ::4]
    patch_size = (64, 1, 64)
    patches = Patches(image, patch_size, named=False, scale_factor=(1, 1, 4))
    patch = patches[202135]
    print(len(patches))
    print(patches.image.shape)
    print(patch.shape)
    assert torch.equal(patch.squeeze(), patches.image[24:24+64, 37, 50:50+64])

    dirname = Path('results_xz')
    dirname.mkdir(exist_ok=True)
    plt.subplot(1, 2, 1)
    plt.imshow(patch.squeeze(), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(orig_image[24:24+64, 37, 50:50+64], cmap='gray')
    plt.gcf().savefig(dirname.joinpath('patch.png'))


if __name__ == '__main__':
    test_xz()
