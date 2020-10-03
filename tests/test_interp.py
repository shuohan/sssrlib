#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate

from sssrlib.patches import Patches


def test_interp():
    image = np.load('shepp3d.npy')
    scale_factor = 2.1
    mode = 'linear'
    slice_ind = 64
    patches_2d = Patches(image, 64, scale_factor=scale_factor, mode=mode,
                         squeeze=False, expand_channel_dim=False)

    plt.subplot(1, 2, 1)
    plt.imshow(patches_2d.image.numpy()[:, :, slice_ind], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(image[:, :, slice_ind], cmap='gray')
    plt.gcf().savefig('interp.png')

    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            image_line = image[:, i, j][None, None, ...]
            ref_line = interpolate(torch.tensor(image_line),
                                   scale_factor=scale_factor, mode=mode)
            interp_line = patches_2d.image[:, i, j]
            assert torch.equal(ref_line.squeeze(), interp_line)
    print('successful')

    # patch = patches_2d[13140]
    # assert len(patches_2d) == 37 * 27 * 121
    # assert np.array_equal(patch, image[4:68, :64, 72:73])


if __name__ == '__main__':
    test_interp()
