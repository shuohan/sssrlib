#!/usr/bin/env python

import numpy as np
import torch
from torch.nn.functional import interpolate

from sssrlib.patches import Patches
from sssrlib.transform import Rot90, Flip, create_rot_flip


def test_patches():
    image_shape = (100, 90, 121)
    ps3d = (70, 81, 30)
    ps2d = (60, 45, 1)
    ps1d = (50, 1, 1)
    image = np.random.rand(*image_shape).astype(np.float32)
    patches_3d = Patches(image, ps3d, squeeze=False, expand_channel_dim=False)
    patch = patches_3d[101]
    assert np.array_equal(patch, image[:ps3d[0], 1:1+ps3d[1], 9:9+ps3d[2]])
    patch = patches_3d[1314]
    assert np.array_equal(patch, image[1:1+ps3d[0], 4:4+ps3d[1], 26:26+ps3d[2]])
    assert len(patches_3d) == 31 * 10 * 92

    patches_2d = Patches(image, ps2d, squeeze=False, expand_channel_dim=False)
    patch = patches_2d[13140]
    assert np.array_equal(patch, image[2:2+ps2d[0], 16:16+ps2d[1], 72:72+ps2d[2]])
    assert len(patches_2d) == 41 * 46 * 121

    patches_2d = Patches(image, ps2d, squeeze=True, expand_channel_dim=False)
    patch = patches_2d[13140]
    assert np.array_equal(patch, image[2:2+ps2d[0], 16:16+ps2d[1], 72])
    assert len(patches_2d) == 41 * 46 * 121

    patches_2d = Patches(image, ps2d, squeeze=True, expand_channel_dim=True)
    patch = patches_2d[13140]
    assert np.array_equal(patch, image[2:2+ps2d[0], 16:16+ps2d[1], 72][None, ...])
    assert len(patches_2d) == 41 * 46 * 121

    patches_1d = Patches(image, ps1d, squeeze=False, expand_channel_dim=False)
    patch = patches_1d[131400]
    assert np.array_equal(patch, image[12:12+ps1d[0], 5:5+ps1d[1], 115:115+ps1d[2]])
    assert len(patches_1d) == 51 * 90 * 121

    patches_1d = Patches(image, ps1d, squeeze=True, expand_channel_dim=True)
    patch = patches_1d[131400]
    assert np.array_equal(patch, image[12:12+ps1d[0], 5, 115][None, ...])
    assert len(patches_1d) == 51 * 90 * 121

    # permute
    image_trans = np.transpose(image, [1, 2, 0])
    patches_3d = Patches(image, ps3d, x=1, y=2, z=0, squeeze=False,
                         expand_channel_dim=False)
    patch = patches_3d[13140]
    assert len(patches_3d) == 21 * 41 * 71
    assert np.array_equal(patch, image_trans[4:4+ps3d[0], 21:21+ps3d[1], 5:5+ps3d[2]])

    # interp
    scale_factor = 4.3
    ref_image = interpolate(torch.tensor(image)[None, None, ...],
                            scale_factor=(scale_factor, 1, 1), mode='trilinear')
    print(ref_image.shape)
    patches_3d = Patches(image, ps3d, scale_factor=scale_factor,
                         mode='linear', squeeze=False, expand_channel_dim=False)
    print(patches_3d.image.shape)
    assert torch.allclose(ref_image.squeeze(), patches_3d.image)

    # same size
    patches_2d = Patches(image, 64, squeeze=False, expand_channel_dim=False)
    patch = patches_2d[13140]
    assert len(patches_2d) == 37 * 27 * 121
    assert np.array_equal(patch, image[4:68, :64, 72:73])

    # augment
    transforms = create_rot_flip()
    patches_2d = Patches(image, ps2d, transforms=transforms, squeeze=False,
                         expand_channel_dim=False)
    patch = patches_2d[1314000]
    patch = np.flip(np.rot90(patch, k=2, axes=(0, 1)), axis=0)
    assert np.array_equal(patch, image[31:31+ps2d[0], 3:3+ps2d[1], 61:61+ps2d[2]])
    assert len(patches_2d) == 8 * 41 * 46 * 121

    # together
    patches_2d = Patches(image, 64, x=2, y=0, z=1, transforms=transforms,
                         squeeze=False, expand_channel_dim=False)
    image_trans = np.transpose(image, [2, 0, 1])
    assert np.array_equal(image_trans, patches_2d.image)
    patch = patches_2d[1314000]
    patch = np.rot90(patch, k=1, axes=(0, 1))
    assert np.array_equal(patch, image_trans[46:110, 22:86, 0:1])
    assert len(patches_2d) == 8 * 58 * 37 * 90

    # corner cases
    image_shape = (4, 2, 3)
    ps = (1, 1, 1)
    image = np.arange(24).reshape(image_shape)
    patches = Patches(image, ps, squeeze=False, expand_channel_dim=False)
    patch = patches[10]
    assert np.array_equal(patch, [[[10]]])
    assert len(patches) == 24

    ps = (4, 2, 3)
    patches = Patches(image, ps, squeeze=False, expand_channel_dim=False)
    patch = patches[0]
    assert np.array_equal(patch, image)
    assert len(patches) == 1

    ps = (4, 2, 1)
    patches = Patches(image, ps, squeeze=False, expand_channel_dim=False)
    patch = patches[1]
    assert np.array_equal(patch, image[:, :, 1:2])
    assert len(patches) == 3

    ps = (4, 1, 1)
    patches = Patches(image, ps, squeeze=False, expand_channel_dim=False)
    patch = patches[5]
    assert np.array_equal(patch, image[:, 1:2, 2:3])

    ps = (3, 2, 3)
    patches = Patches(image, ps, squeeze=False, expand_channel_dim=False)
    patch = patches[1]
    assert np.array_equal(patch, image[1:, :, :])
    assert len(patches) == 2


if __name__ == '__main__':
    test_patches()
