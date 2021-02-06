#!/usr/bin/env python

import numpy as np
import torch
import time
from torch.nn.functional import interpolate

from sssrlib.patches import Patches, NamedData
from sssrlib.transform import Rot90, Flip, create_rot_flip


def test_patches():
    image_shape = (100, 90, 121)
    ps3d = (70, 81, 30)
    ps2d = (60, 45, 1)
    ps1d = (50, 1, 1)
    image = np.random.rand(*image_shape).astype(np.float32)
    patches_3d = Patches(image, ps3d, squeeze=False, expand_channel_dim=False, named=False)
    patch = patches_3d[101]
    assert np.array_equal(patch, image[:ps3d[0], 1:1+ps3d[1], 9:9+ps3d[2]])
    patch = patches_3d[1314]
    assert np.array_equal(patch, image[1:1+ps3d[0], 4:4+ps3d[1], 26:26+ps3d[2]])
    assert len(patches_3d) == 31 * 10 * 92

    patches_2d = Patches(image, ps2d, squeeze=False, expand_channel_dim=False, named=False)
    patch = patches_2d[13140]
    assert np.array_equal(patch, image[2:2+ps2d[0], 16:16+ps2d[1], 72:72+ps2d[2]])
    assert len(patches_2d) == 41 * 46 * 121

    patches_2d = Patches(image, ps2d, squeeze=True, expand_channel_dim=False, named=False)
    patch = patches_2d[13140]
    assert np.array_equal(patch, image[2:2+ps2d[0], 16:16+ps2d[1], 72])
    assert len(patches_2d) == 41 * 46 * 121

    patches_2d = Patches(image, ps2d, squeeze=True, expand_channel_dim=True, named=False)
    patch = patches_2d[13140]
    assert np.array_equal(patch, image[2:2+ps2d[0], 16:16+ps2d[1], 72][None, ...])
    assert len(patches_2d) == 41 * 46 * 121

    patches_1d = Patches(image, ps1d, squeeze=False, expand_channel_dim=False, named=False)
    patch = patches_1d[131400]
    assert np.array_equal(patch, image[12:12+ps1d[0], 5:5+ps1d[1], 115:115+ps1d[2]])
    assert len(patches_1d) == 51 * 90 * 121

    patches_1d = Patches(image, ps1d, squeeze=True, expand_channel_dim=True, named=False)
    patch = patches_1d[131400]
    assert np.array_equal(patch, image[12:12+ps1d[0], 5, 115][None, ...])
    assert len(patches_1d) == 51 * 90 * 121

    # named
    patches_2d = Patches(image, ps2d, squeeze=True, expand_channel_dim=True, named=True)
    patch = patches_2d[13140]
    assert isinstance(patch, NamedData)
    assert patch.name == 'ind-02-16-072'
    assert np.array_equal(patch.data, image[2:2+ps2d[0], 16:16+ps2d[1], 72][None, ...])
    assert len(patches_2d) == 41 * 46 * 121

    # permute
    image_trans = np.transpose(image, [1, 2, 0])
    patches_3d = Patches(image, ps3d, x=1, y=2, z=0, squeeze=False,
                         expand_channel_dim=False, named=False)
    patch = patches_3d[13140]
    assert len(patches_3d) == 21 * 41 * 71
    assert np.array_equal(patch, image_trans[4:4+ps3d[0], 21:21+ps3d[1], 5:5+ps3d[2]])

    # interp
    scale_factor = 4.3

    start = time.time()
    ref_image = interpolate(torch.tensor(image)[None, None, ...],
                            scale_factor=(scale_factor, 1, 1), mode='trilinear')
    ref_image = ref_image.squeeze()
    print('trilinear', time.time() - start)

    start = time.time()
    patches_3d = Patches(image, ps3d, scale_factor=scale_factor, named=False,
                         mode='linear', squeeze=False, expand_channel_dim=False)
    print('linear', time.time() - start)

    start = time.time()
    ref_image = interpolate(torch.tensor(image)[None, None, ...],
                            scale_factor=(scale_factor, 1, 1), mode='trilinear')
    ref_image = ref_image.squeeze()
    print('trilinear', time.time() - start)

    patch = patches_3d[131400]
    assert torch.allclose(ref_image, patches_3d.image)
    assert torch.allclose(patch, ref_image[142:142+ps3d[0], 8:8+ps3d[1], 24:24+ps3d[2]])

    # same size
    patches_2d = Patches(image, 64, squeeze=False, expand_channel_dim=False, named=False)
    patch = patches_2d[13140]
    assert len(patches_2d) == 37 * 27 * 121
    assert np.array_equal(patch, image[4:68, :64, 72:73])

    # augment
    transforms = create_rot_flip()
    patches_2d = Patches(image, ps2d, transforms=transforms, squeeze=False,
                         expand_channel_dim=False, named=False)
    patch = patches_2d[1314000]
    patch = np.flip(np.rot90(patch, k=2, axes=(0, 1)), axis=0)
    assert np.array_equal(patch, image[31:31+ps2d[0], 3:3+ps2d[1], 61:61+ps2d[2]])
    assert len(patches_2d) == 8 * 41 * 46 * 121

    # together
    patches_2d = Patches(image, 64, x=2, y=0, z=1, transforms=transforms,
                         scale_factor=scale_factor, mode='linear', named=True,
                         squeeze=True, expand_channel_dim=True)
    patches_2d.cuda()
    image_trans = np.transpose(image, [2, 0, 1])
    ref_image = interpolate(torch.tensor(image_trans).cuda()[None, None, ...],
                            scale_factor=(scale_factor, 1, 1), mode='trilinear')
    ref_image = ref_image.squeeze()
    patch = patches_2d[3637383]
    ref_patch = ref_image[178:178+64, 11:11+64, 33]
    ref_patch = torch.rot90(ref_patch, 1)[None, ...]
    assert patch.name == 'ind-178-11-33'
    assert len(patches_2d) == 8 * 457 * 37 * 90
    assert torch.allclose(patch.data, ref_patch)

    # corner cases
    image_shape = (4, 2, 3)
    ps = (1, 1, 1)
    image = np.arange(24).reshape(image_shape)
    patches = Patches(image, ps, squeeze=False, expand_channel_dim=False, named=False)
    patch = patches[10]
    assert np.array_equal(patch, [[[10]]])
    assert len(patches) == 24

    ps = (4, 2, 3)
    patches = Patches(image, ps, squeeze=False, expand_channel_dim=False, named=False)
    patch = patches[0]
    assert np.array_equal(patch, image)
    assert len(patches) == 1

    ps = (4, 2, 1)
    patches = Patches(image, ps, squeeze=False, expand_channel_dim=False, named=False)
    patch = patches[1]
    assert np.array_equal(patch, image[:, :, 1:2])
    assert len(patches) == 3

    ps = (4, 1, 1)
    patches = Patches(image, ps, squeeze=False, expand_channel_dim=False, named=False)
    patch = patches[5]
    assert np.array_equal(patch, image[:, 1:2, 2:3])

    ps = (3, 2, 3)
    patches = Patches(image, ps, squeeze=False, expand_channel_dim=False, named=False)
    patch = patches[1]
    assert np.array_equal(patch, image[1:, :, :])
    assert len(patches) == 2

    print('successful')


if __name__ == '__main__':
    test_patches()
