#!/usr/bin/env python

import numpy as np
import torch
import time
from torch.nn.functional import interpolate

from sssrlib.patches import Patches, NamedData, TransformedPatches
from sssrlib.transform import Rot90, Flip, Compose


def test_patches():
    image_shape = (100, 90, 121)
    ps3d = (70, 81, 30)
    ps2d = (60, 45, 1)
    ps1d = (50, 1, 1)
    image = np.random.rand(*image_shape).astype(np.float32)
    patches_3d = Patches(ps3d, image, squeeze=False, expand_channel_dim=False, named=False)
    assert np.array_equal(patches_3d[101], image[:ps3d[0], 1:1+ps3d[1], 9:9+ps3d[2]])
    assert np.array_equal(patches_3d[1314], image[1:1+ps3d[0], 4:4+ps3d[1], 26:26+ps3d[2]])
    assert len(patches_3d) == 31 * 10 * 92

    patches_2d = Patches(ps2d, image, squeeze=False, expand_channel_dim=False, named=False)
    assert np.array_equal(patches_2d[13140], image[2:2+ps2d[0], 16:16+ps2d[1], 72:72+ps2d[2]])
    assert len(patches_2d) == 41 * 46 * 121

    patches_2d = Patches(ps2d, image, squeeze=True, expand_channel_dim=False, named=False)
    assert np.array_equal(patches_2d[13140], image[2:2+ps2d[0], 16:16+ps2d[1], 72])
    assert len(patches_2d) == 41 * 46 * 121

    patches_2d = Patches(ps2d, image, squeeze=True, expand_channel_dim=True, named=False)
    assert np.array_equal(patches_2d[13140], image[2:2+ps2d[0], 16:16+ps2d[1], 72][None, ...])
    assert len(patches_2d) == 41 * 46 * 121

    patches_1d = Patches(ps1d, image, squeeze=False, expand_channel_dim=False, named=False)
    assert np.array_equal(patches_1d[131400], image[12:12+ps1d[0], 5:5+ps1d[1], 115:115+ps1d[2]])
    assert len(patches_1d) == 51 * 90 * 121

    patches_1d = Patches(ps1d, image, squeeze=True, expand_channel_dim=True, named=False)
    assert np.array_equal(patches_1d[131400], image[12:12+ps1d[0], 5, 115][None, ...])
    assert len(patches_1d) == 51 * 90 * 121

    # named
    patches_2d = Patches(ps2d, image, squeeze=True, expand_channel_dim=True, named=True)
    patch = patches_2d[13140]
    assert isinstance(patch, NamedData)
    assert patch.name == 'x02-y16-z072'
    assert np.array_equal(patch.data, image[2:2+ps2d[0], 16:16+ps2d[1], 72][None, ...])
    assert len(patches_2d) == 41 * 46 * 121

    # permute
    image_trans = np.transpose(image, [1, 2, 0])
    patches_3d = Patches(ps3d, image, x=1, y=2, z=0, squeeze=False,
                         expand_channel_dim=False, named=False)
    assert len(patches_3d) == 21 * 41 * 71
    assert np.array_equal(patches_3d[13140], image_trans[4:4+ps3d[0], 21:21+ps3d[1], 5:5+ps3d[2]])

    # same size
    patches_2d = Patches(64, image, squeeze=False, expand_channel_dim=False, named=False)
    assert len(patches_2d) == 37 * 27 * 121
    assert np.array_equal(patches_2d[13140], image[4:68, :64, 72:73])

    # augment
    transform = Compose(Flip(), Rot90(k=2))
    patches_2d = Patches(ps2d, image, squeeze=False, expand_channel_dim=False, named=False)
    patches_2d = TransformedPatches(patches_2d, transform)
    patch = patches_2d[172970]
    patch = np.flip(np.rot90(patch, k=2, axes=(0, 1)), axis=0)
    assert np.array_equal(patch, image[31:31+ps2d[0], 3:3+ps2d[1], 61:61+ps2d[2]])
    assert len(patches_2d) == 41 * 46 * 121

    # together
    patches_2d = Patches(64, image, x=2, y=0, z=1, named=True, squeeze=True, expand_channel_dim=True)
    transform = Compose(Rot90(k=1), Flip())
    patches_2d = TransformedPatches(patches_2d, transform)
    patches_2d.cuda()
    image_trans = np.transpose(image, [2, 0, 1])
    patch = patches_2d[155160]
    patch_data = torch.rot90(torch.flip(patch.data, [0]), 3)[None, ...]
    ref_patch = image_trans[46:46+64, 22:22+64, 0]
    ref_patch = torch.tensor(ref_patch, dtype=torch.float32).cuda()
    assert patch.name == 'x46-y22-z00_rot90-flip'
    assert len(patches_2d) == 58 * 37 * 90
    assert torch.allclose(patch_data, ref_patch)

    # # create from patches
    # patches_p1 = Patches(64, patches=patches_2d,  transforms=transforms,
    #                     named=True, squeeze=True, expand_channel_dim=True)
    # patch1 = patches_2d[1314000]
    # patch2 = patches_p1[1314000]
    # assert torch.equal(patch1.data, patch2.data)

    # patches_p2 = Patches((64, 1, 16), patches=patches_2d, transforms=[],
    #                     named=False, squeeze=False, expand_channel_dim=False)
    # assert len(patches_p2) == len(patches_p1) // 8
    # assert patches_p2[0].data.shape == (64, 1, 16)

    print('successful')


if __name__ == '__main__':
    test_patches()
