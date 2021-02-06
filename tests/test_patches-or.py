#!/usr/bin/env python

import numpy as np
import torch
import time
from torch.nn.functional import interpolate

from sssrlib.patches import Patches, PatchesOr, NamedData
from sssrlib.transform import Rot90, Flip, create_rot_flip


def test_patches():
    image_shape = (100, 90, 121)
    ps2d = (60, 45, 1)
    image = np.random.rand(*image_shape).astype(np.float32)

    patches0 = Patches(image, ps2d, x=2, y=1, z=0)
    patches1 = Patches(image, ps2d, x=2, y=0, z=1)
    patches2 = Patches(image, ps2d, x=1, y=2, z=0)
    patches = PatchesOr(patches0, patches1, patches2)
    
    assert len(patches) == len(patches0) + len(patches1) + len(patches2)

    assert np.array_equal(patches[len(patches0)+len(patches1)].data, patches2[0].data)
    names = patches2[0].name.split('-')
    names.insert(1, 'p2')
    assert patches[len(patches0)+len(patches1)].name == '-'.join(names)
    names = patches1[len(patches1)-1].name.split('-')
    names.insert(1, 'p1')
    assert np.array_equal(patches[len(patches0)+len(patches1)-1].data,
                          patches1[len(patches1)-1].data)

    assert np.array_equal(patches[len(patches0)].data, patches1[0].data)
    names = patches1[0].name.split('-')
    names.insert(1, 'p1')
    assert patches[len(patches0)].name == '-'.join(names)
    names = patches0[len(patches0) - 1].name.split('-')
    names.insert(1, 'p0')
    assert np.array_equal(patches[len(patches0) - 1].data, patches0[len(patches0) - 1].data)


    dataloader0 = patches0.get_dataloader(32)
    dataloader1 = patches1.get_dataloader(32)
    dataloader2 = patches2.get_dataloader(32)

    dataloader = patches.get_dataloader(32)
    weights = dataloader.sampler.weights
    ref_weights = torch.cat([dataloader0.sampler.weights,
                             dataloader1.sampler.weights,
                             dataloader2.sampler.weights])
    assert torch.equal(weights, ref_weights)
    assert len(dataloader) == 1

    print('successful')


if __name__ == '__main__':
    test_patches()
