#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sssrlib.patches import Patches


def test_dataloader():
    image = np.random.rand(256, 256, 32)
    patch_size = (64, 64, 1)
    patches = Patches(image, patch_size)
    batch_size = 32
    loader = patches.get_dataloader(batch_size)
    assert len(loader) == 1
    for data in loader:
        assert data.shape == (32, 1, 64, 64)

    weights = torch.zeros(len(patches))
    weights[100] = 1
    loader.sampler.weights = weights
    assert len(loader) == 1
    for data in loader:
        assert np.array_equal(data, np.tile(patches[100], [32, 1, 1, 1]))

    print('successful.')


if __name__ == '__main__':
    test_dataloader() 
