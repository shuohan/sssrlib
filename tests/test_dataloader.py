#!/usr/bin/env python

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sssrlib.patches import Patches


def test_dataloader():
    image = np.random.rand(256, 256, 32)
    patch_size = (64, 64, 1)
    patches = Patches(image, patch_size)
    weights = np.ones(len(patches))
    num_samples = 32
    sampler = WeightedRandomSampler(weights, num_samples)
    loader = DataLoader(patches, batch_size=num_samples, sampler=sampler)
    assert len(loader) == 1
    for data in loader:
        assert data.shape == (32, 1, 64, 64)

    weights = np.zeros(len(patches))
    weights[0] = 1
    sampler = WeightedRandomSampler(weights, num_samples)
    loader = DataLoader(patches, batch_size=num_samples, sampler=sampler)
    assert len(loader) == 1
    for data in loader:
        assert np.array_equal(data, np.tile(patches[0], [32, 1, 1, 1]))


if __name__ == '__main__':
    test_dataloader() 
