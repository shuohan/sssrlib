#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sssrlib.patches import Patches
from pathlib import Path
import matplotlib.pyplot as plt


def test_dataloader():
    dirname = Path('results_dataloader')
    dirname.mkdir(exist_ok=True)

    image = np.load('shepp3d.npy')
    patch_size = (64, 64, 1)
    patches = Patches(image, patch_size, named=False)
    weights = patches.get_sample_weights()
    indices = torch.argsort(weights, descending=True)[:50]

    for i, ind in enumerate(indices):
        patch = patches[ind].cpu().numpy().squeeze()
        filename = dirname.joinpath('patch-%d.png' % i)
        plt.imsave(filename, patch, cmap='gray')

    # batch_size = 32
    # loader = patches.get_dataloader(batch_size)
    # assert len(loader) == 1
    # for data in loader:
    #     assert data.shape == (32, 1, 64, 64)

    # weights = torch.zeros(len(patches))
    # weights[100] = 1
    # loader.sampler.weights = weights
    # assert len(loader) == 1
    # for data in loader:
    #     assert np.array_equal(data, np.tile(patches[100], [32, 1, 1, 1]))

    # print('successful.')


if __name__ == '__main__':
    test_dataloader() 
