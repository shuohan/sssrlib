#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sssrlib.patches import Patches
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib


def test_dataloader():
    dirname = Path('results_dataloader')
    dirname.mkdir(exist_ok=True)

    filename = '/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-4p0_scale-0p5_len-13.nii'
    image = nib.load(filename).get_fdata(dtype=np.float32)
    patch_size = (64, 1, 32)
    # patches = Patches(image, patch_size, named=False)
    # weights = patches.get_sample_weights()
    # indices = torch.argsort(weights, descending=True)[:50]

    # for i, ind in enumerate(indices):
    #     patch = patches[ind].cpu().numpy().squeeze()
    #     filename = dirname.joinpath('patch-%d.png' % i)
    #     plt.imsave(filename, patch, cmap='gray')

    weight_stride = (17, 17, 5)
    patches = Patches(image, patch_size, sigma=2,
                      named=False, avg_grad=False,
                      weight_stride=weight_stride).cuda()
    weights = patches.get_sample_weights()
    # indices = torch.argsort(weights, descending=True)[:500000:10000]

    indices = torch.multinomial(weights, 100, replacement=True)
    print(indices)

    for i, ind in enumerate(indices):
        patch = patches[int(ind)].cpu().numpy().squeeze()
        filename = dirname.joinpath('avg_patch-%d.png' % i)
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
