#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sssrlib.transform import Flip, Identity, create_rot_flip
from sssrlib.patches import Patches, PatchesOr
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib


def test_dataloader():
    dirname = Path('results_dataloader')
    dirname.mkdir(exist_ok=True)

    filename = '/data/oasis3/simu/sub-OAS30001_ses-d0129_acq-mprage_run-01_T1w_type-gauss_fwhm-4p0_scale-0p5_len-13.nii'
    image = nib.load(filename).get_fdata(dtype=np.float32)
    patch_size = (32, 1, 16)
    voxel_size = (1, 1, 2)
    # patches = Patches(image, patch_size, named=False)
    # weights = patches.get_sample_weights()
    # indices = torch.argsort(weights, descending=True)[:50]

    # for i, ind in enumerate(indices):
    #     patch = patches[ind].cpu().numpy().squeeze()
    #     filename = dirname.joinpath('patch-%d.png' % i)
    #     plt.imsave(filename, patch, cmap='gray')

    weight_stride = (2, 2, 1)
    transforms = [Identity(), Flip((0, )), Flip((2, )), Flip((0, 2))]
    # transforms = create_rot_flip()
    patches = Patches(patch_size, image=image, sigma=1, voxel_size=voxel_size,
                      transforms=transforms, verbose=True,
                      named=False, avg_grad=False, compress=True,
                      weight_stride=weight_stride, weight_dir=dirname).cuda()

    weights = patches.get_sample_weights()
    ind = torch.argmax(weights)
    patch = patches[int(ind)]
    orig_ind = patches._ind_mapping[int(ind)]

    ref_patches = Patches(patch_size, image=image, sigma=1,
                          voxel_size=voxel_size, transforms=transforms,
                          verbose=True, named=False, avg_grad=False,
                          compress=False, weight_stride=weight_stride).cuda()
    ref_patch = ref_patches[orig_ind]
    assert torch.equal(ref_patch, patch)

    # # indices = torch.argsort(weights, descending=True)[:500000:10000]

    # indices = torch.multinomial(weights, 100, replacement=True)
    # print(indices)

    # for i, ind in enumerate(indices):
    #     patch = patches[int(ind)].cpu().numpy().squeeze()
    #     filename = dirname.joinpath('weighted_patch-%d.png' % i)
    #     plt.imsave(filename, patch, cmap='gray')

    patches_xz = Patches(patch_size, image=image, sigma=1, voxel_size=voxel_size,
                         transforms=transforms, verbose=True,
                         named=False, avg_grad=False, compress=True,
                         weight_stride=weight_stride, weight_dir=dirname).cuda()

    patch_size_yz = (patch_size[1], patch_size[0], patch_size[2])
    patches_yz = Patches(patch_size_yz, patches=patches_xz, sigma=1,
                         voxel_size=voxel_size, transforms=transforms,
                         verbose=True, named=False, avg_grad=False,
                         compress=True, weight_stride=weight_stride,
                         weight_dir=dirname).cuda()
    patches = PatchesOr(patches_xz, patches_yz)
   
    batch_size = 100
    loader = patches.get_dataloader(batch_size, weighted=True)
    # assert len(loader) == 1
    for data in loader:
        pass
        # assert data.shape == (100, 1, 32, 16)

    for i, patch in enumerate(data.squeeze()):
        patch = patch.cpu().numpy()
        filename = dirname.joinpath('patch-%d.png' % i)
        plt.imsave(filename, patch, cmap='gray')

    print(len(loader.sampler.weights), 2 ** 24) 
# 
#     weights = torch.zeros(len(patches))
#     weights[100] = 1
#     loader.sampler.weights = weights
#     assert len(loader) == 1
#     for data in loader:
#         assert np.array_equal(data, np.tile(patches[100], [32, 1, 1, 1]))

    print('successful.')


if __name__ == '__main__':
    test_dataloader() 
