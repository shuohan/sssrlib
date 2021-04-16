#!/usr/bin/env python

import os
import nibabel as nib
import torch
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path

from sssrlib.patches import Patches
from sssrlib.sample import AvgGradSample, SuppressWeights


def test_sup_sample():
    image = np.load('shepp3d.npy')
    image = zoom(image, (1, 1, 0.4))
    patches = Patches((32, 1, 10), image, named=False)

    dirname = 'results_sup_sample'
    os.system('rm -rf %s' % dirname)
    sample = AvgGradSample(patches, voxel_size=(1, 1, 2.5))
    sample = SuppressWeights(sample, stride=[16, 16, 1], kernel_size=[32, 32, 1])
    sample.save_figures(dirname)
    indices = sample.sample_indices(3)
    batch = sample.get_patches(indices)
    for i in range(batch.shape[0]):
        fig = plt.figure()
        plt.imshow(batch[i, ...].squeeze())
        fig.savefig(Path(dirname, '%s.png' % i))

    filename = Path(dirname, 'sup-weights.nii.gz')
    nib.Nifti1Image(sample._sup_weights.cpu().numpy(), np.eye(4)).to_filename(filename)
    filename = Path(dirname, 'weights.nii.gz')
    nib.Nifti1Image(sample.sample._weights.cpu().numpy(), np.eye(4)).to_filename(filename)

if __name__ == '__main__':
    test_sup_sample()
