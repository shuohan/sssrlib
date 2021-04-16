#!/usr/bin/env python

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sssrlib.patches import Patches, PatchesCollection, TransformedPatches
from sssrlib.patches import NamedData
from sssrlib.sample import SampleCollection, GradSample, AvgGradSample
from sssrlib.sample import SuppressWeights
from sssrlib.transform import Flip


def test_patches():
    ps2d = (60, 45, 1)
    image = np.load('shepp3d.npy')

    patches0 = Patches(ps2d, image, x=2, y=1, z=0)
    patches1 = Patches(ps2d, image, x=2, y=0, z=1)
    patches2 = Patches(ps2d, image, x=1, y=2, z=0)
    patches2 = TransformedPatches(patches2, Flip())
    patches = PatchesCollection(patches0, patches1, patches2)
    
    assert len(patches) == len(patches0) + len(patches1) + len(patches2)
    assert np.array_equal(patches[2, 100].data, patches2[100].data)
    assert patches[2, 13140].name.startswith('p2')

    dirname = Path('results_patches-collection')
    os.system('rm -rf %s' % dirname)
    dirname.mkdir(exist_ok=True)

    sample0 = GradSample(patches0, use_grads=[True, False, False])
    sample1 = AvgGradSample(patches1, use_grads=[True, False, False])
    sample2 = AvgGradSample(patches2, use_grads=[True, False, False])
    sample2 = SuppressWeights(sample2)
    sample = SampleCollection(sample0, sample1, sample2)
    indices = sample.sample_indices(10)
    
    batch = sample.get_patches(indices)
    assert list(batch.data.shape) == [10, 1, 60, 45]
    for i in range(batch.data.shape[0]):
        data = batch.data[i, 0, ...]
        name = batch.name[i]
        fig = plt.figure()
        plt.imshow(data)
        fig.savefig(Path(dirname, name + '.png'))
    sample.save_figures(dirname)

    print('successful')


if __name__ == '__main__':
    test_patches()
