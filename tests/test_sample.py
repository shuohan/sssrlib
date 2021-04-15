#!/usr/bin/env python

import os
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from pathlib import Path

from sssrlib.patches import Patches
from sssrlib.sample import UniformSample, GradSample


def test_sample():
    image = np.load('shepp3d.npy')
    image = zoom(image, (1, 1, 0.4))
    patches = Patches((32, 1, 10), image, named=False)

    dirname = 'results_sample'
    os.system('rm -rf %s' % dirname)
    sample = GradSample(patches, 3, voxel_size=(1, 1, 2.5))
    sample.save_figures(dirname)
    indices = sample.sample()

    for ind in indices:
        patch = patches[ind]
        fig = plt.figure()
        plt.imshow(patch.squeeze())
        fig.savefig(Path(dirname, '%s.png' % ind))

    assert len(sample._weights) == len(patches)
    grads = [sample._gradients[0], sample._gradients[2]]

    starts = [(ps - 1) // 2 for ps in patches.patch_size]
    sum0 = grads[0][starts[0] : starts[0]+patches.xnum,
                    starts[1] : starts[1]+patches.ynum,
                    starts[2] : starts[2]+patches.znum].sum()
    sum1 = grads[1][starts[0] : starts[0]+patches.xnum,
                    starts[1] : starts[1]+patches.ynum,
                    starts[2] : starts[2]+patches.znum].sum()
    grads[0] = grads[0] / sum0
    grads[1] = grads[1] / sum1
    grad = grads[0] * grads[1]
    for i in range(100):
        ind = int(np.random.uniform(0, len(sample._weights)))
        x, y, z = np.unravel_index(ind, [patches.xnum, patches.ynum, patches.znum])
        assert sample._weights[ind] == grad[x + (patches.patch_size[0] - 1)//2,
                                            y + (patches.patch_size[1] - 1)//2,
                                            z + (patches.patch_size[2] - 1)//2]

    print('successful')

if __name__ == '__main__':
    test_sample()
