#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Sample patches from an image.')
parser.add_argument('-i', '--image', required=True)
parser.add_argument('-o', '--outdir', required=True)
parser.add_argument('-p', '--patch-size', default=(32, 32, 1), type=int,
                    nargs=3, help='The size of patches to sample.')
parser.add_argument('-a', '--to-axial-view', action='store_true',
                    help='Transpose to the axial (RAI minus) view.')
parser.add_argument('-n', '--num-patches', default=1000, type=int,
                    help='The number of patches to sample.')

help = ('The indices of gradient directinos to use. 0, 1, 2: the gradient '
        'calculated along the first, second, third image dimensions, '
        'respectively. If this option is not specified, use uniform sampling; '
        'otherwise, the sampling is weighted by the gradients specified. ')
parser.add_argument('-g', '--use-gradients', nargs='+', type=int,
                    choices={0, 1, 2}, help=help)

parser.add_argument('-S', '--sigma', default=3, type=int,
                    help='Denoising sigma when calculating the gradients.')
parser.add_argument('-A', '--use-agg-grads', action='store_true',
                    help='Aggregate image gradients within each patch.')
parser.add_argument('-s', '--suppress-weights', action='store_true',
                    help='Supress non-maxima locally.')
parser.add_argument('-sk', '--suppress-kernel-size', default=(4, 4, 1),
                    nargs=3, type=int,
                    help='The kernel size used when suppressing weights.')
parser.add_argument('-ss', '--suppress-stride', default=(2, 2, 1),
                    nargs=3, type=int,
                    help='The stride used when suppressing weights.')

help = ('The type of fuzzy operator to combine the probability maps calculated '
        'from the gradients. Suppose -g 0 1, patches that have high gradients '
        'x or/and y directions are likely sampled.')
parser.add_argument('-f', '--fuzzy-op', choices={'and', 'or'}, default='or',
                    help=help)

help = ('A description string such as flip01, flip0, rot901, and rot902. '
        'Multiple transforms are joined with _, e.g., flip12_rot901. Note that '
        'when using rot90, the sizes of patches should be equal along the two '
        'dimensions.')
parser.add_argument('-t', '--transforms', help=help)

args = parser.parse_args()


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from improc3d import transform_to_axial
from pathlib import Path

from sssrlib.patches import Patches, TransformedPatches
from sssrlib.transform import create_transform_from_desc
from sssrlib.sample import Sampler, ImageGradients
from sssrlib.sample import SuppressWeights, GradSampleWeights
from sssrlib.utils import calc_avg_kernel


obj = nib.load(args.image)
image = obj.get_fdata(dtype=np.float32)
if args.to_axial_view:
    image = transform_to_axial(image, obj.affine, coarse=True).copy()
patches = Patches(args.patch_size, image)
if args.transforms not in {None, 'none'}:
    trans = create_transform_from_desc(args.transforms)
    patches = TransformedPatches(patches, trans)
dirname = Path(args.outdir, 'image')
dirname.mkdir(exist_ok=True, parents=True)
patches.save_figures(dirname, d3=False)

print('-' * 80)
print('Patches:')
print(patches)

if args.use_gradients:
    im_grads = ImageGradients(patches, sigma=args.sigma)
    dirname = Path(args.outdir, 'grads')
    dirname.mkdir(exist_ok=True, parents=True)
    im_grads.save_figures(dirname, d3=False)

    print('-' * 80)
    print('Gradients:')
    print(im_grads)

    gradients = tuple(im_grads.gradients[i] for i in args.use_gradients)
    agg_k = calc_avg_kernel(args.patch_size) if args.use_agg_grads else None
    weights = GradSampleWeights(patches, gradients, weights_op=args.fuzzy_op,
                                agg_kernel=agg_k)
    if args.suppress_weights:
        weights = SuppressWeights(weights,
                                  kernel_size=args.suppress_kernel_size,
                                  stride=args.suppress_stride)
    dirname = Path(args.outdir, 'weights')
    dirname.mkdir(exist_ok=True, parents=True)
    weights.save_figures(dirname, d3=False)
    print('-' * 80)
    print('Weights:')
    print(weights)

weights_flat = weights.weights_flat if args.use_gradients else None
weights_mapping = weights.weights_mapping if args.suppress_weights else None
sampler = Sampler(patches, weights_flat, weights_mapping)

indices = sampler.sample_indices(args.num_patches)
batch = sampler.get_patches(indices)
dirname = Path(args.outdir, 'patches')
dirname.mkdir(exist_ok=True, parents=True)
for i in range(batch.data.shape[0]):
    if i % 1000 == 0:
        print('Save patch', i)
    data = batch.data[i, 0, ...]
    name = batch.name[i]
    plt.imsave(Path(dirname, name + '.jpg'), data.squeeze().T, cmap='gray')
np.save(Path(args.outdir, 'patches.npy'), batch.data)
