"""Classes to output patches from a 3D image.

TODO:
    Add :attr:`Patches.patch_gaps`.

"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import namedtuple
from collections.abc import Iterable
from enum import IntEnum
from improc3d import permute3d
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from pathlib import Path

from .transform import Identity


NamedData = namedtuple('NamedData', ['name', 'data'])
"""Data with its name.

Args:
    name (str): The name of the data.
    data (numpy.ndarray): The data.

"""


class _AbstractPatches:
    """Abstract class to output patches from a 3D image.

    """
    def __getitem__(self, ind):
        """Returns a patch at index."""
        raise NotImplementedError

    def __len__(self):
        """Returns the number of possible patches."""
        raise NotImplementedError

    def get_dataloader(self, batch_size, weighted=True, num_workers=0):
        """Returns the torch.utils.data.DataLoader of ``self``.

        Warning:
            Only support 2D patches with size [x, y, 1] for now.

        Args:
            batch_size (int): The number of samples per mini-batch.
            weighted (bool): Weight sampling with image gradients.

        Returns:
            torch.utils.data.DataLoader: The data loader of the :class:`Patches`
                instance itself.

        """
        if weighted:
            weights = self.get_sample_weights()
            sampler = WeightedRandomSampler(weights, batch_size)
        else:
            sampler = RandomSampler(self, replacement=True,
                                    num_samples=batch_size)
        dataloader = DataLoader(self, batch_size=batch_size, sampler=sampler,
                                num_workers=num_workers)
        return dataloader

    def get_sample_weights(self):
        """Returns the sampling weights of each patch."""
        raise NotImplementedError


class Patches(_AbstractPatches):
    """Outputs patches from a 3D image.

    Args:
        patch_size (int or tuple[int]): The size of the output patch. If
            its type is :class:`tuple`, the length should be 3. If it is an
            integer, it is assumed to be a 2D patch with the same size along x
            and y axes. To get 1D patches, its last two elements should be 1.
            To get 2D patches, its last one elements should be 1.

    Attributes:
        patch_size (tuple[int]): The parsed patch size.
        image (numpy.ndarray): The image to get patches from.
        x (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the x-axis in the result image.
        y (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the y-axis in the result image.
        z (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the z-axis in the result image.
        patches (Patches): Initialize from another patches.
        named (bool): If ``True``, return the patch index as well.
        squeeze (bool): If ``True``, squeeze sampled patches.
        expand_channel_dim (bool): If ``True``, expand a channel dimension in
                before the 0th dimension.
        verbose (bool): If ``True``, print message in :meth:`__getitem__`.

    Raises:
        RuntimeError: Incorrect :attr:`patch_size`.

    """
    def __init__(self, patch_size, image=None, x=0, y=1, z=2,
                 patches=None, named=True, squeeze=True,
                 expand_channel_dim=True, verbose=False):
        self.patch_size = self._parse_patch_size(patch_size)
        self.named = named
        self.squeeze = squeeze
        self.expand_channel_dim = expand_channel_dim
        self.verbose = verbose

        if image is not None:
            self._init_from_image(image, x, y, z)
        elif patches is not None:
            self._init_from_patches(patches)
        else:
            raise RuntimeError('"image" and "patches" cannot be both None.')

        self._xnum, self._ynum, self._znum = self._init_patch_numbers()
        self._len = self._xnum * self._ynum * self._znum

    def _parse_patch_size(self, patch_size):
        """Converts :class:`int` patch size to :class:`tuple`."""
        if not isinstance(patch_size, Iterable):
            patch_size = (patch_size, ) * 2 + (1, )
        assert len(patch_size) == 3
        return patch_size

    def _init_from_image(self, image, x=0, y=1, z=2):
        self.x, self.y, self.z = (x, y, z)
        self.image = self._permute_image(image)

    def _init_from_patches(self, patches):
        attrs = ['x', 'y', 'z', 'image']
        for attr in attrs:
            setattr(self, attr, getattr(patches, attr))

    def _permute_image(self, image):
        """Permutes the input image."""
        image = torch.tensor(image).float().contiguous()
        image = permute3d(image, self.x, self.y, self.z)[0]
        return image

    def _init_patch_numbers(self, shape=None):
        shape = self.image.shape if shape is None else shape
        return [s - ps + 1 for s, ps in zip(shape, self.patch_size)]

    def cuda(self):
        """Puts patches into CUDA.

        Returns:
            Patches: The instance itself.

        """
        self.image = self.image.cuda()
        return self

    def cpu(self):
        """Puts patches into CPU.

        Returns:
            Patches: The instance itself.

        """
        self.image = self.image.cpu()
        return self

    def __len__(self):
        """Returns the number of patches"""
        return self._len

    def __str__(self):
        message = [t.__str__() for t in self.transforms]
        prefix = ' ' * (len('Transforms:') + 2)
        message = '[{}]'.format((',\n' + prefix).join(message))
        message = ['X axis: %d' % self.x,
                   'Y axis: %d' % self.y,
                   'Z axis: %d' % self.z,
                   'Patch size: {}'.format(self.patch_size),
                   'Transforms: {}'.format(message),
                   'Number of transforms: %d' % self._tnum,
                   'Number of patches along X: %d' % self._xnum,
                   'Number of patches along Y: %d' % self._ynum,
                   'Number of patches along Z: %d' % self._znum,
                   'Number of total patches: %d' % len(self)]
        return '\n'.join(message)

    def __getitem__(self, ind):
        """Returns a patch at indices.

        Args:
            ind (tuple[int]): The indices (x_start, y_start, z_start) of a
                patch.

        Returns:
            torch.Tensor or NamedData: The returned tensor.

        """
        if self.verbose:
            print('x_start %d, y_start %d, z_start %d' % ind)
        bbox = tuple(slice(s, s + p) for s, p in zip(ind, self.patch_size))
        patch = self.image[bbox]
        patch = patch.squeeze() if self.squeeze else patch
        patch = patch[None, ...] if self.expand_channel_dim else patch
        patch = NamedData(self._get_name(ind), patch) if self.named else patch
        return patch

    def _get_name(self, ind):
        if not hasattr(self, '_name_pattern'):
            nx = len(str(self._xnum))
            ny = len(str(self._ynum))
            nz = len(str(self._znum))
            self._name_pattern = 'ind-x%%0%dd-y%%0%dd-z%%0%dd'
            self._name_pattern = self._name_pattern % (nx, ny, nz)
        return self._name_pattern % ind


class PatchesOr(_AbstractPatches):
    """Selects one of :class:`Patches` instance to output patches.

    Note:
        This class is mainly used to select from different image orientations,
        for example, x-z or y-z patches.

    Attributes:
        patches (list[Patches]): The :class:`Patches` instances to select from.

    """
    def __init__(self, *patches):
        self.patches = list(patches)
        assert len(np.unique([p.named for p in self.patches])) == 1
        self.named = self.patches[0].named
        self._nums = [len(p) for p in self.patches]
        self._cumsum = np.cumsum(self._nums)

    def __len__(self):
        return np.sum(self._nums)

    def __str__(self):
        message = ['Patches #%d\n%s' % (i,  p.__str__())
                   for i, p in enumerate(self.patches)]
        message = '\n----------\n'.join(message)
        return message

    def __getitem__(self, ind):
        pind = np.digitize(ind, self._cumsum)
        ind = ind - self._cumsum[pind - 1] if pind > 0 else ind
        patch = self.patches[pind][ind]

        if self.named:
            names = patch.name.split('-')
            pattern = 'p%%0%dd' % len(str(len(self.patches)))
            names.insert(1, pattern % pind)
            patch = NamedData('-'.join(names), patch.data)

        return patch

    def get_sample_weights(self):
        weights = [p.get_sample_weights() for p in self.patches]
        self._cumsum = np.cumsum([len(w) for w in weights])
        weights = torch.cat(weights)
        return weights

    def register(self, patches):
        raise NotImplementedError


class PatchesAnd(_AbstractPatches):
    """Returns a patch from each of :class:`Patches` instances.

    Note:
        This class is mainly used to yield training pairs. The number of patches
        for each :class:`Patches` instance should be the same.

    Attributes:
        patches (list[Patches]): The :class:`Patches` instances to output
            patches from.

    """
    def __init__(self, *patches):
        self.patches = list(patches)

    def __len__(self):
        return len(self.patches[0])

    def __getitems__(self, ind):
        return tuple(p[ind] for p in self.patches)


################# 

    def get_sample_weights(self):
        """Returns the sampling weights of each patch.

        """
        if self._weight_map is None:
            self._calc_weight_map()

        weights = self._weight_map.repeat(len(self.transforms))
        weights = weights / torch.sum(weights)

        if self.compress:
            self._ind_mapping = torch.where(weights > 0)[0]
            self._ind_mapping = self._ind_mapping.cpu().numpy().tolist()
            weights = weights[self._ind_mapping]

        if self.weight_dir is not None:
            self.weight_dir.mkdir(exist_ok=True, parents=True)
            for i, w in enumerate(self._grad_w):
                self._save_figures(w, 'grad_weights%d' % i, cmap='jet')
            self._save_figures(self._prod_w, 'prod_weights', cmap='jet')
            self._save_figures(self._unpool_w, 'pool_weights', cmap='jet')

        return weights

    def _calc_weight_map(self):
        grads = self._calc_image_grads()
        shifts = [self._calc_shift(self.patch_size[0], self._xnum),
                  self._calc_shift(self.patch_size[1], self._ynum),
                  self._calc_shift(self.patch_size[2], self._znum)]
        grad_w = [grad[tuple(shifts)] for grad in grads]
        grad_w = [w for w, ug in zip(grad_w, self.use_grads) if ug]
        grad_w = [w / torch.sum(w) for w in grad_w]

        num_grads = len(grad_w)
        prod_w = torch.prod(torch.stack(grad_w), axis=0) ** (1 / num_grads)
        prod_w = prod_w[None, None, ...]

        kernel_size = [2 * s for s in self.weight_stride]
        stride = self.weight_stride
        pool_w, indices = F.max_pool3d(prod_w, kernel_size=kernel_size,
                                       stride=stride, return_indices=True)
        unpool_w = F.max_unpool3d(pool_w, indices, kernel_size, stride=stride,
                                  output_size=prod_w.shape)

        self._grad_w = grad_w
        self._prod_w = prod_w
        self._unpool_w = unpool_w
        self._weight_map = unpool_w.flatten()

    def _save_figures(self, image, prefix, cmap='jet'):
        image = (image.squeeze() - image.min()) / (image.max() - image.min())
        views = {'12': image[image.shape[0]//2, :, :].cpu().numpy(),
                 '02': image[:, image.shape[1]//2, :].cpu().numpy(),
                 '01': image[:, :, image.shape[2]//2].cpu().numpy()}
        for k, v in views.items():
            filename = Path(self.weight_dir, '%s_%s.png' % (prefix, k))
            plt.imsave(filename, v, vmin=0, vmax=0.95, cmap=cmap)


    def _calc_avg_grad(self, grad):
        avg_kernel = torch.ones(self.patch_size, dtype=grad.dtype,
                                device=grad.device)[None, None, ...]
        avg_kernel = avg_kernel / torch.sum(avg_kernel)
        padding = [ps // 2 for ps in self.patch_size]
        avg_grad = F.conv3d(grad, avg_kernel, padding=padding)
        return avg_grad

    def _calc_shift(self, patch_size, image_size):
        left_shift = (patch_size - 1) // 2
        right_shift = image_size + left_shift
        shift = slice(left_shift, right_shift)
        return shift

    def _calc_image_grads(self):
        """Calculates the image graident magnitude.

        """
        image = self.image[None, None, ...]
        dn_im = self._denoise(image) if self.sigma > 0 else image

        mode = 'trilinear'
        scale_factor = [vs / max(self.voxel_size) for vs in self.voxel_size]
        interp_im = F.interpolate(dn_im, scale_factor=scale_factor, mode=mode)

        shape = self.image.shape
        grads = self._calc_sobel_grads(interp_im)
        grads = tuple(F.interpolate(g, size=shape, mode=mode) for g in grads)
        grads = tuple(g.squeeze() for g in grads)

        if self.weight_dir is not None:
            self._save_figures(image, 'image', cmap='gray')
            self._save_figures(dn_im, 'denosie_image', cmap='gray')
            self._save_figures(interp_im, 'interp_image', cmap='gray')
            for i, g in enumerate(grads):
                self._save_figures(g, 'grads%d' % i, cmap='jet')

        return grads

    def _denoise(self, image):
        gauss_kernel = self._get_gaussian_kernel()
        gauss_kernel = gauss_kernel.to(image.device)
        padding = [s // 2 for s in gauss_kernel.shape[2:]]
        image = F.conv3d(image, gauss_kernel, padding=padding)

        if self.weight_dir is not None:
            self._save_figures(gauss_kernel, 'denosie_kernel', cmap='jet')

        return image

    def _get_gaussian_kernel(self):
        length = 4 * self.sigma * 2 + 1
        coord = np.arange(length) - length // 2
        grid = np.meshgrid(coord, coord, coord, indexing='ij')
        sigmas = self.sigma / np.array(self.voxel_size)
        kernels = [np.exp(-(g ** 2) / (2 * s ** 2))
                   for g, s in zip(grid, sigmas)]
        kernel = np.prod(kernels, axis=0)
        kernel = kernel / np.sum(kernel)
        kernel = torch.tensor(kernel, device=self.image.device).float()
        kernel = kernel[None, None, ...]
        return kernel

    def _calc_sobel_grads(self, image):
        sz = torch.stack((torch.tensor([[0, 0,  0], [1, 0, -1], [0, 0,  0]]),
                          torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
                          torch.tensor([[0, 0,  0], [1, 0, -1], [0, 0,  0]])))
        sz = sz.float().to(image.device)
        sx = sz.permute([2, 0, 1])
        sy = sz.permute([1, 2, 0])
        grads = list()
        for s in (sx, sy, sz):
            grad = F.conv3d(image, s[None, None, ...], padding=1)
            grad = torch.abs(grad)
            grad = self._calc_avg_grad(grad) if self.avg_grad else grad
            grads.append(grad)
        return grads


"""
'Deniose sigma: {}'.format(self.sigma),
'Voxel size: {}'.format(self.voxel_size),
'Weight stride: {}'.format(self.weight_stride),
'Use gradients: {}'.format(self.use_grads),

        self._ind_mapping = None
        self._weight_map = None
        self._grad_w = None
        self._prod_w = None
        self._unpool_w = None
self.compress = compress
self.use_grads = use_grads
self.weight_dir = weight_dir
self.voxel_size = voxel_size
self.sigma = sigma
self.avg_grad = avg_grad
self.weight_stride = weight_stride
sigma=0, voxel_size=(1, 1, 1),
                 weight_stride=(1, 1, 1), weight_dir=None, avg_grad=False,
                 use_grads=[True, True, True], compress=False,

        voxel_size (tuple[float]): The size of the voxel after permutation.
        sigma (float): The sigma of Gaussian filter to denoise before
            calculating the probability map. Do not denoise if 0.
        avg_grad (bool): Average the image gradients when calculating sampling
            weights.
        weight_stride(tuple[int]): The strides between sampling weights after
            permutation.
        weight_dir (str): Save figures of weights into this directory if not
            ``None``.
        compress (bool): Compress the number of available patches according to
            non-zero sampling weights.
"""
