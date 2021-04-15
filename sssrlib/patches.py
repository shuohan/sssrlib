"""Classes to output patches from a 3D image.

TODO:
    Add :attr:`Patches.patch_gaps`.

"""
import numpy as np
import torch
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


NamedData = namedtuple('NamedData', ['name', 'data'])
"""Data with its name.

Args:
    name (str): The name of the data.
    data (numpy.ndarray): The data.

"""


class AbstractPatches:
    """Abstract class to output patches from a 3D image.

    """
    def __getitem__(self, ind):
        """Returns a patch at index."""
        raise NotImplementedError

    def __len__(self):
        """Returns the number of possible patches."""
        raise NotImplementedError

    def cpu(self):
        """Puts patches into CPU.

        Returns:
            Patches: The instance itself.

        """
        raise NotImplementedError

    def cuda(self):
        """Puts patches into CUDA.

        Returns:
            Patches: The instance itself.

        """
        raise NotImplementedError


class Patches(AbstractPatches):
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

    @property
    def xnum(self):
        return self._xnum

    @property
    def ynum(self):
        return self._ynum

    @property
    def znum(self):
        return self._znum

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
        self.image = self.image.cuda()
        return self

    def cpu(self):
        self.image = self.image.cpu()
        return self

    def __len__(self):
        """Returns the number of patches"""
        return self._len

    def __str__(self):
        message = ['X axis: %d' % self.x,
                   'Y axis: %d' % self.y,
                   'Z axis: %d' % self.z,
                   'Patch size: {}'.format(self.patch_size),
                   'Number of patches along X: %d' % self._xnum,
                   'Number of patches along Y: %d' % self._ynum,
                   'Number of patches along Z: %d' % self._znum,
                   'Number of total patches: %d' % len(self)]
        return '\n'.join(message)

    def __getitem__(self, ind):
        """Returns a patch at the index.

        Args:
            ind (int): The flattened index of a patch.

        Returns:
            torch.Tensor or NamedData: The returned tensor.

        """
        ind = self.unravel_index(ind)
        patch = self.get_patch(ind)
        if self.named:
            name = self.get_name(ind)
            patch = NamedData(name, patch)
        return patch

    def unravel_index(self, ind):
        """Converts the flattened index into transform index and array coords.

        Args:
            index (int): The flattened index.

        Returns:
            tuple[int]: The array coordinates.

        """
        return np.unravel_index(ind, [self.xnum, self.ynum, self.znum])

    def get_patch(self, ind):
        if self.verbose:
            print('x_start %d, y_start %d, z_start %d' % ind)
        bbox = tuple(slice(s, s + p) for s, p in zip(ind, self.patch_size))
        patch = self.image[bbox]
        patch = patch.squeeze() if self.squeeze else patch
        patch = patch[None, ...] if self.expand_channel_dim else patch
        return patch

    def get_name(self, ind):
        if not hasattr(self, '_name_pattern'):
            nx = len(str(self._xnum))
            ny = len(str(self._ynum))
            nz = len(str(self._znum))
            self._name_pattern = 'ind-x%%0%dd-y%%0%dd-z%%0%dd'
            self._name_pattern = self._name_pattern % (nx, ny, nz)
        return self._name_pattern % ind


class TransformedPatches(AbstractPatches):
    """Wrapper class to transform Pathces.

    See :mod:`ssrlib.transform` for more details of the transforms.

    """
    def __init__(self, patches, transform):
        self.patches = patches
        self.transform = transform

    def __getitem__(self, ind):
        ind = self.patches.unravel_index(ind)
        patch = self.patches.get_patch(ind)
        patch = self.transform(patch)
        if self.patches.named:
            name = self.patches.get_name(ind)
            name = '_'.join([name, self.transform.get_name()])
            patch = NamedData(name, patch)
        return patch

    def __len__(self):
        return self.patches.__len__()

    def __str__(self):
        return '\n'.join([self.transform.__str__(), self.patches.__str__()])

    def cpu(self):
        self.patches.cpu()
        return self

    def cuda(self):
        self.patches.cuda()
        return self


class PatchesOr(AbstractPatches):
    """Selects one of :class:`AbstractPatches` instance to output patches.

    Note:
        This class can be used to select from different image orientations,
        for example, x-z or y-z patches, and transforms.

    Attributes:
        patches (list[AbstractPatches]): The :class:`AbstractPatches` instances
            to select from.

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


class PatchesAnd(AbstractPatches):
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
