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

from .utils import save_fig


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

    def save_figures(self, dirname, d3=True):
        """Saves the image.

        Args:
            dirname (str): The output directory.
            d3 (bool): Save as 3D nifti file if ``True``.

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
        voxel_size (iterable[float]): The size of the voxel before permutation.

    Attributes:
        patch_size (tuple[int]): The parsed patch size.
        image (numpy.ndarray): The image to get patches from.
        x (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the x-axis in the result image.
        y (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the y-axis in the result image.
        z (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the z-axis in the result image.
        voxel_size (numpy.ndarray): The size of the voxel AFTER permutation.
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
                 voxel_size=(1, 1, 1), patches=None, named=True, squeeze=True,
                 expand_channel_dim=True, verbose=False):
        self.patch_size = self._parse_patch_size(patch_size)
        self.named = named
        self.squeeze = squeeze
        self.expand_channel_dim = expand_channel_dim
        self.verbose = verbose

        if image is not None:
            self._init_from_image(image, x, y, z, voxel_size)
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

    def _init_from_image(self, image, x, y, z, voxel_size):
        self.x = x
        self.y = y
        self.z = z
        self.voxel_size = np.array((voxel_size[x],
                                    voxel_size[y],
                                    voxel_size[z]))
        self.image = self._permute_image(image)

    def _init_from_patches(self, patches):
        attrs = ['x', 'y', 'z', 'image', 'voxel_size']
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
                   'Voxel size: {}'.format(self.voxel_size),
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
            self._name_pattern = 'x%%0%dd-y%%0%dd-z%%0%dd'
            self._name_pattern = self._name_pattern % (nx, ny, nz)
        return self._name_pattern % ind

    def save_figures(self, dirname, d3=True):
        save_fig(dirname, self.image, 'image', cmap='gray', d3=d3)


class TransformedPatches(AbstractPatches):
    """Wrapper class to transform Pathces.

    See :mod:`ssrlib.transform` for more details of the transforms.

    """
    def __init__(self, patches, transform):
        self.patches = patches
        self.transform = transform

    @property
    def named(self):
        return self.patches.named

    @property
    def patch_size(self):
        return self.patches.patch_size

    @property
    def image(self):
        return self.patches.image

    @property
    def xnum(self):
        return self.patches.xnum

    @property
    def ynum(self):
        return self.patches.ynum

    @property
    def znum(self):
        return self.patches.znum

    @property
    def voxel_size(self):
        return self.patches.voxel_size

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

    def save_figures(self, dirname, d3=True):
        self.patches.save_figures(dirname, d3=d3)


class PatchesCollection(AbstractPatches):
    """A collection of :class:`AbstractPatches`

    Note:
        This class can be used to select from different image orientations,
        for example, x-z or y-z patches, and transforms.

    Args:
        patches (AbstractPatches): An :class:`AbstractPatches` instance.

    """
    def __init__(self, *patches):
        self._collection = list(patches)
        assert len(np.unique([p.named for p in self._collection])) == 1
        self.named = self._collection[0].named

    def __len__(self):
        return np.sum([len(p) for p in self._collection])

    def __str__(self):
        message = ['Patches #%d\n%s' % (i,  p.__str__())
                   for i, p in enumerate(self._collection)]
        message = '\n----------\n'.join(message)
        return message

    def __getitem__(self, ind):
        """Retunrs a patch.

        Args:
            ind (tuple[ind]): The first index is for the patches instances, the
                second index is for a patch given the patches instance.

        """
        patch = self._collection[ind[0]][ind[1]]

        if self.named:
            num_digits = len(str(len(self._collection)))
            pind = ('p%%0%dd' % num_digits) % ind[0]
            name = '_'.join([pind, patch.name])
            patch = NamedData(name, patch.data)

        return patch

    def append(self, patches):
        self._collection.append(patches)

    def save_figures(self, dirname, d3=True):
        num_digits = len(str(len(self._collection)))
        for i, patch in enumerate(self._collection):
            subdir = Path(dirname, ('%%0%dd' % num_digits) % i)
            patch.save_figures(subdir, d3=d3)
