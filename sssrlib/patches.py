"""Classes to output patches from a 3D image.

TODO:
    Add :attr:`Patches.patch_gaps`.

"""
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from collections.abc import Iterable
from enum import IntEnum
from image_processing_3d import permute3d
from torch.nn.functional import interpolate as interp
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.ndimage import gaussian_filter

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

    def get_dataloader(self, batch_size, num_workers=0):
        """Returns the torch.utils.data.DataLoader of ``self``.

        Warning:
            Only support 2D patches with size [x, y, 1] for now.

        Args:
            batch_size (int): The number of samples per mini-batch.

        Returns:
            torch.utils.data.DataLoader: The data loader of the :class:`Patches`
                instance itself.

        """
        weights = self.get_sample_weights()
        sampler = WeightedRandomSampler(weights, batch_size)
        return DataLoader(self, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers)

    def get_sample_weights(self):
        """Returns the sampling weights of each patch."""
        raise NotImplementedError


class Patches(_AbstractPatches):
    """Outputs patches from a 3D image.

    * This class permutes the input image to put the axes into a fixed order.
    * The patch indices are arranged in the following way:

    .. code-block:: none

        [[coords, ...], [coords, ...], ...]
        --------------  -------------
          transform 0    transform 1   ...

    where ``coords`` corresponds to the x, y, z starts of the patch.

    Note:
        If the number of patches exceeds 2^24, the image is cropped
        symmetrically from its boundary. See
        https://github.com/pytorch/pytorch/issues/2576.

    Args:
        patch_size (int or tuple[int]): The size of the output patch. If
            its type is :class:`tuple`, the length should be 3. If it is an
            integer, it is assumed to be a 2D patch with the same size along x
            and y axes. To get 1D patches, its last two elements should be 1.
            To get 2D patches, its last one elements should be 1.

    Attributes:
        image (numpy.ndarray): The image to get patches from.
        patch_size (tuple[int]): The parsed patch size.
        x (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the x-axis in the result image.
        y (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the y-axis in the result image.
        z (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the z-axis in the result image.
        transforms (iterable[sssrlib.transform.Transform]): Transform the an
            output image patch. If empty, :class:`sssrlib.transform.Identity`
            will be used.
        scale_factor (float): The scale factor of :attr:`image` along
            the x-axis dimension after permutation.
        mode (str): The interpolation mode for
            :func:`torch.nn.functional.interpolate`.
        named (bool): If ``True``, return the patch index as well.
        squeeze (bool): If ``True``, squeeze sampled patches.
        expand_channel_dim (bool): If ``True``, expand a channel dimension in
                before the 0th dimension.

    Raises:
        RuntimeError: Incorrect :attr:`patch_size`.

    """
    def __init__(self, image, patch_size, x=0, y=1, z=2, transforms=[],
                 scale_factor=1, mode='linear', named=True,
                 squeeze=True, expand_channel_dim=True):
        self.x, self.y, self.z = (x, y, z)
        self.scale_factor = scale_factor
        self.mode = mode
        self.image = self._proc_image(image)
        self.patch_size = self._parse_patch_size(patch_size)
        self.transforms = [Identity()] if len(transforms) == 0 else transforms
        self.scale_factor = scale_factor
        self.named = named
        self.squeeze = squeeze
        self.expand_channel_dim = expand_channel_dim
        self._tnum = len(self.transforms)
        self._xnum, self._ynum, self._znum = self._init_patch_numbers()
        self._len = self._tnum * self._xnum * self._ynum * self._znum
        self._name_pattern = None

    def _proc_image(self, image):
        """Permutes the input image and interpolates along the x-axis (0th)."""
        image = torch.tensor(image).float().contiguous()
        image, self._xinv, self._yinv, self._zinv \
            = permute3d(image, self.x, self.y, self.z)
        if self.scale_factor != 1:
            image = self._interp_image(image)
        return image

    def _interp_image(self, image):
        """Interpolates the image along the x-axis."""
        orig_shape = image.shape
        image = image.permute(2, 1, 0)
        image = image.reshape(-1, 1, orig_shape[0])
        image = interp(image, scale_factor=self.scale_factor, mode=self.mode)
        result_shape = (orig_shape[2], orig_shape[1], -1)
        image = image.reshape(*result_shape)
        image = image.permute(2, 1, 0)
        return image

    def _parse_patch_size(self, patch_size):
        """Converts :class:`int` patch size to :class:`tuple`."""
        if not isinstance(patch_size, Iterable):
            patch_size = (patch_size, ) * 2 + (1, )
        assert len(patch_size) == 3
        return patch_size

    def _init_patch_numbers(self, shape=None):
        """Calculates the possible numbers of patches along x, y, and z.

        """
        shape = self.image.shape if shape is None else shape

        while True:
            nums = [s - ps + 1 for s, ps in zip(shape, self.patch_size)]
            if np.prod(nums) <= 2 ** 24 // self._tnum:
                break
            shape = [s - 1 for s in shape]

        if shape != self.image.shape:
            print('Too many patches. Crop the image from', self.image.shape,
                  'to', tuple(shape))
            lefts = [(s1 - s2)//2 for s1, s2 in zip(self.image.shape, shape)]
            rights = [l + s for l, s in zip(lefts, shape)]
            slices = tuple(slice(l, r) for l, r in zip(lefts, rights))
            self.image = self.image[slices]
            assert nums == [s - ps + 1 for s, ps in
                            zip(self.image.shape, self.patch_size)]

        return nums

    def cuda(self):
        """Puts patches into GPU.

        Returns:
            Patches: The instance itself.

        """
        self.image = self.image.cuda()
        return self

    def __len__(self):
        """Returns the number of patches"""
        return self._len

    def __str__(self):
        message = ['Scale factor: %g' % self.scale_factor,
                   'X axis: %d' % self.x,
                   'Y axis: %d' % self.y,
                   'Z axis: %d' % self.z,
                   'Number of transforms: %d' % self._tnum,
                   'Number of patches along X: %d' % self._xnum,
                   'Number of patches along Y: %d' % self._ynum,
                   'Number of patches along Z: %d' % self._znum,
                   'Number of total patches: %d' % len(self)]
        return '\n'.join(message)

    def __getitem__(self, ind):
        """Returns a patch at index.

        Args:
            ind (int): The flattened index of the patch to return.

        Returns:
            torch.Tensor or NamedData: The returned tensor.

        """
        tind, x, y, z = self._unravel_index(ind)
        loc = tuple(slice(s, s + p) for s, p in zip((x, y, z), self.patch_size))
        patch = self.transforms[tind](self.image[loc])
        patch = patch.squeeze() if self.squeeze else patch
        patch = patch[None, ...] if self.expand_channel_dim else patch

        if self.named:
            name = self._get_name_pattern() % (tind, x, y, z)
            patch = NamedData(name, patch)

        return patch

    def _get_name_pattern(self):
        if self._name_pattern is None:
            nt = len(str(self._tnum))
            nx = len(str(self._xnum))
            ny = len(str(self._ynum))
            nz = len(str(self._znum))
            self._name_pattern = 'ind-t%%0%dd-x%%0%dd-y%%0%dd-z%%0%dd'
            self._name_pattern = self._name_pattern % (nt, nx, ny, nz)
        return self._name_pattern

    def _unravel_index(self, ind):
        """Converts the flattened index into transform index and array coords.

        Args:
            index (int): The flattened index.

        Returns:
            tuple[int]: The tranform index and array coordinates.

        """
        shape = [self._tnum, self._xnum, self._ynum, self._znum]
        tind, xind, yind, zind = np.unravel_index(ind, shape)
        return tind, xind, yind, zind

    def get_sample_weights(self):
        """Returns the sampling weights of each patch."""
        grad = self._calc_image_grad()
        shifts = [self._calc_shift(self.patch_size[0], self._xnum),
                  self._calc_shift(self.patch_size[1], self._ynum),
                  self._calc_shift(self.patch_size[2], self._znum)]
        weights = grad[tuple(shifts)]
        weights = weights.flatten()
        weights = weights.repeat(len(self.transforms))
        return weights

    def _calc_shift(self, patch_size, image_size):
        left_shift = (patch_size - 1) // 2
        right_shift = image_size + left_shift
        shift = slice(left_shift, right_shift)
        return shift

    def _calc_image_grad(self):
        """Calculates the image graident magnitude."""
        image = self._denoise(self.image[None, None, ...])
        grad = self._calc_sobel_grad(image)
        return grad.squeeze()

    def _denoise(self, image):
        gauss_kernel = self._get_gaussian_kernel()
        padding = [s // 2 for s in gauss_kernel.shape[2:]]
        image = F.conv3d(image, gauss_kernel, padding=padding)
        return image

    def _get_gaussian_kernel(self):
        assert np.sum(np.array(self.patch_size) != 1) == 2
        sigma = 3
        length = 4 * sigma * 2 + 1
        coord = np.arange(length) - length // 2
        xcoord, ycoord = np.meshgrid(coord, coord, indexing='ij')
        kernel = np.exp(-(xcoord ** 2 + ycoord ** 2) / (2 * sigma ** 2))
        kernel = kernel / np.sum(kernel)
        kernel = torch.tensor(kernel, device=self.image.device).float()
        kernel = kernel[None, None, ..., None]
        return kernel

    def _calc_sobel_grad(self, image):
        device = self.image.device
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1,  0, -1]],
                               device=device, dtype=torch.float)
        sobel_x = sobel_x[None, None, ..., None]
        sobel_y = torch.tensor([[1,  2, 1], [0,  0,  0], [-1, -2, -1]],
                               device=device, dtype=torch.float)
        sobel_y = sobel_y[None, None, ..., None]
        grad_x = F.conv3d(image, sobel_x, padding=[1, 1, 0])
        grad_y = F.conv3d(image, sobel_y, padding=[1, 1, 0])
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return grad

    # def _calc_avg_grad(self):
    #     avg_kernel = torch.ones(self.patch_size, dtype=torch.float,
    #                             device=self.image.device)[None, None, ...]
    #     avg_kernel = avg_kernel / torch.sum(avg_kernel)
    #     padding = [ps // 2 for ps in self.patch_size]
    #     avg_grad = F.conv3d(grad, avg_kernel, padding=padding)
    #     return avg_grad.squeeze()


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
        return self._cumsum[-1]

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
