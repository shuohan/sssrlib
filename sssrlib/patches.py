"""Classes to output patches from a 3D image.

TODO:
    Add :attr:`Patches.patch_gaps`.

"""
import math
import numpy as np
from enum import IntEnum
from image_processing_3d import permute3d


class Dim(IntEnum):
    """The dimension of an array.

    Attributes:
        ONE (int): Correspond to 1D arrays.
        TWO (int): Correspond to 2D arrays.
        THREE (int): Correspond to 3D arrays.

    """
    ONE = 1
    TWO = 2
    THREE = 3


class _AbstractPatches:
    """Abstract class to output patches from a 3D image.

    """
    def __getitem__(self, ind):
        """Returns a patch at index."""
        raise NotImplementedError

    def __len__(self):
        """Returns the number of possible patches."""
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

    Attributes:
        im (numpy.ndarray): The image to get patches from.
        patch_size (tuple[int]): The parsed patch size.
        x (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the x-axis in the result image.
        y (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the y-axis in the result image.
        z (image_processing_3d.Axis or int): The axis in the input ``image`` to
            permute to the z-axis in the result image.
        transforms (iterable[sssrlib.transform.Transform]): Transform the an
            output image patch.
        dim (int or Dim): The dimension of the output patch.

    Args:
        patch_size (int or tuple[int]): The size of the output patch. If
            :class:`tuple`, the length should be 3 and the last few (2 for 1D,
            and 1 for 2D, and 0 for 3D) elements should be 1.

    Raises:
        RuntimeError: Incorrect :attr:`patch_size`.

    """
    def __init__(self, im, patch_size, x=0, y=1, z=2, transforms=[], dim=1):
        self.dim = Dim(dim) if type(dim) is int else dim
        self.im, self._xinv, self._yinv, self._zinv = permute3d(im, x, y, z)
        self.patch_size = self._parse_patch_size(patch_size)
        self.transforms = transforms

        self._xnum, self._ynum, self._znum = self._init_patch_numbers()
        self._len = len(self.transforms) * self._xnum * self._ynum * self._znum

    def _parse_patch_size(self, patch_size, dim):
        """Converts :class:`int` patch size to :class:`tuple`."""
        if type(patch_size) is int:
            patch_size = (self.patch_size, ) * self.dim.value
            patch_size += (1, ) * (3 - self.dim.value)
        if self._patch_size_is_correct(patch_size)
            message = ('Incorrect "patch_size" %s. The length of "patch_size" '
                       'should be 3. If 1D/2D patches are desired, use 1s in '
                       'the last few dimensions.') % str(patch_size)
            raise RuntimeError(message)
        return patch_size

    def _patch_size_is_correct(self, patch_size):
        result = (len(patch) == 3)
        for i in range(3 - self.dim.value):
            result = result and (patch_size[-1 - i] == 1)
        return result

    def _init_patch_numbers(self):
        """Calculates the possible numbers of patches along x, y, and z."""
        return tuple(s - ps for s, ps in zip(self.im.shape, self.patch_size))

    def __len__(self):
        """Returns the number of patches"""
        return self._len

    def __getitem__(self, ind):
        """Returns a patch at index.

        Args:
            ind (int): The flattened index of the patch to return.

        Returns:
            numpy.ndarray: The returned tensor.
        
        """
        tind, x, y, z = self._unravel_index(ind)
        bbox = tuple(slice(s, s + p) for s in zip((x, y, z), self.patch_size))
        return self.transforms[tind].transform(self.im[boox])

    def _unravel_index(self, ind):
        """Converts the flattened index into transform index and array coords.

        Args:
            index (int): The flattened index.

        Returns:
            tuple[int]: The tranform index and array coordinates.

        """
        shape = [len(self.transforms), self._xnum, self._ynum, self._znum]
        xind, yind, zind = np.unravel_index(ind, shape)
        return xind, yind, zind


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
        self._nums = [len(p) for p in self.patches]
        self._cumsum = np.cumsum(self._nums)
        self._len = math.prod(self._nums)

    def __len__(self):
        return self._len

    def __getitem__(self, ind):
        pind = np.digitize(ind, self._nums)
        ind = ind - self._cumsum[pind - 1] if pind > 0 else ind
        return self.patches[pind][ind]

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
