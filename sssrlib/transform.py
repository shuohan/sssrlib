import re
import numpy as np
import torch
from enum import Enum


class TransformType(str, Enum):
    """Enum of the transforms."""
    FLIP = 'flip'
    ROT90 = 'rot90'


def create_transform_from_desc(trans_args):
    """Creates a transform from a string description.

    Multiple transforms are joined with "_", for example, "rot901_flip01". For
    rot90, the last digit is k, for example, "rot902" is rot90 with k = 2. For
    flip, the digits are flipping axes, for example, "flip02" is flip with
    axis = (0, 2).

    Args:
        trans_args (str): The transform description.

    Returns:
        Transform: The created transform.

    """
    trans_args = trans_args.split('_')
    results = list()
    for args in trans_args:
        if args.startswith('rot90'):
            trans = ('rot90', {'k': int(re.sub('^rot90', '', args))})
        elif args.startswith('flip'):
            axes = tuple(int(d) for d in re.sub('^flip', '', args))
            trans = ('flip', {'axis': axes})
        results.append(trans)
    return create_transform(*results)


def create_transform(*trans_args):
    """Creates a transform.

    Args:
        trans_args (iterable): Describes the transform. For single transform,
            use ('rot90', {'k': 1}). For muliple transforms (compose), use
            ('rot90', {'k': 2}), ('flip', {'axis': (1, )}).

    Returns:
        Transform: The created transform.

    """
    results = list()
    for args in trans_args:
        trans_type = TransformType(args[0])
        if trans_type is TransformType.FLIP:
            trans = Flip(**args[1])
        elif trans_type is TransformType.ROT90:
            trans = Rot90(**args[1])
        results.append(trans)

    if len(results) > 1:
        return Compose(*results)
    else:
        return results[0]


class Transform:
    """Abstract class to transform an image patch.

    """
    def __call__(self, patch, channel_dim=True):
        """Calls the method :meth:`transform`."""
        return self.transform(patch, channel_dim=channel_dim)

    def transform(self, patch, channel_dim=True):
        """Transforms an image patch.

        Args:
            patch (numpy.ndarray): The patch to transform.
            channel_dim (bool): Whether the channel dim has been expanded for
                this patch.

        Returns:
            numpy.ndarray: The transformed image patch.

        """
        patch = patch.squeeze(0) if channel_dim else patch
        patch = self._transform(patch)
        patch = patch[None, ...] if channel_dim else patch
        return patch

    def _transform(self, patch):
        raise NotImplementedError

    def get_name(self):
        """Returns the name of the transform."""
        raise NotImplementedError


class Rot90(Transform):
    """Rotates an image patch by 90 * k degrees in a specific plane.

    Warning:
        Currently, :attr:`axes` is fixed to (0, 1) which is rotation in the x-y
        plane. The allowed :attr:`k` is in {1, 2, 3}.

    Attributes:
        k (int): The number of 90 degrees to rotate.
        axes (tuple[int]): The plane to rotate the array in.

    """
    def __init__(self, k=1, axes=(0, 1)):
        super().__init__()
        assert k in {1, 2, 3}
        self.k = k
        # self.axes = axes
        self.axes = (0, 1)

    def _transform(self, patch):
        return rot90(patch, self.k, self.axes)

    def __str__(self):
        message = 'Rotation by %d degrees in %s plane'
        return message % ((self.k * 90), str(self.axes))

    def get_name(self):
        return 'rot%d' % (self.k * 90)


class Flip(Transform):
    """Flips an image patch along an axis.

    Attributes:
        array (numpy.ndarray or torch.Tensor): The array to flip.
        axis (int or tuple[int]): The axis/axes to flip.

    """
    def __init__(self, axis=(0, )):
        super().__init__()
        self.axis = axis
        # self.axis = (0, )

    def _transform(self, patch):
        return flip(patch, axis=self.axis)

    def __str__(self):
        return 'flip along %s axis' % str(self.axis)

    def get_name(self):
        return 'flip'


class Compose(Transform):
    """Composes multiple transforms.

    Attributes:
        transforms (tuple[Transform]): The transforms to compose.

    """
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = transforms

    def transform(self, patch, channel_dim=True):
        result = patch
        for trans in self.transforms:
            result = trans.transform(result, channel_dim=channel_dim)
        return result

    def __str__(self):
        result = ', '.join([t.__str__() for t in self.transforms])
        return result

    def get_name(self):
        result = '-'.join([t.get_name() for t in self.transforms])
        return result


def rot90(array, k=0, axes=(0, 1)):
    """Rotates an array by 90``k`` degrees in the plane specified by ``axes``.

    Args:
        array (numpy.ndarray or torch.Tensor): The array to rotate.
        k (int): The number of 90 degrees to rotate.
        axes (tuple[int]): The plane to rotate the array in.

    Returns:
        numpy.ndarray or torch.Tensor: The rotated array.

    """
    if type(array) is np.ndarray:
        return np.rot90(array, k, axes)
    elif type(array) is torch.Tensor:
        return torch.rot90(array, k, axes)


def flip(array, axis=0):
    """Flips an array along an axis.

    Args:
        array (numpy.ndarray or torch.Tensor): The array to flip.
        axis (int or tuple[int]): The axis/axes to flip.

    Returns:
        numpy.ndarray or torch.Tensor: The flipped array.

    """
    if type(array) is np.ndarray:
        return np.flip(array, axis)
    elif type(array) is torch.Tensor:
        return torch.flip(array, axis)
