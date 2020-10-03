import numpy as np
import torch


def create_rot_flip():
    """Create transforms combining :class:`Rot90` and :class:`Flip`.

    Returns:
        list[Transform]: The created transforms.

    """
    transforms = list()
    transforms.append(Identity())
    transforms.append(Compose(Identity(), Flip()))
    for k in [1, 2, 3]:
        transforms.append(Rot90(k=k))
        transforms.append(Compose(Rot90(k=k), Flip()))
    return transforms


class Transform:
    """Abstract class to transform an image patch.

    """
    def __call__(self, patch):
        """Calls the method :meth:`transform`."""
        return self.transform(patch)

    def transform(self, patch):
        """Transforms an image patch.

        Args:
            patch (numpy.ndarray): The patch to transform.

        Returns:
            numpy.ndarray: The transformed image patch.

        """
        raise NotImplementedError


class Identity(Transform):
    """Does not change the input patch.

    """
    def transform(self, patch):
        return patch

    def __str__(self):
        return 'identity transform'


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

    def transform(self, patch):
        return rot90(patch, self.k, self.axes)

    def __str__(self):
        message = 'rotation by %d degrees in %s plane'
        return message % ((self.k * 90), str(self.axes))


class Flip(Transform):
    """Flips an image patch along an axis.

    Warning:
        Currently, :attr:`axis` is fixed to 0 which is flipping along the
        x-axis.

    Attributes:
        array (numpy.ndarray or torch.Tensor): The array to flip.
        axis (int or tuple[int]): The axis/axes to flip.

    """
    def __init__(self, axis=0):
        super().__init__()
        # self.axis = axis
        self.axis = (0, )

    def transform(self, patch):
        return flip(patch, axis=self.axis)

    def __str__(self):
        return 'flip along %s axis' % str(self.axis)


class Compose(Transform):
    """Composes multiple transforms.

    Attributes:
        transforms (tuple[Transform]): The transforms to compose.

    """
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = transforms

    def transform(self, patch):
        result = patch
        for trans in self.transforms:
            result = trans.transform(result)
        return result

    def __str__(self):
        result = ', '.join([t.__str__() for t in self.transforms])
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
        return np.flip(arary, axis)
    elif type(array) is torch.Tensor:
        return torch.flip(array, axis)
