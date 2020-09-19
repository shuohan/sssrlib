import numpy as np


class Transform:
    """Abstract class to transform an image patch.
    
    """
    def transform(self, patch):
        """Transforms an image patch.

        Args:
            patch (numpy.ndarray): The patch to transform.

        Returns:
            numpy.ndarray: The transformed image patch.

        """
        raise NotImplementedError


class Rot90(Transform):
    """Rotates an image patch by 90 * k degrees in a specific plane.

    Attributes:
        k (int): The number of 90 degrees to rotate.
        axes (tuple[int]): The plane to rotate the array in.

    """
    def __init__(self, k=0, axes=(0, 1)):
        super().__init__()
        self.k = k
        self.axes = axes

    def transform(self, patch):
        return np.rot90(patch, self.k, self.axes)


class Flip(Transform):
    """Flips an image patch along an axis.

    Attributes:
        array (numpy.ndarray or torch.Tensor): The array to flip.
        axis (int or tuple[int]): The axis/axes to flip.

    """
    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def transform(self, patch):
        return np.flip(patch, axis=self.axis)


# import torch
# 
# def rot90(array, k=0, axes=(0, 1)):
#     """Rotates an array by 90``k`` degrees in the plane specified by ``axes``.
# 
#     Args:
#         array (numpy.ndarray or torch.Tensor): The array to rotate.
#         k (int): The number of 90 degrees to rotate.
#         axes (tuple[int]): The plane to rotate the array in.
# 
#     Returns:
#         numpy.ndarray or torch.Tensor: The rotated array.
# 
#     """
#     if type(array) is np.ndarray:
#         return np.rot90(array, k, axes)
#     elif type(array) is torch.Tensor:
#         return torch.rot90(array, k, axes)
# 
# 
# def flip(array, axis=0):
#     """Flips an array along an axis.
# 
#     Args:
#         array (numpy.ndarray or torch.Tensor): The array to flip.
#         axis (int or tuple[int]): The axis/axes to flip.
# 
#     Returns:
#         numpy.ndarray or torch.Tensor: The flipped array.
# 
#     """
#     if type(array) is np.ndarray:
#         return np.flip(arary, axis)
#     elif type(array) is torch.Tensor:
#         return torch.flip(array, axis)
