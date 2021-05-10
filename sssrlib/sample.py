import numpy as np
import random
import torch
import torch.nn.functional as F
from enum import Enum
from pathlib import Path
from improc3d import padcrop3d
from torch.utils.data._utils.collate import default_collate

from .transform import create_transform_from_desc
from .patches import PatchesCollection
from .utils import save_fig, calc_conv_padding


class ProbOp(str, Enum):
    """Enum of the probability operator.

    """
    AND = 'and'
    OR = 'or'


class _SampleWeights:
    """Abstract class to calculate sample weights.

    """
    def __init__(self, patches):
        self.patches = patches

    @property
    def weights_flat(self):
        """Returns the flattened weights."""
        raise NotImplementedError

    @property
    def weights(self):
        """Returns the weights in its 3D shape."""
        raise NotImplementedError

    def save_figures(self, dirname, d3=True):
        """Saves figures of intermediate results.

        Args:
            dirname (str): The directory to save these figures.
            d3 (bool): Save the image as 3D nifti file if ``True``.

        """
        raise NotImplementedError


class ImageGradients:
    """Calculates image gradient magnitude.

    * Denoise the image
    * Interpolate the image to account for different resolution along
      each axis
    * Calculate image gradients along each axis using the Sobel operator

    Args:
        patches (sssrlib.patches.AbstractPatches): The patches to calculate
            image gradients for.
        sigma (float): The sigma (in mm) of Gaussian filter to denoise before
            calculating the probability map. Do not denoise if 0.

    """
    def __init__(self, patches, sigma=1):
        self.patches = patches
        self.sigma = sigma
        self._interp_mode = 'trilinear'
        self._init_denoise_kernel()
        self._denoise_image()
        self._interpolate_image()
        self._calc_image_grads()

    def __str__(self):
        return 'Sigma: {}'.format(self.sigma)

    @property
    def denoised(self):
        """Returns the denoised image."""
        return self._denoised.squeeze(0).squeeze(0)

    @property
    def interpolated(self):
        """Returns the isotropic interpolated from the denoised image."""
        return self._interpolated.squeeze(0).squeeze(0)

    @property
    def gradients(self):
        """Returns a list of image gradient magnitude along three axes."""
        return tuple(g.squeeze(0).squeeze(0) for g in self._gradients)

    def save_figures(self, dirname, d3=True):
        """Saves figures of intermediate results.

        Args:
            dirname (str): The directory to save these figures.
            d3 (bool): Save the image as 3D nifti file if ``True``.

        """
        save_fig(dirname, self._denoise_kernel, 'denosie_kernel', d3=d3)
        save_fig(dirname, self._denoised, 'denoise', cmap='gray', d3=d3)
        save_fig(dirname, self._interpolated, 'interpolate', cmap='gray', d3=d3)
        for i, g in enumerate(self._gradients):
            save_fig(dirname, g, 'gradiant-%d' % i, d3=d3)

    def _init_denoise_kernel(self):
        """Use a 3D Gaussian function as the denoising kernel.
        
        Note:
            The sigma along each axis depends on the relative voxel size. A
            larger voxel size needs a smaller sigma to introduce similar amount
            of blur in each axis.

        """
        length = 4 * self.sigma * 2 + 1
        coord = np.arange(length) - length // 2
        grid = np.meshgrid(coord, coord, coord, indexing='ij')
        sigmas = self.sigma / np.array(self.patches.voxel_size)
        kernels = [np.exp(-(g**2) / (2 * s**2)) for g, s in zip(grid, sigmas)]
        kernel = np.prod(kernels, axis=0)
        kernel = kernel / np.sum(kernel)
        kernel = torch.tensor(kernel, device=self.patches.image.device).float()
        kernel = kernel[None, None, ...]
        self._denoise_kernel = kernel.to(device=self.patches.image.device)

    def _denoise_image(self):
        self._denoised = self.patches.image[None, None, ...]
        if self.sigma > 0:
            padding = calc_conv_padding(self._denoise_kernel.shape[2:])
            padded = torch.nn.ReplicationPad3d(padding)(self._denoised)
            self._denoised = F.conv3d(padded, self._denoise_kernel)

    def _interpolate_image(self):
        scale_factor = np.array(self.patches.voxel_size)
        scale_factor = scale_factor / min(scale_factor)
        self._interpolated = F.interpolate(self._denoised,
                                           mode=self._interp_mode,
                                           scale_factor=scale_factor.tolist())

    def _calc_image_grads(self):
        grads = self._calc_sobel_grads(self._interpolated)
        shape = self.patches.image.shape
        self._gradients = [F.interpolate(g, size=shape, mode=self._interp_mode)
                           for g in grads]

    def _calc_sobel_grads(self, image):
        """Calculate image gradient magnitude using the Sobel operator."""
        sz = torch.stack((torch.tensor([[0, 0,  0], [1, 0, -1], [0, 0,  0]]),
                          torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
                          torch.tensor([[0, 0,  0], [1, 0, -1], [0, 0,  0]])))
        sz = sz.float().to(image.device)
        sx = sz.permute([2, 0, 1])
        sy = sz.permute([1, 2, 0])
        grads = list()
        for s in (sx, sy, sz):
            padded = torch.nn.ReplicationPad3d(1)(image)
            grad = F.conv3d(padded, s[None, None, ...])
            grad = torch.abs(grad)
            grads.append(grad)
        return grads


class Aggregate:
    """Weighted average within sliding windows.

    Args:
        kernel (numpy.ndarray): The kernel to aggregate values within sliding
            windows.

    """
    def __init__(self, kernel, images):
        self.kernel = torch.tensor(kernel)[None, None, ...]
        self.images = images
        self._agg_images = [self._aggregate(im) for im in self.images]

    @property
    def agg_images(self):
        return self._agg_images

    def _aggregate(self, image):
        padding = calc_conv_padding(self.kernel.shape[2:])
        pad = torch.nn.ReplicationPad3d(padding)
        image_padded = pad(image[None, None, ...])
        kernel = self.kernel.to(image)
        result = F.conv3d(image_padded, kernel).squeeze(0).squeeze(0)
        return result

    def __str__(self):
        return 'Aggregating kernel shape: {}'.format(self.kernel.shape)

    def save_figures(self, dirname, d3=True):
        save_fig(dirname, self.kernel, 'agg-kernel', d3=d3)
        for i, im in enumerate(self._agg_images):
            save_fig(dirname, im, 'agg-image-%d' % i, d3=d3)


class SampleWeights(_SampleWeights):
    """Samples patches weighted by image gradients.

    Suppose the sampling probability of a patch depends on gradients along some
    of the axes, e.g., p(gx, gy), and the gradients are independent from
    each other. Therefore, p(gx, gy) = p(gx) p(gy). The probabilty along an
    axis, e.g., p(gz), should be normalized across all patches.

    If :attr:`weights_op` is OR, use fuzzy OR to combine p(gx) and p(gy)
    instead of p(gx) p(gy) (fuzzy AND).

    Attributes:
        patches (sssrlib.patches.AbstractPatches): The patches to calculate
            weights for.
        weight_maps (tuple[torch.Tensor]): Multiple weight maps used to
            calculate the sampling weights.
        weights_op (ProbOp): How to combine the probabilties calculated from
            difrerent weight maps.

    """
    def __init__(self, patches, weight_maps, weights_op=ProbOp.OR):
        super().__init__(patches)
        self.weight_maps = weight_maps
        self.weights_op = ProbOp(weights_op)
        self._calc_sample_weights()

    @property
    def weights(self):
        return self._weights

    @property
    def weights_flat(self):
        return self._weights_flat

    def save_figures(self, dirname, d3=True):
        save_fig(dirname, self._weights, 'weights', d3=d3)

    def __str__(self):
        return 'Weights operator: {}'.format(self.weights_op)

    def _calc_sample_weights(self):
        weights = self._calc_weights_at_patch_center(self.weight_maps)
        weights = [w / torch.sum(w) for w in weights]
        self._weights = self._combine_weights(weights)
        self._weights_flat = self._weights.flatten()

    def _calc_weights_at_patch_center(self, weights):
        """Crops the gradiant maps to align them with the patch centers.

        For example, the first element in the gradient map is the corner of the
        first image patch. To use the center value as the first value in the
        maps, they should be cropped by half size of the patch.

        """
        patch_size = self.patches.patch_size
        shifts = [self._calc_shift(patch_size[0], self.patches.xnum),
                  self._calc_shift(patch_size[1], self.patches.ynum),
                  self._calc_shift(patch_size[2], self.patches.znum)]
        weights = [w[tuple(shifts)] for w in weights]
        return weights

    def _calc_shift(self, patch_size, image_size):
        left_shift = (patch_size - 1) // 2
        right_shift = image_size + left_shift
        shift = slice(left_shift, right_shift)
        return shift

    def _combine_weights(self, weights):
        """Combines normalized weights using fuzzy or/and operator."""
        if self.weights_op is ProbOp.AND:
            weights = torch.prod(torch.stack(weights), axis=0)
        elif self.weights_op is ProbOp.OR:
            rev_weights = [1.0 - w for w in weights]
            weights = 1.0 - torch.prod(torch.stack(rev_weights), axis=0)
        return weights


class SuppressWeights(SampleWeights):
    """Wrapper class to suppresse non-maxima locally of sampling weights.

    Since this class assigns zeros to non-maxima, it extracts the indices of the
    non-zeros and returns a "squeezed" array of weights. When accessing a
    particular weight, it first outputs the index in the "squeezed" weights
    :meth:`weights_flat` then uses :meth:`weights_mapping` to get the index in
    the array before "squeezing".

    Attributes:
        sample_weights (SampleWeights): The instance to wrap around.
        kernel_size (tuple[int]): The window size to calculate local maxima.
        stride(tuple[int]): The strides between sampling weights after
            permutation.

    """
    def __init__(self, sample_weights, kernel_size, stride):
        self.sample_weights = sample_weights
        self.kernel_size = kernel_size
        self.stride = stride
        self._suppress_weights()

    @property
    def patches(self):
        return self.sample_weights.patches

    @property
    def weights(self):
        """Returns the suppressed weights in the original shape."""
        return self._sup_weights

    @property
    def weights_flat(self):
        """Returns the flattened suppressed weights."""
        return self._sup_weights_flat

    @property
    def pooled_weights(self):
        """Returns the pooled weights."""
        return self._pooled_weights

    @property
    def weights_mapping(self):
        """Returns the indices in the original shape."""
        return self._weights_mapping

    def save_figures(self, dirname, d3=True):
        self.sample_weights.save_figures(dirname, d3=d3)
        save_fig(dirname, self._pooled_weights, 'pooled-weights', d3=d3)
        save_fig(dirname, self._sup_weights, 'sup-weights', d3=d3)

    def __str__(self):
        message = [self.sample_weights.__str__(),
                   'Stride: {}'.format(self.stride),
                   'Kernel size: {}'.format(self.kernel_size)]
        return '\n'.join(message)

    def _suppress_weights(self):
        """Suppresses non-maxima of weights locally.

        Args:
            weights (torch.Tensor): The weights to suppress.
            weights_stride (iterable[ind]): The distance0 between two adjacent
                windows.

        Returns:
            torch.Tensor: The suppressed weights.

        """
        weights = self.sample_weights.weights[None, None, ...]
        pooled, indices = F.max_pool3d(weights, kernel_size=self.kernel_size,
                                       stride=self.stride, return_indices=True)
        self._pooled_weights = pooled.squeeze(0).squeeze(0)
        unpooled = F.max_unpool3d(pooled, indices, self.kernel_size,
                                  stride=self.stride, output_size=weights.shape)
        self._sup_weights = unpooled.squeeze(0).squeeze(0)
        sup_weights_flat = self._sup_weights.flatten()
        self._weights_mapping = torch.where(sup_weights_flat > 0)[0]
        self._sup_weights_flat = sup_weights_flat[self._weights_mapping]


class Sampler:
    """Generates indices to sample patches.

    Attributes:
        patches (sssrlib.patches.AbstractPatches): The patches to sample from.
        weights (torch.Tensor): The 1D sampling weights (flattened) of patches.
            If ``None``, sample the patches uniformly.
        weights_mapping (torch.Tensor): It is used to index the weights in the
            array before "squeezing". If ``None``, use the index directly in
            :attr:`weights`.

    """
    def __init__(self, patches, weights_flat=None, weights_mapping=None):
        self.patches = patches
        self.weights_flat = weights_flat
        self.weights_mapping = weights_mapping

    def sample_indices(self, num_samples):
        """Returns patch indices to sample.

        Args:
            num_samples (int): The number of patches to sample.

        Returns: 
            list[int]: The sampled patch indices.

        """
        if self.weights_flat is None:
            return random.choices(range(len(self.patches)), k=num_samples)
        else:
            indices = torch.multinomial(self.weights_flat, num_samples,
                                        replacement=True)
            if self.weights_mapping is not None:
                indices = self.weights_mapping[indices]
            return indices.cpu().tolist()

    def get_patches(self, indices):
        """Returns patches with indices.

        Args:
            indices (list): The patch indices.

        Returns:
            torch.Tensor or sssrlib.patches.NamedData: The selected patches.

        """
        result = list()
        for ind in indices:
            result.append(self.patches[ind])
        result = default_collate(result)
        return result


class SamplerCollection(Sampler):
    """Samples patches from :class:`sssrlib.patches.PatchesCollection.

    Args:
        sampler (Sampler): Samples patches.

    """
    def __init__(self, *sampler):
        self._sampler_collection = list(sampler)
        patches = [s.patches for s in self._sampler_collection]
        self._patches_collection = PatchesCollection(*patches)

    @property
    def patches(self):
        """It is used when calling parent method :meth:`get_patches`."""
        return self._patches_collection

    def sample_indices(self, num_samples):
        num_patches = len(self._sampler_collection)
        patches_ind = random.choices(range(num_patches), k=num_samples)
        counts = np.bincount(patches_ind, minlength=num_patches)
        results = list()
        for i, num in enumerate(counts):
            if num == 0:
                continue
            patches = self._sampler_collection[i]
            indices = patches.sample_indices(num)
            indices = list(zip([i] * num, indices))
            results += indices
        random.shuffle(results)
        return results
