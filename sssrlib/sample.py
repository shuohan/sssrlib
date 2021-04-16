import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import random
import torch
import torch.nn.functional as F
from enum import Enum
from pathlib import Path
from improc3d import padcrop3d
from torch.utils.data._utils.collate import default_collate

from .patches import PatchesCollection


"""
'Deniose sigma: {}'.format(self.sigma),
'Weight stride: {}'.format(self.weight_stride),
'Use gradients: {}'.format(self.use_grads),

"""


class Sample:
    """Generates indices to sample patches.

    """
    def __init__(self, patches):
        self.patches = patches

    def sample_indices(self, num_samples):
        """Returns patch indices to sample.

        Args:
            num_samples (int): The number of patches to sample.

        Returns: 
            list[int]: The sampled patch indices.

        """
        raise NotImplementedError

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

    def save_figures(self, dirname):
        """Saves figures of intermediate results.

        Args:
            dirname (str): The directory to save these figures.

        """
        pass

    def _save_fig(self, dirname, image, prefix, cmap='jet'):
        while image.ndim > 3:
            image = image.squeeze(0)
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        views = {'view-yz': image[image.shape[0]//2, :, :].cpu().numpy(),
                 'view-xz': image[:, image.shape[1]//2, :].cpu().numpy(),
                 'view-xy': image[:, :, image.shape[2]//2].cpu().numpy()}
        for k, v in views.items():
            filename = Path(dirname, '%s_%s.png' % (prefix, k))
            plt.imsave(filename, v, vmin=0, vmax=0.95, cmap=cmap)


class UniformSample(Sample):
    """Samples patches uniformly.

    """
    def sample_indices(self, num_samples):
        return random.choices(range(len(self.patches)), k=num_samples)


class ProbOp(str, Enum):
    """Enum of the probability operator.

    """
    AND = 'and'
    OR = 'or'


class GradSample(Sample):
    """Samples patches weighted by image gradients.

    Attributes:
        sigma (float): The sigma (in mm) of Gaussian filter to denoise before
            calculating the probability map. Do not denoise if 0.
        use_grads (iterable[bool]): Whether to use a gradients in the order of
            x, y, z axes.
        weights_op (ProbOp): How to combine the probabilties calculated from
            difrerent gradients.

    """
    def __init__(self, patches, sigma=1, use_grads=[True, False, True],
                 weights_op=ProbOp.AND):
        super().__init__(patches)
        self.sigma = sigma
        self.use_grads = use_grads
        self.weights_op = ProbOp(weights_op)

    def _calc_image_grads(self):
        """Calculates the image graident magnitude.

        There are four steps to calculate image gradients:

            * Denoise the image
            * Interpolate the image to account for different resolution along
              each axis
            * Calculate image gradients along each axis using the Sobel operator
            * Interpolate the image back to the original shape

        """
        self._calc_denoise_kernel()
        self._denoised = self.patches.image[None, None, ...]
        if self.sigma > 0:
            padding = [s // 2 for s in self._denoise_kernel.shape[2:]]
            self._denoised = F.conv3d(self._denoised, self._denoise_kernel,
                                      padding=padding)

        mode = 'trilinear'
        scale_factor = np.array(self.patches.voxel_size)
        scale_factor = scale_factor / min(scale_factor)
        self._interpolated = F.interpolate(self._denoised, mode=mode,
                                           scale_factor=scale_factor.tolist())

        grads = self._calc_sobel_grads(self._interpolated)
        shape = self.patches.image.shape
        grads = tuple(F.interpolate(g, size=shape, mode=mode) for g in grads)
        self._gradients = tuple(g.squeeze() for g in grads)
        return self._gradients

    def _calc_denoise_kernel(self):
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

    def _calc_sobel_grads(self, image):
        """Calculate image gradient magnitude using the Sobel operator.

        """
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
            grads.append(grad)
        return grads

    def _calc_sample_weights(self):
        gradients = self._calc_image_grads()
        grads = [g for g, ug in zip(gradients, self.use_grads) if ug]
        weights = self._calc_weights_at_patch_center(grads)
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

    def sample_indices(self, num_samples):
        self.calc_sample_weights()
        indices = torch.multinomial(self._weights_flat, num_samples,
                                    replacement=True)
        return indices.cpu().tolist()

    def calc_sample_weights(self):
        """Calculates sampling weights for patches.

        Suppose the sampling probability of a patch depends on gradients along
        some of the axes, e.g., p(gx, gy). Further suppose the gradients are
        independent from each other; therefore, p(gx, gy) = p(gx) p(gy). The
        probabilty along an axis, e.g., p(gz), should be normalized across all
        patches.

        If :attr:`weights_op` is OR, use fuzzy OR to combine p(gx) and p(gy)
        instead of p(gx) p(gy) (fuzzy AND).

        """
        if not hasattr(self, '_weights_flat'):
            self._calc_sample_weights()

    def save_figures(self, dirname):
        self.calc_sample_weights()
        Path(dirname).mkdir(exist_ok=True, parents=True)
        self._save_fig(dirname, self.patches.image, 'image', cmap='gray')
        self._save_fig(dirname, self._denoise_kernel, 'denosie_kernel')
        self._save_fig(dirname, self._denoised, 'denosied', cmap='gray')
        self._save_fig(dirname, self._interpolated, 'interpolated', cmap='gray')
        for i, g in enumerate(self._gradients):
            self._save_fig(dirname, g, 'gradiant-%d' % i, cmap='jet')
        self._save_fig(dirname, self._weights, 'weights', cmap='jet')


class AvgGradSample(GradSample):
    """Uses a kernel to do weighted average within sliding windows.

    This class is use to aggregate all gradients within the patch size to the
    patch center, so the patch center can represent the amount of all gradients
    within each patch.

    Attributes:
        avg_kernel (numpy.ndarray): The kernel to aggregate gradients to the
            patch center.

    """
    def __init__(self, patches, sigma=1, use_grads=[True, False, True],
                 weights_op=ProbOp.AND, avg_kernel=None):
        super().__init__(patches, sigma=sigma, use_grads=use_grads,
                         weights_op=weights_op)
        self.avg_kernel = self._init_avg_kernel(avg_kernel)

    def _init_avg_kernel(self, kernel):
        if kernel is None:
            kernel_size = self.patches.patch_size
            kernel = np.ones(kernel_size, dtype=np.float32)
            kernel = kernel / np.sum(kernel)
        elif kernel.ndim <= 3:
            extra_dims = len(self.patches.patch_size) - kernel.ndim
            kernel = kernel[(..., ) + (None, ) * extra_dims]
            kernel = padcrop3d(kernel, self.patches.patch_size, False)
        elif kernel.ndim <= 5:
            kernel = kernel[-3:]
            kernel = padcrop3d(kernel, self.patches.patch_size, False)
        kernel = torch.tensor(kernel, dtype=self.patches.image.dtype,
                              device=self.patches.image.device)
        kernel = kernel[None, None, ...]
        return kernel

    def _calc_image_grads(self):
        """Calculates the image gradient magnitude.

        In addtion to :meth:`GradSample._calc_image_grad`, this method use an
        averaging kernel to aggregate the gradients to the patch center.

        """
        super()._calc_image_grads()
        self._avg_gradients = [self._calc_avg_grad(g) for g in self._gradients]
        return self._avg_gradients

    def _calc_avg_grad(self, grad):
        starts = [(s - 1) // 2 for s in self.patches.patch_size]
        stops = [s - 1 - ss for s, ss in zip(self.patches.patch_size, starts)]
        padding = np.array([starts[::-1], stops[::-1]]).T.flatten().tolist()
        padded_grad = torch.nn.ReplicationPad3d(padding)(grad[None, None, ...])
        avg_grad = F.conv3d(padded_grad, self.avg_kernel).squeeze(0).squeeze(0)
        return avg_grad

    def save_figures(self, dirname):
        super().save_figures(dirname)
        self._save_fig(dirname, self.avg_kernel, 'avg-kernel', cmap='jet')
        for i, g in enumerate(self._avg_gradients):
            self._save_fig(dirname, g, 'avg-gradiant-%d' % i, cmap='jet')


class SuppressWeights(Sample):
    """Suppresses non-maxima locally of sampling weights.

    Attributes:
        sample (Sample):
        kernel_size (tuple[int]): The window size to calculate local maxima.
        stride(tuple[int]): The strides between sampling weights after
            permutation.

    """
    def __init__(self, sample, kernel_size=(32, 32, 1), stride=(16, 16, 1)):
        self.sample = sample
        self.stride = stride
        self.kernel_size = kernel_size

    @property
    def patches(self):
        return self.sample.patches

    def calc_sample_weights(self):
        self.sample.calc_sample_weights()
        if not hasattr(self, '_sup_weights'):
            self._suppress_weights()
        sup_weights_flat = self._sup_weights.flatten()
        self._weights_mapping = torch.where(sup_weights_flat > 0)[0]
        self._sup_weights_flat = sup_weights_flat[self._weights_mapping]

    def _suppress_weights(self):
        """Suppresses non-maxima of weights locally.

        Args:
            weights (torch.Tensor): The weights to suppress.
            weights_stride (iterable[ind]): The distances between two adjacent
                windows.

        Returns:
            torch.Tensor: The suppressed weights.

        """
        weights = self.sample._weights[None, None, ...]
        pooled, indices = F.max_pool3d(weights, kernel_size=self.kernel_size,
                                       stride=self.stride, return_indices=True)
        self._pooled_weights = pooled.squeeze()
        unpooled = F.max_unpool3d(pooled, indices, self.kernel_size,
                                  stride=self.stride, output_size=weights.shape)
        # assert torch.equal(weights.flatten()[indices],
        #                    unpooled.flatten()[indices])
        self._sup_weights = unpooled.squeeze()

    def sample_indices(self, num_samples):
        self.calc_sample_weights()
        indices = torch.multinomial(self._sup_weights_flat, num_samples,
                                    replacement=True)
        indices = self._weights_mapping[indices]
        return indices.cpu().tolist()

    def save_figures(self, dirname):
        self.calc_sample_weights()
        self.sample.save_figures(dirname)
        self.sample._save_fig(dirname, self._pooled_weights, 'pooled-weights')
        self.sample._save_fig(dirname, self._sup_weights, 'sup-weights')


class SampleCollection(Sample):
    """Samples patches from :class:`sssrlib.patches.PatchesCollection.

    """
    def __init__(self, *sample):
        self._sample_collection = list(sample)
        patches = [s.patches for s in self._sample_collection]
        self._patches_collection = PatchesCollection(*patches)

    @property
    def patches(self):
        return self._patches_collection

    def sample_indices(self, num_samples):
        num_patches = len(self._sample_collection)
        patches_ind = random.choices(range(num_patches), k=num_samples)
        counts = np.bincount(patches_ind, minlength=num_patches)
        results = list()
        for i, num in enumerate(counts):
            if num == 0:
                continue
            patches = self._sample_collection[i]
            indices = patches.sample_indices(num)
            indices = list(zip([i] * num, indices))
            results += indices
        return results

    def save_figures(self, dirname):
        num_digits = len(str(len(self._sample_collection)))
        pattern = '%%0%dd' % num_digits
        for i, sample in enumerate(self._sample_collection):
            subdir = Path(dirname, pattern % i)
            sample.save_figures(subdir)
