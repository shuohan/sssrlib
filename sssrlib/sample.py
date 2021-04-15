import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import random
import torch
import torch.nn.functional as F
from enum import Enum
from pathlib import Path


class Sample:
    """Generates indices to sample patches.

    """
    def __init__(self, patches, num_samples):
        self.patches = patches
        self.num_samples = num_samples

    def sample(self):
        """Returns patch indices to sample."""
        raise NotImplementedError


class UniformSample(Sample):
    """Samples patches uniformly.

    """
    def sample(self):
        return random.choices(range(len(self.patches)), k=self.num_samples)


class ProbOp(str, Enum):
    """Enum of the probability operator.

    """
    AND = 'and'
    OR = 'or'


class GradSample(Sample):
    """Samples patches weighted by image gradients.

    """
    def __init__(self, patches, num_sel, sigma=1, voxel_size=(1, 1, 1),
                 use_grads=[True, False, True], weights_op=ProbOp.AND):
        super().__init__(patches, num_sel)
        self.sigma = 3
        self.voxel_size = np.array(voxel_size) / min(voxel_size)
        self.use_grads = use_grads
        self.weights_op = ProbOp(weights_op)
        self._calc_sample_weights()

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
        scale_factor = self.voxel_size.tolist()
        self._interpolated = F.interpolate(self._denoised, mode=mode,
                                           scale_factor=scale_factor)

        grads = self._calc_sobel_grads(self._interpolated)
        shape = self.patches.image.shape
        grads = tuple(F.interpolate(g, size=shape, mode=mode) for g in grads)
        self._gradients = tuple(g.squeeze() for g in grads)

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
        sigmas = self.sigma / self.voxel_size
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
        """Calculates sampling weights for patches.

        Suppose the sampling probability of a patch depends on gradients along
        some of the axes, e.g., p(gx, gy). Further suppose the gradients are
        independent from each other; therefore, p(gx, gy) = p(gx) p(gy). The
        probabilty along an axis, e.g., p(gz), should be normalized across all
        patches.

        """
        self._calc_image_grads()
        grads = [g for g, ug in zip(self._gradients, self.use_grads) if ug]
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

    def sample(self):
        return torch.multinomial(self._weights_flat, self.num_samples,
                                 replacement=True)

    def save_figures(self, dirname):
        """Saves figures of intermediate results.

        Args:
            dirname (str): The directory to save these figures.

        """
        Path(dirname).mkdir(exist_ok=True, parents=True)
        self._save_fig(dirname, self.patches.image, 'image', cmap='gray')
        self._save_fig(dirname, self._denoise_kernel, 'denosie_kernel')
        self._save_fig(dirname, self._denoised, 'denosied', cmap='gray')
        self._save_fig(dirname, self._interpolated, 'interpolated', cmap='gray')
        for i, g in enumerate(self._gradients):
            self._save_fig(dirname, g, 'gradiant-%d' % i, cmap='jet')
        self._save_fig(dirname, self._weights, 'weights', cmap='jet')

    def _save_fig(self, dirname, image, prefix, cmap='jet'):
        image = (image.squeeze() - image.min()) / (image.max() - image.min())
        views = {'view-yz': image[image.shape[0]//2, :, :].cpu().numpy(),
                 'view-xz': image[:, image.shape[1]//2, :].cpu().numpy(),
                 'view-xy': image[:, :, image.shape[2]//2].cpu().numpy()}
        for k, v in views.items():
            filename = Path(dirname, '%s_%s.png' % (prefix, k))
            plt.imsave(filename, v, vmin=0, vmax=0.95, cmap=cmap)


"""

    def _calc_avg_grad(self, grad):
        avg_kernel = torch.ones(self.patch_size, dtype=grad.dtype,
                                device=grad.device)[None, None, ...]
        avg_kernel = avg_kernel / torch.sum(avg_kernel)
        padding = [ps // 2 for ps in self.patch_size]
        avg_grad = F.conv3d(grad, avg_kernel, padding=padding)
        return avg_grad

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
"""
