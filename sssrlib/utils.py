import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path


def save_fig(dirname, image, prefix, cmap='jet'):
    """Saves 3 views of an image.

    Args:
        dirname (str): The output directory.
        image (torch.Tensor): The 3D image to save.
        prefix (str): The basename of the file.
        cmap (str): The colormap to use for the image.

    """
    Path(dirname).mkdir(exist_ok=True, parents=True)
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


def save_fig_3d(dirname, image, prefix, affine=None, header=None):
    """Saves the image as 3D nifti file.

    Args:
        dirname (str): The output directory.
        image (torch.Tensor): The 3D image to save.
        prefix (str): The basename of the file.
        cmap (str): The colormap to use for the image.

    """
    Path(dirname).mkdir(exist_ok=True, parents=True)
    while image.ndim > 3:
        image = image.squeeze(0)
    affine = np.eye(4) if affine is None else affine
    obj = nib.Nifti1Image(image.cpu().numpy(), affine, header)
    filename = Path(dirname, prefix + '.nii.gz')
    obj.to_filename(filename)


def calc_avg_kernel(kernel_size):
    """Calculates a kernel that has the same values at all locations.

    Args:
        kernel_size (tuple[int]): The size of the kernel
    
    Returns:
        numpy.ndarray: The averaging kernel.

    """
    kernel = np.ones(kernel_size, dtype=np.float32)
    kernel = kernel / np.sum(kernel)
    return kernel


def save_sample_weights_figures(all_sample_weights, dirname, d3=True):
    """Saves figures of all :class:`SampleWeights` instances.

    Args:
        all_sample_weights (list[SampleWeights]): All instance to save figures.
        dirname (str): The figure output directory.
        d3 (bool): Save as 3D nifti files if ``True``.

    """
    num_digits = len(str(len(all_sample_weights)))
    pattern = '%%0%dd' % num_digits
    for i, sw in enumerate(all_sample_weights):
        subdir = Path(dirname, pattern % i)
        sw.save_figures(subdir, d3=d3)
