#!/usr/bin/env python

import torch
from torch.nn.functional import interpolate


def test_interp():
    image = torch.rand((4, 1, 128, 128)).float()
    image_cuda = image.cuda()

    scale_factor = 3.14
    mode = 'bicubic'
    image_interp = interpolate(image, scale_factor=scale_factor, mode=mode)
    image_cuda_interp = interpolate(image_cuda, scale_factor=scale_factor, mode=mode)
    assert torch.allclose(image_interp.cuda(), image_cuda_interp)


if __name__ == '__main__':
    test_interp()
