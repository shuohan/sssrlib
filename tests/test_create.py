#!/usr/bin/env python

import os
import numpy as np
from pathlib import Path

from sssrlib.patches import Patches
from sssrlib.sample import UniformSample, GradSample
from sssrlib.transform import create_transform_from_desc


def test_create():
    trans = create_transform_from_desc('rot902')
    assert trans.__str__() == 'Rotation by 180 degrees in (0, 1) plane'
    trans = create_transform_from_desc('rot903_flip02')
    assert trans.__str__() == 'Rotation by 270 degrees in (0, 1) plane, flip along (0, 2) axis'
    print('successful')


if __name__ == '__main__':
    test_create()
