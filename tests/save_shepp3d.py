#!/usr/bin/env python

import numpy as np
from tomopy.misc.phantom import shepp3d


output = 'shepp3d.npy'
phantom = shepp3d()
np.save(output, phantom)
