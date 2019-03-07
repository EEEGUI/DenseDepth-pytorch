import os
import scipy
import skimage
import numpy as np
from pypardiso import spsolve
from PIL import Image


def img2depth(filename):
    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    return depth

def depth2img(depth, filename):
    depth = (depth * 256).astype('int16')
    depth