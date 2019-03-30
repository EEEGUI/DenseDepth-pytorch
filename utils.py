import os
import torch
import logging
import datetime
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


def scale_up(images, size):
    transform_scale = transforms.Resize(size)
    images = transform_scale(images)
    return images

def img2depth(filename):
    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    return depth


def depth2img(depth, filename):
    depth = (depth * 256).astype('int16')
    depth_png = Image.fromarray(depth)
    depth_png.save(filename)


def get_logger(logdir):
    logger = logging.getLogger("DDNet")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
