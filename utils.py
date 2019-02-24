from skimage.transform import resize
import numpy as np
import torchvision.transforms as transforms
import torch

def scale_up(images, size):
    transform_scale = transforms.Resize(size)
    images = transform_scale(images)
    return images


