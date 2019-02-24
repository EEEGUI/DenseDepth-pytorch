import torchvision.transforms as transforms


def scale_up(images, size):
    transform_scale = transforms.Resize(size)
    images = transform_scale(images)
    return images


