import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

import pandas as pd
from PIL import Image
import numpy as np


class KittiDepthData(Dataset):
    def __init__(self, filepath_csv, transform=None):
        self.transform = transform
        self.df_filepath = pd.read_csv(filepath_csv)

    def __len__(self):
        return len(self.df_filepath)

    def __getitem__(self, index):
        img_name = self.df_filepath.loc[index, 'image']
        depth_name = self.df_filepath.loc[index, 'depth']

        image = Image.open(img_name)
        depth = Image.open(depth_name)

        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        h, w = image.size[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = F.resize(image, (new_h, new_w))
        depth = F.resize(depth, (new_h, new_w))

        return {'image': image, 'depth': depth}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        h, w = image.size[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = F.crop(image, top, left, new_h, new_w)
        depth = F.crop(depth, top, left, new_h, new_w)

        return {'image': image, 'depth': depth}


class Normalize(object):
    """
    在ToTensor之后使用
    """
    def __init__(self, mean, std, max_depth):
        self.mean = mean
        self.std = std
        self.max_depth = max_depth

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = F.normalize(image, self.mean, self.std)
        depth = (depth / 256).float() / self.max_depth

        return {'image': image, 'depth': depth}


class ToTensor(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = F.to_tensor(image)
        depth = F.to_tensor(depth)

        return {'image': image, 'depth': depth}


class RandomChannel(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        p = np.random.random()

        index = [0, 1, 2]
        if p < 0.5:
            np.random.shuffle(index)

        image = F.to_pil_image(np.array(image)[:, :, index])

        return {'image': image, 'depth': depth}


def get_train_loader(cfg):
    # TODO channel变换后与Normalization的值对应不上
    train_transform = transforms.Compose([Rescale((384, 1280)),
                                          RandomChannel(),
                                          ToTensor(),
                                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                                    max_depth=80)])

    train_set = KittiDepthData(cfg['data']['train_split'], transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)

    return train_loader


def get_valid_loader(cfg):
    # TODO channel变换后与Normalization的值对应不上
    valid_transform = transforms.Compose([Rescale((384, 1280)),
                                         ToTensor(),
                                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                                   max_depth=80)])

    valid_set = KittiDepthData(cfg['data']['valid_split'], transform=valid_transform)

    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=True, num_workers=4)

    return valid_loader


def get_test_loader(cfg):
    # TODO channel变换后与Normalization的值对应不上
    valid_transform = transforms.Compose([Rescale((384, 1280)),
                                         ToTensor(),
                                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                                   max_depth=80)])

    valid_set = KittiDepthData(cfg['data']['test_split'], transform=valid_transform)

    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=True, num_workers=4)

    return valid_loader










