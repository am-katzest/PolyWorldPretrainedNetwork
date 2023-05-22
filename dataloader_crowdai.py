import random

import numpy as np
import torch
from pycocotools.coco import COCO
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset


class CrowdAI(Dataset):
    """CrowdAI dataset"""

    def __init__(self, images_directory):
        self.IMAGES_DIRECTORY = images_directory

        self.len = len([0])

        self.window_size = 320
        self.max_points = 256

    def loadSample(self, idx):
        image = io.imread(str(idx) + ".png")
        image = resize(
            image,
            (self.window_size, self.window_size, 3),
            anti_aliasing=True,
            preserve_range=True,
        )

        image_idx = torch.tensor([idx])
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1) / 255.0

        sample = {"image": image, "image_idx": image_idx}
        return sample

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.loadSample(idx)
        return sample
