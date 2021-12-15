import os

import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image

import torch
import torch.nn as nn

from torch.utils.data import Dataset

class MaybeTransformation(nn.Module):
    """Apply a transformation with a given probability."""
    
    def __init__(self, transform, prob:float=1.0):

        super(MaybeTransformation, self).__init__()
        self.transform = transform
        self.prob = prob

    def forward(self, x):

        if torch.rand(1) < self.prob:
            x = self.transform(x)
        return x

class RandomScaleAndRotate(MaybeTransformation):
    """Randomly scale and rotate an image."""

    def __init__(self, prob:float=1.0):

        super(RandomScaleAndRotate, self).__init__(
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(1, 1.4)),
            prob
        )

class RandomColorJitter(MaybeTransformation):
    """Randomly change the brightness, contrast and saturation of an image."""

    def __init__(self, prob:float=1.0):

        super(RandomColorJitter, self).__init__(
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4),
            prob
        )

class RandomTransformation(nn.Module):
    """Apply a set of transformations randomly to an image."""

    def __init__(self, transforms:list):

        super(RandomTransformation, self).__init__()
        self.transforms = transforms

    def forward(self, x):

        # Loop through transformations
        for t in self.transforms:

            x = t(x)

        return x


class ContrastiveFrameDataset(Dataset):
    """Dataset class for contrastive learning."""

    IMG_EXT = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    def __init__(self, path, transforms:RandomTransformation, ext=".jpg", size=None):

        super(ContrastiveFrameDataset, self).__init__()

        if ext not in self.IMG_EXT:
            raise ValueError(f"{ext} is not a valid image file extension.")

        self.ext = ext
        self.path = path
        self.transforms = transforms
        self.read_mode = ImageReadMode.UNCHANGED

        self.file_list = self._parse_path(self.path)

        # Optionally limit the size of the dataset
        if size is not None:
            self.file_list = self.file_list[:size]


    def _parse_path(self, path):

        fileList = []

        for r, _, f in os.walk(path):
            fullpaths = [os.path.join(r, fl) for fl in f]    
            fileList.append(fullpaths)

        flatList = [p for paths in fileList for p in paths]
        flatList = [f for f in filter(lambda x: self.ext in x[-5:], flatList)]
        
        return flatList

    def __getitem__(self, idx):
        
        p = self.file_list[idx]
        x = read_image(p, self.read_mode)
        # We need this format for stuff.
        a = self.transforms(x).float() / 255
        b = self.transforms(x).float() / 255

        return {"a": a, "b": b}

    def __len__(self):

        return len(self.file_list)

if __name__ == "__main__":

    from torchvision.utils import save_image

    p = 0.8

    trs = RandomTransformation(
        [
            RandomScaleAndRotate(p),
            RandomColorJitter(p)
        ]
    )

    ds = ContrastiveFrameDataset(
        "data/ravens-lions",
        trs,
        ".jpg",
    )

    print(len(ds))
    print(ds[0])
    example = ds[0]
    save_image(example["a"], "example_a.jpg")
    save_image(example["b"], "example_b.jpg")
    print(ds[0]["a"].shape)