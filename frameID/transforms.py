import os

import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image

import torch
import torch.nn as nn

from torch.utils.data import Dataset

class MaybeTransformation(nn.Module):
    """Apply a transformation with a given probability."""
    
    def __init__(self, transform, args:dict, prob:float=1.0):

        super(MaybeTransformation, self).__init__()
        self.transform = transform
        self.prob = prob
        self.args = args

    def forward(self, x):

        if torch.rand(1) < self.prob:
            x = self.transform(x, **self.args)
        return x


def random_adjust_brightness(batch:torch.Tensor, range):

    n_img = batch.shape[0]

    # Brightness factors must have same number of dimensions as batch of images.
    brightness_factors = range[0] + (torch.rand([n_img, 1, 1, 1]) * (range[1] - range[0]))
    zero_tensor = torch.zeros_like(batch)

    return torch.clamp((brightness_factors * batch) + ((1 - brightness_factors) * zero_tensor), 0.0, 1.0)

def random_adjust_contrast(batch:torch.Tensor, range):

    n_img = batch.shape[0]

    contrast_factors = range[0] + (torch.rand([n_img, 1, 1, 1]) * (range[1] - range[0]))
    img_means = torch.mean(transforms.functional.rgb_to_grayscale(batch), dim=(-3, -2, -1), keepdim=True)

    return torch.clamp(contrast_factors * batch + (1 - contrast_factors) * img_means, 0.0, 1.0)

class RandomTransformation(nn.Module):
    """Apply a set of transformations randomly to a batch."""

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
        x = x.float() / 255

        return x

    def __len__(self):

        return len(self.file_list)


if __name__ == "__main__":

    from torchvision.utils import save_image
    from torch.utils.data import DataLoader

    p = 0.8q

    trs = RandomTransformation(
        [
            MaybeTransformation(random_adjust_brightness, {"range": [0.5, 1.5]}, p),
            MaybeTransformation(random_adjust_contrast, {"range": [0.5, 1.5]}, p),
        ]
    )

    ds = ContrastiveFrameDataset("data/ravens-lions", ".jpg")

    dl = DataLoader(ds, 64)

    batch = next(iter(dl))

    print(batch.shape)

    a = trs(batch)

    print(a.shape)

    save_image(a[0, :], "example_a.jpg")

    b = trs(batch)
    save_image(b[0, :], "example_b.jpg")