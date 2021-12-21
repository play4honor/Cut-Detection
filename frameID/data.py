import os
import json

import torch
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image

from torch.utils.data import Dataset

import cv2


def open_video(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, {
        "fps": fps,
        "length": length,
        "width": width,
        "height": height,
    }


class ContrastiveFrameDataset(Dataset):
    """Dataset class for contrastive learning."""

    IMG_EXT = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )

    def __init__(self, path, trs: transforms.Compose, ext=".jpg", size=None):

        super(ContrastiveFrameDataset, self).__init__()

        if ext not in self.IMG_EXT:
            raise ValueError(f"{ext} is not a valid image file extension.")

        self.ext = ext
        self.path = path
        self.read_mode = ImageReadMode.UNCHANGED
        self.trs = trs

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

        # We apply each transformation twice.

        return {
            "x": x,
            "x_t1": self.trs(x),
            "x_t2": self.trs(x),
        }

    def __len__(self):

        return len(self.file_list)


class SupervisedFrameDataset(Dataset):

    IMG_EXT = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )

    # This is horrible but whatever.
    lab_enum = {"A22": 0, "EZ": 1, "B": 2}

    def __init__(self, path, ext=".jpg", size=None, labs_file: str = "labels.json"):

        super(SupervisedFrameDataset, self).__init__()

        if ext not in self.IMG_EXT:
            raise ValueError(f"{ext} is not a valid image file extension.")

        self.ext = ext
        self.path = path
        self.read_mode = ImageReadMode.UNCHANGED

        with open(f"{self.path}/{labs_file}", "r") as f:
            labs = json.load(f)

        # Make a dictionary of idx: (image_idx, image_label)
        self.labels = {
            idx: (int(item[0]), self.lab_enum[item[1]])
            for idx, item in enumerate(labs.items())
        }

        self.file_list = self._parse_path(self.path)

    def _parse_path(self, path):

        fileList = []

        for r, _, f in os.walk(path):
            fullpaths = [os.path.join(r, fl) for fl in f]
            fileList.append(fullpaths)

        flatList = [p for paths in fileList for p in paths]
        flatList = [f for f in filter(lambda x: self.ext in x[-5:], flatList)]

        return flatList

    def __getitem__(self, idx):

        file_idx, lab = self.labels[idx]

        p = self.file_list[file_idx]
        x = read_image(p, self.read_mode)
        # We need this format for stuff.
        x = x.float() / 255

        return {"x": x, "y": torch.tensor([lab], dtype=torch.int64)}

    def __len__(self):

        return len(self.labels)


if __name__ == "__main__":

    from torchvision.utils import save_image, make_grid
    from torch.utils.data import DataLoader

    ds = SupervisedFrameDataset("data/ravens-lions", ext=".jpg")
    print(len(ds))

    dl = DataLoader(ds, 64)

    batch = next(iter(dl))

    print(batch)
