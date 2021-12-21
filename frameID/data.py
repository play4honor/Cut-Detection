import os
import csv

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
            self.file_list = self.file_list[: min(size, len(self.file_list))]

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
    lab_enum = {"a22": 0, "ez": 1, "b": 2}

    def __init__(self, path, labs_file: str, ext=".jpg", size=None):

        super(SupervisedFrameDataset, self).__init__()

        if ext not in self.IMG_EXT:
            raise ValueError(f"{ext} is not a valid image file extension.")

        self.ext = ext
        self.path = path
        self.read_mode = ImageReadMode.UNCHANGED

        # Turn the contents of the csv into a tensor. This may be a bad idea.
        with open(f"{self.path}/{labs_file}", "r") as f:
            lab_reader = csv.reader(f, delimiter=",")
            raw_ranges = [(int(row[0]), row[1]) for row in lab_reader]
            self.label_ranges = torch.stack(
                (
                    torch.tensor([row[0] for row in raw_ranges], dtype=torch.int32),
                    torch.tensor(
                        [self.lab_enum[row[1]] for row in raw_ranges], dtype=torch.int32
                    ),
                ),
                dim=0,
            )

        self.file_list = self._parse_path(self.path)

        # Optionally limit the size of the dataset
        if size is not None:
            self.file_list = self.file_list[: min(size, len(self.file_list))]

    def _parse_path(self, path):

        fileList = []

        for r, _, f in os.walk(path):
            fullpaths = [os.path.join(r, fl) for fl in f]
            fileList.append(fullpaths)

        flatList = [p for paths in fileList for p in paths]
        flatList = [f for f in filter(lambda x: self.ext in x[-5:], flatList)]

        return flatList

    def _get_label(self, idx):

        pos = torch.searchsorted(self.label_ranges[0, :], idx, right=True).item()
        return self.label_ranges[1, pos - 1]

    def __getitem__(self, idx):

        label = self._get_label(idx)

        p = self.file_list[idx]
        x = read_image(p, self.read_mode)
        # We need this format for stuff.
        x = x.float() / 255

        return {"x": x, "y": label.long()}

    def __len__(self):

        return len(self.file_list)


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    ds = SupervisedFrameDataset(
        "data/browns-ravens", labs_file="frames.csv", ext=".jpg"
    )
    print(len(ds))
    print(ds.file_list[0:10])
    print(ds.file_list[73412])

    dl = DataLoader(ds, 64, shuffle=True)

    batch = next(iter(dl))

    print(batch)
