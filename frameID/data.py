import os
import csv

import torch
import torchvision.transforms as transforms
from torchvision.io import ImageReadMode, read_image

from torch.utils.data import Dataset, IterableDataset

import cv2


def open_video(video_path):
    """
    Simple function to open a video with opencv and get some information in
    actually usable form.
    """

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
    """Dataset class for basic classification task."""

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
        """Gets a label for a given frame number."""

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


class VideoDataset(IterableDataset):
    """
    Dataset class for reading frames directly from video. Intended to be used
    to provide data to a trained model.
    """

    def __init__(self, file_path, resize=None):

        super().__init__()

        self.cap, self.video_info = open_video(file_path)

        # Calculate the correct dimensions
        if resize is not None:

            self.new_width = resize
            self.new_height = int(
                self.video_info["height"] * (self.new_width / self.video_info["width"])
            )
        else:

            self.new_width = None
            self.new_height = None

    def __iter__(self):
        return self

    def __next__(self):

        ret, frame = self.cap.read()

        if not ret:
            raise StopIteration

        if self.new_width is not None:

            frame = cv2.resize(
                frame, (self.new_width, self.new_height), interpolation=cv2.INTER_LINEAR
            )

        # openCV and torch don't remotely represent images the same way.
        frame = (
            torch.flip(torch.tensor(frame, dtype=torch.float).permute(2, 0, 1), (0,))
            / 255
        )

        return frame

    def __len__(self):
        """Note iterables don't necessarily have len, but this one does."""
        return self.video_info["length"]


class FrameSequenceDataset(SupervisedFrameDataset):
    def __init__(self, seq_length: int, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.seq_length = seq_length

    def __getitem__(self, idx):

        idx_range = [idx, min(idx + self.seq_length, len(self))]

        frames = []

        for i in range(*idx_range):

            frames.append(super().__getitem__(i))

        frame_tensor = torch.stack([f["x"] for f in frames])
        label_tensor = torch.stack([f["y"] for f in frames])

        ts = frame_tensor.shape

        if idx + self.seq_length > len(self):

            excess = (idx + self.seq_length) - len(self)

            frame_tensor = torch.cat(
                [frame_tensor, torch.zeros(excess, ts[1], ts[2], ts[3])]
            )
            label_tensor = torch.cat([label_tensor, torch.zeros(excess)])
            mask_tensor = torch.cat(
                (torch.zeros(self.seq_length - excess), torch.ones(excess))
            ).bool()
        else:
            mask_tensor = torch.zeros(self.seq_length).bool()

        return {"x": frame_tensor, "y": label_tensor, "mask": mask_tensor}


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    ds = FrameSequenceDataset(
        path="data/browns-ravens",
        labs_file="frames.csv",
        ext=".jpg",
        seq_length=128,
    )
    print(len(ds))
    example = ds[133774]
    print(example["x"].shape)
    print(example["y"].shape)
    print(example)

    # dl = DataLoader(ds, 64, shuffle=True)

    # batch = next(iter(dl))

    # print(batch)
