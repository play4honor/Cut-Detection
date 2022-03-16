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


class SupervisedFrameDataset(IterableDataset):
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
        self.curr_frame = 0

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

    def _is_blank(self, img_tensor):
        return img_tensor.mean().item() < 0.1

    def __iter__(self):
        return self

    def __next__(self):

        idx = self.curr_frame
        # Increment the current frame
        self.curr_frame += 1

        if idx >= len(self.file_list):
            raise StopIteration

        p = self.file_list[idx]
        x = read_image(p, self.read_mode)
        x = x.float() / 255

        label = self._get_label(idx)

        # Check if image is blank
        if self._is_blank(x) or label.item() == 2:

            return next(self)

        else:

            return {"x": x, "y": label.long()}

    # def __len__(self):

    #     return len(self.file_list)


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


class CompressedDataset(Dataset):
    def __init__(self, file_path, seq_length=128):

        super().__init__()

        data = torch.load(file_path)

        self.seq_length = seq_length
        self.x = data["x"]
        self.y = data["y"]
        self.conv_scores = data["scores"]

        self.weights = self._construct_weights()

    def __len__(self):

        return self.x.shape[0]

    def __getitem__(self, idx):

        idx_range = [idx, min(idx + self.seq_length, len(self))]

        x = self.x[idx_range[0] : idx_range[1], :]
        y = self.y[idx_range[0] : idx_range[1]]
        weight = self.weights[idx_range[0] : idx_range[1]]
        score = self.conv_scores[idx_range[0] : idx_range[1], :]

        x, mask, y, score, weight = self.transform_tensor(
            x, self.seq_length, y, score, weight
        )

        return {"x": x, "y": y, "score": score, "mask": mask, "weight": weight}

    def _construct_weights(self):

        return 1 - torch.sigmoid(self.conv_scores.max(dim=1)[0])

    @staticmethod
    def transform_tensor(x, seq_length, y=None, score=None, weight=None):

        pad_size = max(0, seq_length - x.shape[0])

        if pad_size > 0:

            x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
            y = (
                torch.nn.functional.pad(y, (0, pad_size), value=3)
                if y is not None
                else None
            )
            score = (
                torch.nn.functional.pad(score, (0, 0, 0, pad_size), value=3)
                if score is not None
                else None
            )
            weight = (
                torch.nn.functional.pad(weight, (0, pad_size), value=0)
                if weight is not None
                else None
            )

            mask = torch.cat(
                (torch.zeros(seq_length - pad_size), torch.ones(pad_size))
            ).bool()

        else:

            mask = torch.zeros(seq_length).bool()

        return x, mask, y, score, weight


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    ds = CompressedDataset("data/training_compressed_frames.pt", seq_length=128)

    print(len(ds))
    print(ds[0])

    raise NotImplementedError
    print(ds[801530]["score"])
    print(ds[801530]["y"].dtype)
    print(ds[801530]["mask"])

    dl = DataLoader(ds, 64, shuffle=True)

    batch = next(iter(dl))

    print(batch["x"].shape)
    print(batch["y"].shape)
    print(batch["mask"].shape)
