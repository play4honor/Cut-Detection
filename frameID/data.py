import os

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


if __name__ == "__main__":

    from torchvision.utils import save_image, make_grid
    from torch.utils.data import DataLoader

    p = 0.8

    trs = transforms.Compose(
        [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(1, 1.4)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomResizedCrop(
                size=(144, 256), scale=(0.8, 1), ratio=(1.77, 1.78)
            ),
        ]
    )

    ds = ContrastiveFrameDataset("data/ravens-lions", trs=trs, ext=".jpg")

    dl = DataLoader(ds, 64)

    batch = next(iter(dl))

    print(batch)

    ex_a = make_grid([ds[0]["x"], ds[0]["x_transformed"], ds[0]["x_transformed"]])

    save_image(ex_a, "example_a.jpg")
