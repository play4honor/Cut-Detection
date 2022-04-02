from frameID.data import SupervisedFrameDataset
from torchvision.io import read_image

import json


class BlankFinder(SupervisedFrameDataset):
    def check_frame(self, idx):

        p = self.file_list[idx]
        x = read_image(p, self.read_mode)
        x = x.float() / 255

        label = self._get_label(idx)

        is_blank = self._is_blank(x) or label.item() == 2

        return (p, label, is_blank)


frame_dirs = [
    "data/bengals-ravens",
    "data/browns-ravens",
    "data/bears-ravens",
    "data/dolphins-ravens",
    "data/ravens-browns",
    "data/ravens-bengals",
    "data/ravens-packers",
    "data/steelers-ravens",
]

if __name__ == "__main__":

    for dir in frame_dirs:

        print(f"Indexing {dir}")
        ds = BlankFinder(dir, "frames.csv", ext=".jpg")

        frame_info = [ds.check_frame(idx) for idx in range(len(ds))]
        non_blank_frames = [
            (dir, label.item()) for dir, label, is_blank in frame_info if not is_blank
        ]

        with open(f"{dir}/frame_index.json", "w") as f:

            json.dump(non_blank_frames, f)
