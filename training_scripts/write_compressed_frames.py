from frameID.net import load_and_glue_nets
from frameID.data import SupervisedFrameDataset

import torch
from torch.utils.data import DataLoader, ConcatDataset

import logging
import os
import math
import json
from itertools import chain

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
)

# Resourcing
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Using {device}")
NUM_WORKERS = 3

BATCH_SIZE = 128
WRITE_EVERY_N = 1000

MODEL_DIR = "./models"
MODEL_NAME = "frame_compression_model"
OUTPUT_PATH = "./data/compressed_frames.pt"

# Load the network.
net, _ = load_and_glue_nets(
    param_file=f"{MODEL_DIR}/{MODEL_NAME}_model_params.json",
    conv_file=f"{MODEL_DIR}/{MODEL_NAME}_classifier_conv.pt",
)

net.to(device)

# Initialize the dataset class.
# 100% should come from a config file.
data_dirs = [
    "data/bengals-ravens",
    "data/browns-ravens",
    "data/bears-ravens",
    "data/dolphins-ravens",
    "data/ravens-browns",
    "data/ravens-bengals",
    # "data/ravens-packers",
    # "data/steelers-ravens",
]
labs_files = ["frames.csv"] * len(data_dirs)

ds_list = [
    SupervisedFrameDataset(dir, lf, ext=".jpg")
    for dir, lf in zip(data_dirs, labs_files)
]

ds = ConcatDataset(ds_list)

# Data loader
loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False,
)
logging.info(f"{len(ds)} frames, {len(loader)} batches.")

if __name__ == "__main__":

    # Score frames
    net.eval()
    with torch.no_grad():

        all_compressed_frames = []
        all_labels = []

        for i, data in enumerate(loader):

            x = data["x"].to(device)
            labels = data["y"].squeeze()
            out_frames = net(x)

            all_labels.append(labels)
            all_compressed_frames.append(out_frames)

            if i % WRITE_EVERY_N == WRITE_EVERY_N - 1:

                logging.info(f"Scored {i+1} batches")

        compressed_frames = torch.cat(all_compressed_frames)
        labels = torch.cat(all_labels)

        print(compressed_frames.shape)
        print(labels.shape)

        logging.info(f"Writing results to {OUTPUT_PATH}")
        torch.save({"x": compressed_frames.to("cpu"), "y": labels}, OUTPUT_PATH)
