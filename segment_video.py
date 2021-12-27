from frameID.net import load_default_net
from frameID.data import VideoDataset, SupervisedFrameDataset
from frameID.segmentation import Segmentation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse
import logging

MODEL_DIR = "./models"
MODEL_NAME = "init_model"

# Logging and cuda
logging.basicConfig(
    level="INFO",
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Using {device}")

# Set up parser and parse arguments
parser = argparse.ArgumentParser("Segment a video into scenes.")
parser.add_argument("input_path", type=str, help="Path to video to segment.")
parser.add_argument(
    "output_path",
    type=str,
    help="Path to output csv",
)
parser.add_argument(
    "--base-threshold",
    type=int,
    default=100,
    help="Number of frames below which an A22 or EZ segment will be considered an orphan.",
)
parser.add_argument(
    "--blank-threshold",
    type=int,
    default=10,
    help="Number of frames below which a blank segment will be considered an orphan.",
)
parser.add_argument(
    "--batch-size", type=int, default=128, help="Batch size for loading frames."
)
parser.add_argument(
    "--print-every",
    type=int,
    default=50,
    help="Log message every n batches. 0 to disable.",
)

args = parser.parse_args()

ds = VideoDataset(args.input_path, resize=256)
dl = DataLoader(ds, args.batch_size)

net, params = load_default_net()

net.eval()
net.to(device)

logging.info("Loaded default classifier.")

with torch.no_grad():

    yy = []

    for i, batch in enumerate(iter(dl)):

        batch = batch.to(device)
        yy.append(net(batch))

        if args.print_every > 0:
            if i % args.print_every == args.print_every - 1:
                logging.info(f"Scored batch {i+1} ({(i+1) * args.batch_size} frames).")

    yy = torch.cat(yy, 0).to("cpu")

    seg = Segmentation(yy)
    logging.info(f"Found {len(seg)} initial segments")
    seg.glue_orphans(args.base_threshold, args.blank_threshold)
    logging.info(f"Revised to {len(seg)} segments.")

    logging.info(f"Writing {len(seg)} segments to {args.output_path}")
    seg.write_csv(args.output_path)
