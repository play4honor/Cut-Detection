from frameID.net import load_default_net
from frameID.data import VideoDataset
from frameID.segmentation import Segmentation

import torch
from torch.utils.data import DataLoader

import argparse
import logging
import os

# Logging setup
logging.basicConfig(
    level="INFO",
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
)

# Set up parser and parse arguments
parser = argparse.ArgumentParser("Segment a video into scenes.")
parser.add_argument("input_path", type=str, help="Path to video to segment.")
parser.add_argument(
    "output_path",
    type=str,
    default=None,
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
parser.add_argument(
    "--frame-limit",
    type=int,
    default=None,
    help="Limit how many frames are processed. Mainly for testing.",
)
parser.add_argument(
    "--cpu", action="store_true", help="Don't use cuda even if it's available."
)


# Start doing stuff here. Probably should be main()

args = parser.parse_args()

if not os.path.isfile(args.input_path):
    raise ValueError(f"{args.input_path} does not exist.")

device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
logging.info(f"Using {device}")

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

        # This may get removed.
        if (
            args.frame_limit is not None
            and (i + 1) * args.batch_size > args.frame_limit
        ):
            break

    yy = torch.cat(yy, 0).to("cpu")

    seg = Segmentation(yy)
    logging.info(f"Found {len(seg)} initial segments")
    seg.glue_orphans(args.base_threshold, args.blank_threshold)
    logging.info(f"Revised to {len(seg)} segments through orphan combination.")
    seg.combine_adjacent_segments()
    logging.info(
        f"Revised to {len(seg)} segments through matching adjacent combination."
    )

    if args.output_path is None:
        out_path = os.path.splitext(args.input_path)[0] + "_frames.csv"
    else:
        out_path = args.output_path

    logging.info(f"Writing {len(seg)} segments to {out_path}")
    seg.write_csv(out_path)
