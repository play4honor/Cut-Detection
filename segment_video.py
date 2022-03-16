from frameID.net import load_stacked_model, LovieNet
from frameID.data import VideoDataset, CompressedDataset
from frameID.segmentation import Segmentation

import torch
from torch.utils.data import DataLoader

import logging
import os
import argparse


# Logging setup
logging.basicConfig(
    level="INFO",
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
)


def main(args):

    if not os.path.isfile(args.input_path):
        raise ValueError(f"{args.input_path} does not exist.")

    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    logging.info(f"Using {device}")

    ds = VideoDataset(args.input_path, resize=256)
    dl = DataLoader(ds, args.batch_size)

    logging.info(f"{len(ds)} total frames | {len(dl)} total batches")

    conv_net, nagy_net = load_stacked_model(
        "models/frame_compression_model_model_params.json",
        "models/frame_compression_model_classifier_conv.pt",
        LovieNet,
        "models/frame_compression_model_transformer_model_params.json",
        "models/frame_compression_model_transformer.pt",
    )

    conv_net.to(device)
    nagy_net.to(device)

    logging.info("Loaded Cyber-Nagy")

    with torch.no_grad():

        yy = []

        for i, batch in enumerate(iter(dl)):

            batch = batch.to(device)
            compressed_frames = conv_net(batch)

            x, mask, _, _, _ = CompressedDataset.transform_tensor(
                compressed_frames, args.batch_size
            )

            yy.append(nagy_net(x.unsqueeze(0)).squeeze(0))

            if args.print_every > 0:
                if i % args.print_every == args.print_every - 1:
                    logging.info(
                        f"Scored batch {i+1} ({(i+1) * args.batch_size} frames)."
                    )

            # This may get removed.
            if (
                args.frame_limit is not None
                and (i + 1) * args.batch_size > args.frame_limit
            ):
                break

        yy = torch.cat(yy, 0).to("cpu")

        torch.save(yy, "model_output.pt")

        seg = Segmentation(yy)
        logging.info(f"Found {len(seg)} initial segments")
        seg.glue_orphans(args.base_threshold, args.blank_threshold)
        logging.info(f"Revised to {len(seg)} segments through orphan combination.")
        seg.combine_adjacent_segments()
        logging.info(
            f"Revised to {len(seg)} segments through matching adjacent combination."
        )

        if args.output_path is None:
            out_path = os.path.splitext(args.input_path)[0] + "_segments.csv"
        else:
            out_path = args.output_path

        logging.info(f"Writing {len(seg)} segments to {out_path}")
        seg.write_csv(out_path)


# Set up parser and parse arguments
sv_parser = argparse.ArgumentParser(
    "Segment a video into scenes.", fromfile_prefix_chars="@"
)
sv_parser.add_argument("input_path", type=str, help="Path to video to segment.")
sv_parser.add_argument(
    "--output-path",
    type=str,
    default=None,
    help="Path to output csv",
)
sv_parser.add_argument(
    "--base-threshold",
    type=int,
    default=100,
    help="Number of frames below which an A22 or EZ segment will be considered an orphan.",
)
sv_parser.add_argument(
    "--blank-threshold",
    type=int,
    default=10,
    help="Number of frames below which a blank segment will be considered an orphan.",
)
sv_parser.add_argument(
    "--batch-size", type=int, default=128, help="Batch size for loading frames."
)
sv_parser.add_argument(
    "--print-every",
    type=int,
    default=50,
    help="Log message every n batches. 0 to disable.",
)
sv_parser.add_argument(
    "--frame-limit",
    type=int,
    default=None,
    help="Limit how many frames are processed. Mainly for testing.",
)
sv_parser.add_argument(
    "--cpu", action="store_true", help="Don't use cuda even if it's available."
)

if __name__ == "__main__":

    args = sv_parser.parse_args()

    main(args)
