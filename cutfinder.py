import cv2
import numpy as np
import scipy.linalg as la
from scipy.ndimage import uniform_filter1d

import datetime
import argparse
import os

def frame_distance(frame1: np.ndarray, frame2: np.ndarray, stride: int = 20):
    """
    Find the average per-pixel color distance between two frames.
    Stride determines how sparsely pixels are selected, since it's not
    really necessary to check ever single pixel.
    """

    # I'm not error checking because this is for internal use.
    frame1_strided = frame1[::stride, ::stride, :].reshape(-1, frame1.shape[-1]).astype(int)
    frame2_strided = frame2[::stride, ::stride, :].reshape(-1, frame2.shape[-1]).astype(int)

    x = la.norm(
        frame1_strided - frame2_strided,
        ord=2,
        axis=1
        ).mean()

    return x

def find_frame_diffs(file_path, stride:int = 20):
    """
    Find the frame_by_frame difference in a video. This also returns the frame
    rate because we're going to need it later and this is the most convenient
    place to get it.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

    # Start the video capture
    vc = cv2.VideoCapture(file_path)

    # I hate these stupid enumerations.
    fps = vc.get(cv2.CAP_PROP_FPS)

    # Get total frames
    total_frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Total frames: {total_frames}")

    curr_frame = None
    prev_frame = None
    frame_diffs = np.zeros(int(total_frames - 1))
    idx = 0

    while vc.isOpened():

        ret, frame = vc.read()

        if ret:

            if curr_frame is None:
                curr_frame = frame
            else:
                prev_frame = curr_frame
                curr_frame = frame

                frame_diffs[idx] = frame_distance(curr_frame, prev_frame, stride=stride)
                idx += 1

        else:
            vc.release()
            break

    return fps, frame_diffs

def find_cut_indices(frame_diffs: np.ndarray, threshold=5.0):
    """
    Take a set of frame differences and produce the indices where the cuts are.
    threshold controls how sensitive the cut detection is. Higher = less sensitive.
    """

    # Get a local average of frame-by-frame differences.
    smooth_diffs = uniform_filter1d(frame_diffs, 29, mode='mirror', origin = 0)

    # Calculate how much more variation there than the mean, between each pair.
    exceedance = np.maximum((frame_diffs - smooth_diffs) / smooth_diffs, 0)

    # Find the indices where the exceedance is greater than the detection threshold.
    cuts = np.nonzero(exceedance > threshold)[0]

    return cuts

def get_timestamp_from_seconds(seconds):
    """One liner to get a nice timestamp from a count of seconds."""
    return str(datetime.timedelta(seconds=seconds))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args()

    fps, frame_diffs = find_frame_diffs(args.video, stride=args.stride)

    cut_idx = find_cut_indices(frame_diffs, threshold=args.threshold)

    with open(args.output, "w") as f:

        for t in cut_idx * (1/fps):

            nice_time = get_timestamp_from_seconds(t)
            f.write(nice_time + "\n")