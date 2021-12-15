from frameID.data import open_video
import cv2
import argparse
import os


# Set up parser and parse arguments
parser = argparse.ArgumentParser("Split a video into individual frames.")
parser.add_argument("input_path", type=str, help="Path to video to turn into frames.")
parser.add_argument("output_dir", type=str, help="Path to directory to write images. Will be created if it doesn't exist.")
parser.add_argument("--resize", type=int, default=0, help="Size of larger dimension.")
parser.add_argument("--max-frames", type=int, default=-1, help="Number of frames to save.")

args = parser.parse_args()

cap, v_properties = open_video(args.input_path)

frame_limit = v_properties["length"] if args.max_frames < 0 else args.max_frames
print(f"Processing {frame_limit} frames from {args.input_path}.")

# Create the output path if necessary.
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

# Calculate the correct dimensions
if args.resize > 0:

    new_width = args.resize
    new_height = int(v_properties["height"] * (new_width / v_properties["width"]))

for i in range(frame_limit):

    if i % 5000 == 4999:
        print(f"Processing frame {i+1}")

    ret, frame = cap.read()

    if ret:
        if args.resize > 0:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(f"{args.output_dir}/frame_{i}.jpg", frame)

print("Done")