from numpy import datetime64
import streamlit as st

import datetime
import os

import cv2
import pandas as pd

# streamlit run test_app.py

def open_video(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, fps, length

def get_cuts(cut_path):

    if os.path.isfile(cut_path):
        cuts = pd.read_csv(cut_path, header = None)
        return cuts

    else:
        return None

def get_frames(cap, frame_num, max_frame, window = 3):

    start_frame = max(0, frame_num - window)
    end_frame = min(max_frame, frame_num + window)

    frame_nums = list(range(start_frame, end_frame))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []

    for i in range(end_frame - start_frame):

        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    return frames, frame_nums


# Should this be in a name = main gate?

filepath = st.sidebar.text_input("Video Path")
cutpath = st.sidebar.text_input("Cut Path")

if filepath == '':
    st.markdown("### Please choose a video file.")

else:

    cap, fps, length = open_video(filepath)

    max_seconds = length / fps

    cuts = get_cuts(cutpath)

    if cuts is not None:

        curr_cut = st.sidebar.number_input("Cut", min_value=1, max_value=cuts.shape[0], value=1)

        st.sidebar.dataframe(cuts.style.apply(
            lambda x: ['background: lightgreen' if x.name == curr_cut - 1 else '' for i in x], 
            axis=1)
        )

        frame_num = int(cuts.iloc[curr_cut - 1, 0] * fps)

    else:

        frame_num = int(st.slider('Time', 0.0, max_seconds, 0.0) * fps)

    imgs, frame_nums = get_frames(cap, frame_num, length, window=15)

    captions = [f"{x} ({str(datetime.timedelta(seconds=x/fps))})" for x in frame_nums]

    st.image(imgs, caption=captions, width=120, channels='BGR')