import streamlit as st

import os

import cv2
import pandas as pd
import numpy as np
import plotnine as pn

def open_video(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, fps, length

@st.cache
def get_diffs(diff_path):

    if os.path.isfile(diff_path):
        diffs = pd.read_csv(diff_path, header = None)
        return diffs

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

def make_diff_graph(diffs, selected_frame, window):

    min_frame = max(selected_frame - window, 0)
    max_frame = min(selected_frame + window, diffs.shape[0])

    plot = (
        pn.ggplot()
        + pn.geom_line(
            pn.aes(
                x=np.arange(diffs.shape[0]),
                y=diffs.iloc[:, 0].to_numpy()
            ),
            color = "#666666")
        + pn.geom_vline(pn.aes(xintercept=[min_frame, max_frame]), color = "#ff8888")
        + pn.theme_minimal()
        + pn.theme(figure_size=(8, 2))
        )

    return plot

# ----------------------------------------------------------------------------
# App starts here

filepath = st.sidebar.text_input("Video Path")
diffpath = st.sidebar.text_input("Diff Path")
window = st.sidebar.number_input("Window Size", 3, 15, 3)

if filepath == '':
    st.markdown("### Please choose a video file.")

else:

    cap, fps, length = open_video(filepath)

    max_seconds = length / fps

    diffs = get_diffs(diffpath)

    frame_num = int(st.slider('Time', 0, length, 0))

    if diffs is not None:

        plt = make_diff_graph(diffs, frame_num, window)

        st.pyplot(pn.ggplot.draw(plt))


    imgs, frame_nums = get_frames(cap, frame_num, length, window=window)

    captions = [f"{x}" for x in frame_nums]

    st.image(imgs, caption=captions, width=120, channels='BGR')