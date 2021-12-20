from frameID.data import ContrastiveFrameDataset

import streamlit as st

import torch

DATA_DIR = "data/ravens-lions"


def get_frame(dataset, idx):

    img = torch.permute(dataset[idx]["x"], (1, 2, 0))

    return img.numpy()


# We don't want any transforms here.
trs = torch.nn.Identity()

ds = ContrastiveFrameDataset(DATA_DIR, trs=trs, ext=".jpg")

idx = torch.randint(len(ds), size=(1,)).item()
frame = get_frame(ds, idx)

st.image(frame, width=240)

if st.button("Endzone"):
    print("Endzone")

if st.button("All-22"):
    print("All_22")

if st.button("Blank"):
    print("Blank")
