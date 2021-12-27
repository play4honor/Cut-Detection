from frameID.data import ContrastiveFrameDataset

import streamlit as st
import torch

import os
import json

DATA_DIR = "data/ravens-lions"


def get_frame(dataset, idx):

    img = torch.permute(dataset[idx]["x"], (1, 2, 0))

    return img.numpy()


def get_labels(dir):

    if not os.path.isfile(f"{dir}/labels.json"):
        return {}

    else:
        with open(f"{dir}/labels.json", "r") as f:
            return json.load(f)


def write_label(dir, labels, idx, lab):

    labels[idx] = lab

    with open(f"{dir}/labels.json", "w") as f:
        json.dump(labels, f)

    return labels


# We don't want any transforms here.
trs = torch.nn.Identity()

ds = ContrastiveFrameDataset(DATA_DIR, trs=trs, ext=".jpg")

labels = get_labels(DATA_DIR)

idx = torch.randint(len(ds), size=(1,)).item()
st.markdown(f"*Previous label: {labels.get(idx, 'None')}*")
frame = get_frame(ds, idx)

st.image(frame, width=240)

if st.button("Endzone"):
    write_label(DATA_DIR, labels, idx, "EZ")

if st.button("All-22"):
    write_label(DATA_DIR, labels, idx, "A22")

if st.button("Blank"):
    write_label(DATA_DIR, labels, idx, "B")

st.markdown(f"**Labels: {len(labels)}**")
