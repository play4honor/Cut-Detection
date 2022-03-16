from frameID.net import load_and_glue_nets
from frameID.data import SupervisedFrameDataset

import streamlit as st

import torch
import torch.nn.functional as F


@st.cache
def load_data(path):
    return F.softmax(torch.load(path), dim=1).numpy()


@st.cache
def load_model_and_dataset(arg="definitely a real argument"):

    MODEL_DIR = "./models"
    MODEL_NAME = "frame_compression_model"

    # Load the network.
    conv_net, linear_net, _ = load_and_glue_nets(
        param_file=f"{MODEL_DIR}/{MODEL_NAME}_model_params.json",
        conv_file=f"{MODEL_DIR}/{MODEL_NAME}_classifier_conv.pt",
        linear_file=f"{MODEL_DIR}/{MODEL_NAME}_classifier_linear.pt",
        separate=True,
    )

    ds = SupervisedFrameDataset("data/ravens-packers", "frames.csv")

    return conv_net.eval(), linear_net.eval(), ds


scores = load_data("model_output.pt")
conv_net, linear_net, frame_ds = load_model_and_dataset()

img_base_path = "./data/ravens-packers/"

start_frame = st.sidebar.number_input("Start Frame", 0, scores.shape[0])
end_frame = st.sidebar.number_input("End Frame", 0, scores.shape[0], 100)

data_subset = scores[start_frame:end_frame, :]

st.line_chart(data=data_subset)

img_idx = st.sidebar.number_input("Frame", 0, scores.shape[0])
frame_scores = scores[img_idx, :]


# Load frame and score with models
x = frame_ds[img_idx]["x"].unsqueeze(0)
x.requires_grad = True

embedding = conv_net(x)
non_sequence_scores = F.softmax(linear_net(embedding)).squeeze(0)
max_score = non_sequence_scores.max()

# Calculate gradients
gradients = torch.autograd.grad(
    outputs=max_score,
    inputs=x,
    create_graph=True,
    retain_graph=True,
)[0]


grad_image = (
    gradients.squeeze(0)
    .permute(1, 2, 0)
    .mean(dim=2, keepdim=True)
    .abs()
    .detach()
    .numpy()
)
grad_image = grad_image / grad_image.max()

st.image(f"{img_base_path}frame_{img_idx :07}.jpg", width=512)

st.markdown(
    f"Stacked Model Scores: {frame_scores[0] :.3} | {frame_scores[1] :.3} | {frame_scores[2] :.3}"
)
st.markdown(
    f"Conv Model Scores: {non_sequence_scores[0] :.3} | {non_sequence_scores[1] :.3} | {non_sequence_scores[2] :.3}"
)

st.image(grad_image, "Saliency", width=512)
