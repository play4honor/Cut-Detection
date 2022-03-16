from frameID.net import FrameConvNet, FrameLinearNet
from frameID.data import SupervisedFrameDataset, split_iter_workers

import torch
from torch.utils.data import DataLoader, ChainDataset, Subset

import logging
import os
import math
import json
from itertools import chain

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
)

# Resourcing
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Using {device}")
NUM_WORKERS = 3

# TODO Read all this from a config file and save the file with the model.

# Conv Net Design
CONV_LAYERS = 3
CONV_HIDDEN_CHANNELS = 48
AVG_POOL_SIZE = 4
CONV_OUTPUT_SIZE = 32

# Output Network Design
LINEAR_LAYERS = 2
LINEAR_SIZE = 32
OUTPUT_SIZE = 2

# Training Details
DATA_SIZE = 150_000
BATCH_SIZE = 128
EPOCHS = 3
WRITE_EVERY_N = 1000
OPTIMIZER = "AdamW"

MODEL_DIR = "./models"
MODEL_NAME = "frame_compression_model"
LOAD_CONV_NET = False

# Setup optimizer.
opt_class = getattr(torch.optim, OPTIMIZER)

# Initialize the dataset class and then split into train/valid.
# 100% should come from a config file.
train_dirs = [
    "data/bengals-ravens",
    # "data/browns-ravens",
    "data/bears-ravens",
    "data/dolphins-ravens",
    "data/ravens-browns",
    "data/ravens-bengals",
    # "data/ravens-packers",
    # "data/steelers-ravens",
]
train_labs_files = ["frames.csv"] * len(train_dirs)

train_ds_list = [
    SupervisedFrameDataset(dir, lf, ext=".jpg")
    for dir, lf in zip(train_dirs, train_labs_files)
]

ds_train = ChainDataset(train_ds_list)

valid_dirs = ["data/browns-ravens"]
valid_labs_files = ["frames.csv"] * len(valid_dirs)

valid_ds_list = [
    SupervisedFrameDataset(dir, lf, ext=".jpg")
    for dir, lf in zip(valid_dirs, valid_labs_files)
]

ds_valid = ChainDataset(valid_ds_list)

train_loader = DataLoader(
    ds_train,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    drop_last=False,
    worker_init_fn=split_iter_workers
)
valid_loader = DataLoader(
    ds_valid,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    drop_last=False,
    worker_init_fn=split_iter_workers
)

# logging.info(
#     f"Training Batches: {len(train_loader)} | Validation Batches: {len(valid_loader)}"
# )

if __name__ == "__main__":

    conv_net = FrameConvNet(
        hidden_channels=CONV_HIDDEN_CHANNELS,
        n_conv_layers=CONV_LAYERS,
        average_pool_size=AVG_POOL_SIZE,
        output_size=CONV_OUTPUT_SIZE,
    )

    if LOAD_CONV_NET:
        logging.info(
            f"Loading pre-trained conv net from: {MODEL_DIR}/{MODEL_NAME}_conv.pt"
        )
        state_dict = torch.load(f"{MODEL_DIR}/{MODEL_NAME}_conv.pt", map_location="cpu")
        conv_net.load_state_dict(state_dict)

    linear_net = FrameLinearNet(
        n_layers=LINEAR_LAYERS,
        input_size=CONV_OUTPUT_SIZE,
        hidden_size=LINEAR_SIZE,
        output_size=OUTPUT_SIZE,
    )

    conv_net.to(device)
    linear_net.to(device)
    logging.info(f"Convolutional Network Weights: {conv_net.num_params()}")
    logging.info(f"Output Network Weights: {linear_net.num_params()}")

    # We want the optimizer to optimize both networks.
    optimizer = opt_class(
        chain(
            filter(lambda p: p.requires_grad, conv_net.parameters()),
            filter(lambda p: p.requires_grad, linear_net.parameters()),
        )
    )
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    # Training loop
    for epoch in range(EPOCHS):

        # Training

        logging.info(f"Starting epoch {epoch+1} of {EPOCHS}")

        accum_loss = 0.0
        n_obs = 0

        conv_net.train()
        linear_net.train()
        for i, data in enumerate(train_loader):

            optimizer.zero_grad()

            x = data["x"].to(device)
            labels = data["y"].squeeze().to(device)
            intermediate = conv_net(x)
            pred = linear_net(intermediate)

            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            n_obs += x.shape[0]

            if i % WRITE_EVERY_N == WRITE_EVERY_N - 1:

                logging.info(
                    f"Epoch {epoch+1} | Batch {i+1} | Loss: {accum_loss / n_obs :.3f}"
                )
                accum_loss = 0.0
                n_obs = 0.0

        # Validation

        logging.info(f"Starting validation for epoch {epoch+1}")

        accum_loss = 0.0
        n_obs = 0

        correct = torch.zeros([OUTPUT_SIZE]).to(device)
        total = torch.zeros([OUTPUT_SIZE]).to(device)

        conv_net.eval()
        linear_net.eval()
        with torch.no_grad():

            for i, data in enumerate(valid_loader):

                x = data["x"].to(device)
                labels = data["y"].squeeze().to(device)
                intermediate = conv_net(x)
                pred = linear_net(intermediate)

                loss = criterion(pred, labels)

                pc = torch.max(pred, dim=1)[1]

                for i in range(OUTPUT_SIZE):

                    correct[i] += torch.sum(pc[labels == i] == labels[labels == i])
                    total[i] += torch.sum(labels == i)

                accum_loss += loss.item()
                n_obs += x.shape[0]

                if i % WRITE_EVERY_N == WRITE_EVERY_N - 1:

                    logging.info(
                        f"Validation: Epoch {epoch+1} | Batch {i+1} | Loss: {accum_loss / n_obs :.3f}"
                    )
                    accum_loss = 0.0
                    n_obs = 0.0

            logging.info(f"Valid accuracy for A22: {(correct[0]/total[0]).item() :.3f}")
            logging.info(f"Valid accuracy for EZ: {(correct[1]/total[1]).item() :.3f}")

    # We don't have any fancy way to save checkpoints, or stop early or anything.

    # Create the output path if necessary.
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    # Save the two models and the parameters.
    torch.save(conv_net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_classifier_conv.pt")
    torch.save(
        linear_net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_classifier_linear.pt"
    )
    with open(f"{MODEL_DIR}/{MODEL_NAME}_model_params.json", "w") as f:
        json.dump(
            {
                # Convolutional params
                "conv_layers": CONV_LAYERS,
                "conv_channels": CONV_HIDDEN_CHANNELS,
                "avg_pool_size": AVG_POOL_SIZE,
                "conv_output_size": CONV_OUTPUT_SIZE,
                # Linear params
                "linear_layers": LINEAR_LAYERS,
                "linear_size": LINEAR_SIZE,
                "linear_output_size": OUTPUT_SIZE,
                # Training params
                "data_size": DATA_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
            },
            f,
        )
