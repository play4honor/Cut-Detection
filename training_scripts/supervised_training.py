from frameID.net import FrameConvNet, FeedForward, FunNameModel
from frameID.data import FrameSequenceDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset

import logging
import os
import math
import json

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
CONV_HIDDEN_CHANNELS = 32
AVG_POOL_SIZE = 4

# Transformer Design
N_HEADS = 2
TR_LINEAR_SIZE = 64
TR_LAYERS = 3

# Output Network Design
LINEAR_SIZE = 32
OUTPUT_SIZE = 3

# Training Details
DATA_SIZE = 150_000
BATCH_SIZE = 4
EPOCHS = 2
WRITE_EVERY_N = 100
OPTIMIZER = "AdamW"

MODEL_DIR = "./models"
MODEL_NAME = "init_transformer"
LOAD_CONV_NET = False

# Setup optimizer.
opt_class = getattr(torch.optim, OPTIMIZER)

# Initialize the dataset class and then split into train/valid.
# 100% should come from a config file.
data_dirs = [
    "data/bengals-ravens",
    # "data/browns-ravens",
    # "data/bears-ravens",
    # "data/dolphins-ravens",
    # "data/ravens-browns",
    # "data/ravens-bengals",
    # "data/ravens-packers",
    # "data/steelers-ravens",
]
labs_files = ["frames.csv"] * len(data_dirs)

ds_list = [
    FrameSequenceDataset(path=dir, labs_file=lf, ext=".jpg", seq_length=128)
    for dir, lf in zip(data_dirs, labs_files)
]

ds = ConcatDataset(ds_list)

idx_perm = torch.randperm(len(ds))

train_idx = idx_perm[: math.floor(len(ds) * 0.75)].tolist()
valid_idx = idx_perm[math.floor(len(ds) * 0.75) :].tolist()

ds_train = Subset(ds, train_idx)
ds_valid = Subset(ds, valid_idx)

train_loader = DataLoader(
    ds_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=False,
)
valid_loader = DataLoader(
    ds_valid,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=False,
)

logging.info(
    f"Training Batches: {len(train_loader)} | Validation Batches: {len(valid_loader)}"
)

if __name__ == "__main__":

    conv_net = FrameConvNet(
        hidden_channels=CONV_HIDDEN_CHANNELS,
        n_conv_layers=CONV_LAYERS,
        average_pool_size=AVG_POOL_SIZE,
    )

    if LOAD_CONV_NET:
        logging.info(
            f"Loading pre-trained conv net from: {MODEL_DIR}/{MODEL_NAME}_conv.pt"
        )
        state_dict = torch.load(f"{MODEL_DIR}/{MODEL_NAME}_conv.pt", map_location="cpu")
        conv_net.load_state_dict(state_dict)

    mapping_layer = nn.Linear((AVG_POOL_SIZE ** 2) * CONV_HIDDEN_CHANNELS, LINEAR_SIZE)

    output_layer = FeedForward(
        input_dims=LINEAR_SIZE,
        ff_dims=LINEAR_SIZE,
        n_classes=OUTPUT_SIZE,
    )

    enc_layer = nn.TransformerEncoderLayer(
        d_model=LINEAR_SIZE,
        nhead=N_HEADS,
        dim_feedforward=TR_LINEAR_SIZE,
        batch_first=True,
    )

    net = FunNameModel(conv_net, mapping_layer, enc_layer, TR_LAYERS, output_layer)

    net.to(device=device)
    logging.info(f"Total Network Weights: {net.num_params()}")

    # We want the optimizer to optimize both networks.
    optimizer = opt_class(filter(lambda p: p.requires_grad, net.parameters()))
    criterion = nn.CrossEntropyLoss(reduction="none")

    scaler = torch.cuda.amp.GradScaler()

    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=2),
        on_trace_ready=trace_handler,
        with_stack=True,
    ) as profiler:

        # Training loop
        for epoch in range(EPOCHS):

            # Training

            logging.info(f"Starting epoch {epoch+1} of {EPOCHS}")

            accum_loss = 0.0
            n_obs = 0

            net.train()
            for i, data in enumerate(train_loader):

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():

                    x = data["x"].to(device=device)

                    labels = data["y"].unsqueeze(-1).to(device)
                    mask = data["mask"].to(device)
                    pred = net(x, mask)

                    loss = criterion(
                        torch.reshape(pred, [-1, OUTPUT_SIZE]),
                        torch.reshape(labels, [-1]),
                    )

                    loss[torch.reshape(mask, [-1])] = 0.0
                    loss = loss.sum()

                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scaler.update()

                accum_loss += loss.item()
                n_obs += x.shape[0] * x.shape[1] - mask.sum()

                if i % WRITE_EVERY_N == WRITE_EVERY_N - 1:

                    logging.info(
                        f"Epoch {epoch+1} | Batch {i+1} | Loss: {accum_loss / n_obs :.3f}"
                    )
                    accum_loss = 0.0
                    n_obs = 0.0

                profiler.step()

            # Validation

            logging.info(f"Starting validation for epoch {epoch+1}")

            accum_loss = 0.0
            n_obs = 0

            correct = torch.zeros([3]).to(device)
            total = torch.zeros([3]).to(device)

            net.eval()
            with torch.no_grad():

                for i, data in enumerate(valid_loader):

                    with torch.cuda.amp.autocast():

                        x = data["x"].to(device=device)
                        labels = data["y"].squeeze().to(device)
                        print(x)
                        mask = data["mask"].to(device)
                        pred = net(x, mask)

                        loss = criterion(pred, labels)

                        loss[mask] = 0.0
                        loss = loss.sum()

                    pc = torch.max(pred, dim=2)[1]

                    for i in range(3):

                        correct[i] += torch.sum(pc[labels == i] == labels[labels == i])
                        total[i] += torch.sum(labels == i)

                    accum_loss += loss.item()
                    n_obs += x.shape[0] * x.shape[1] - mask.sum()

                    if i % WRITE_EVERY_N == WRITE_EVERY_N - 1:

                        logging.info(
                            f"Validation: Epoch {epoch+1} | Batch {i+1} | Loss: {accum_loss / n_obs :.3f}"
                        )
                        accum_loss = 0.0
                        n_obs = 0.0

                logging.info(
                    f"Valid accuracy for A22: {(correct[0]/total[0]).item() :.3f}"
                )
                logging.info(
                    f"Valid accuracy for EZ: {(correct[1]/total[1]).item() :.3f}"
                )
                logging.info(
                    f"Valid accuracy for blank: {(correct[2]/total[2]).item() :.3f}"
                )

    # We don't have any fancy way to save checkpoints, or stop early or anything.

    # Create the output path if necessary.
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    # Save the two models and the parameters.
    torch.save(net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_transformer.pt")
    with open(f"{MODEL_DIR}/{MODEL_NAME}_model_params.json", "w") as f:
        json.dump(
            {
                # Convolutional params
                "conv_layers": CONV_LAYERS,
                "conv_channels": CONV_HIDDEN_CHANNELS,
                "avg_pool_size": AVG_POOL_SIZE,
                # Linear params
                "linear_size": LINEAR_SIZE,
                "linear_output_size": OUTPUT_SIZE,
                # Training params
                "data_size": DATA_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
            },
            f,
        )
