from frameID.net import FrameConvNet, FrameLinearNet
from frameID.data import SupervisedFrameDataset

import torch
from torch.utils.data import DataLoader, Subset

import torchvision.transforms as transforms

import logging
import os
from itertools import chain

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
)

# Resourcing
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Using {device}")
NUM_WORKERS = 0

# TODO Read all this from a config file and save the file with the model.

# Conv Net Design
CONV_LAYERS = 3
CONV_HIDDEN_CHANNELS = 32

# Output Network Design
LINEAR_LAYERS = 2
LINEAR_SIZE = 32
OUTPUT_SIZE = 3

# Training Details
BATCH_SIZE = 10
EPOCHS = 20
WRITE_EVERY_N = 10
OPTIMIZER = "AdamW"

MODEL_DIR = "./models"
PRETRAINED_MODEL_NAME = "init_model"
LOAD_CONV_NET = False

# Setup optimizer, transforms for images.

opt_class = getattr(torch.optim, OPTIMIZER)


# Initialize the dataset class and then split into train/valid.
ds = SupervisedFrameDataset("data/ravens-lions", ext=".jpg")

torch.random.manual_seed(100)
idx_perm = torch.randperm(len(ds))
train_idx = idx_perm[:300].tolist()
valid_idx = idx_perm[300:].tolist()

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

conv_net = FrameConvNet(hidden_channels=CONV_HIDDEN_CHANNELS, n_conv_layers=CONV_LAYERS)

if LOAD_CONV_NET:
    logging.info(
        f"Loading pre-trained conv net from: {MODEL_DIR}/{PRETRAINED_MODEL_NAME}_conv.pt"
    )
    state_dict = torch.load(
        f"{MODEL_DIR}/{PRETRAINED_MODEL_NAME}_conv.pt", map_location="cpu"
    )
    conv_net.load_state_dict(state_dict)


linear_net = FrameLinearNet(
    n_layers=LINEAR_LAYERS,
    input_size=CONV_HIDDEN_CHANNELS,
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

    correct = 0
    total = 0

    conv_net.eval()
    linear_net.eval()
    with torch.no_grad():

        for i, data in enumerate(valid_loader):

            x = data["x"].to(device)
            labels = data["y"].squeeze().to(device)
            intermediate = conv_net(x)
            pred = linear_net(intermediate)

            loss = criterion(pred, labels)

            predicted_classes = torch.max(pred, dim=1)[1]
            correct += torch.eq(labels, predicted_classes).sum().item()
            total += labels.shape[0]

            accum_loss += loss.item()
            n_obs += x.shape[0]

            if i % WRITE_EVERY_N == WRITE_EVERY_N - 1:

                logging.info(
                    f"Validation: Epoch {epoch+1} | Batch {i+1} | Loss: {accum_loss / n_obs :.3f}"
                )
                accum_loss = 0.0
                n_obs = 0.0

        logging.info(f"Validation accuracy: {correct/total :.3f}")
