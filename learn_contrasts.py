from frameID.net import FrameConvNet, FrameLinearNet
from frameID.data import ContrastiveFrameDataset
from frameID.metrics import ContrastiveLoss

import torch
from torch.utils.data import DataLoader

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

# Projection Head Design
LINEAR_LAYERS = 3
LINEAR_SIZE = 32
OUTPUT_SIZE = 8

# Training Details
BATCH_SIZE = 32
EPOCHS = 2
WRITE_EVERY_N = 50
OPTIMIZER = "AdamW"

MODEL_DIR = "./models"
MODEL_NAME = "init_model"

# Setup optimizer, transforms for images.

opt_class = getattr(torch.optim, OPTIMIZER)

trs = transforms.Compose(
    [
        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(1, 1.4)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomResizedCrop(
            size=(144, 256), scale=(0.5, 1), ratio=(1.77, 1.78)
        ),
    ]
)

ds = ContrastiveFrameDataset("data/ravens-lions", trs=trs, ext=".jpg")
train_loader = DataLoader(
    ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True
)
logging.info(f"Batches: {len(train_loader)}")

# Setup networks

conv_net = FrameConvNet(hidden_channels=CONV_HIDDEN_CHANNELS, n_conv_layers=CONV_LAYERS)
linear_net = FrameLinearNet(
    n_layers=LINEAR_LAYERS,
    input_size=CONV_HIDDEN_CHANNELS,
    hidden_size=LINEAR_SIZE,
    output_size=OUTPUT_SIZE,
)

conv_net.to(device)
linear_net.to(device)
logging.info(f"Convolutional Network Weights: {conv_net.num_params()}")
logging.info(f"Projection Head Weights: {linear_net.num_params()}")

# We want the optimizer to optimize both networks.
optimizer = opt_class(
    chain(
        filter(lambda p: p.requires_grad, conv_net.parameters()),
        filter(lambda p: p.requires_grad, linear_net.parameters()),
    )
)
criterion = ContrastiveLoss(batch_size=BATCH_SIZE).to(device)

# Training loop
for epoch in range(EPOCHS):

    logging.info(f"Starting epoch {epoch+1} of {EPOCHS}")

    accum_loss = 0.0
    n_obs = 0

    for i, data in enumerate(train_loader):

        optimizer.zero_grad()

        # Concatenate the two transformations together, then send through both networks.
        x = torch.cat((data["x_t1"], data["x_t2"]), dim=0).to(device)
        intermediate = conv_net(x)
        res = linear_net(intermediate)

        loss, _, _ = criterion(res)

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

# We don't have any fancy way to save checkpoints, or stop early or anything.

# Create the output path if necessary.
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

torch.save(conv_net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_conv.pt")
torch.save(linear_net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_linear.pt")
