from frameID.net import FrameNet
from frameID.data import ContrastiveFrameDataset
from frameID.metrics import ContrastiveLoss

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import logging

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
)

# Resourcing
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"Using {device}")
NUM_WORKERS = 0

# Model Design
HIDDEN_CHANNELS = 32
LINEAR_SIZE = 32
OUTPUT_SIZE = 8

# Training Details
BATCH_SIZE = 32
EPOCHS = 2
WRITE_EVERY_N = 50
OPTIMIZER = "AdamW"

# Setup optimizer, transforms for images.

opt_class = getattr(torch.optim, OPTIMIZER)

trs = transforms.Compose(
    [
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(1, 1.4)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomResizedCrop(
            size=(144, 256), scale=(0.8, 1), ratio=(1.77, 1.78)
        ),
    ]
)

ds = ContrastiveFrameDataset("data/ravens-lions", trs=trs, ext=".jpg")
train_loader = DataLoader(
    ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True
)
logging.info(f"Batches: {len(train_loader)}")

# Setup network

net = FrameNet(
    hidden_channels=HIDDEN_CHANNELS,
    conv_layers=3,
    fc_size=LINEAR_SIZE,
    output_size=OUTPUT_SIZE,
)

net.to(device)
logging.info(f"Network Weights: {net.num_params()}")

optimizer = opt_class(filter(lambda p: p.requires_grad, net.parameters()))
criterion = ContrastiveLoss(batch_size=BATCH_SIZE).to(device)

for epoch in range(EPOCHS):

    logging.info(f"Starting epoch {epoch+1} of {EPOCHS}")

    accum_loss = 0.0
    n_obs = 0

    for i, data in enumerate(train_loader):

        optimizer.zero_grad()

        x = torch.cat((data["x_t1"], data["x_t2"]), dim=0).to(device)
        res = net(x)

        loss, _, _ = criterion(res)

        loss.backward()
        optimizer.step()

        accum_loss = accum_loss + loss.item()
        n_obs = x.shape[0]

        if i % WRITE_EVERY_N == WRITE_EVERY_N - 1:

            logging.info(
                f"Epoch {epoch+1} | Batch {i+1} | Loss: {accum_loss / n_obs :.3f}"
            )
            accum_loss = 0.0
            n_obs = 0.0
