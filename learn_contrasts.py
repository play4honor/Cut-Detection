from frameID.net import FrameNet
from frameID.data import ContrastiveFrameDataset
from frameID.metrics import ContrastiveLoss

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

# Resourcing
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
NUM_WORKERS = 0

# Model Design
HIDDEN_CHANNELS = 32
LINEAR_SIZE = 32
OUTPUT_SIZE = 8

# Training Details
BATCH_SIZE = 32
EPOCHS = 5
WRITE_EVERY_N = 150
OPTIMIZER = "AdamW"

# Setup optimizer, transforms for images.

opt_class = getattr(torch.optim, OPTIMIZER)

trs = transforms.Compose(
        [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(1, 1.4)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomResizedCrop(size=(144, 256), scale=(0.8, 1), ratio=(1.77, 1.78))
        ]
    )

ds = ContrastiveFrameDataset("data/ravens-lions", trs=trs, ext=".jpg")
trainLoader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
print(f"Batches: {len(trainLoader)}")

# Setup network

net = FrameNet(
    hidden_channels=HIDDEN_CHANNELS,
    conv_layers=3,
    fc_size=LINEAR_SIZE,
    output_size=OUTPUT_SIZE
)

net.to(device)
print(f"Network Weights: {net.num_params()}")

optimizer = opt_class(filter(lambda p: p.requires_grad, net.parameters()))
criterion = ContrastiveLoss()