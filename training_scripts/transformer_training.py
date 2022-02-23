from frameID.net import NagyNet
from frameID.data import CompressedDataset

import torch
from torch.utils.data import DataLoader

import logging
import os
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

# Transformer design
N_ATTENTION_HEADS = 2
FF_SIZE = 256
N_LAYERS = 3
NETWORK_SIZE = 32
OUTPUT_SIZE = 3
SEQ_LENGTH = 256

# Training Details
DATA_SIZE = 150_000
BATCH_SIZE = 128
DROPOUT = 0.1
EPOCHS = 3
WRITE_EVERY_N = 1000
OPTIMIZER = "AdamW"

MODEL_DIR = "./models"
MODEL_NAME = "frame_compression_model"
LOAD_CONV_NET = False

# Setup optimizer.
opt_class = getattr(torch.optim, OPTIMIZER)

ds = CompressedDataset("data/training_compressed_frames.pt", seq_length=SEQ_LENGTH)
vds = CompressedDataset("data/validation_compressed_frames.pt", seq_length=SEQ_LENGTH)

train_loader = DataLoader(
    ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=False,
)

valid_loader = DataLoader(
    vds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=False,
)

logging.info(
    f"Training Batches: {len(train_loader)} | Validation Batches: {len(valid_loader)}"
)

if __name__ == "__main__":

    net = NagyNet(
        net_size=NETWORK_SIZE,
        output_size=OUTPUT_SIZE,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        layer_args={
            "nhead": N_ATTENTION_HEADS,
            "dim_feedforward": FF_SIZE,
        },
    )

    net.to(device)
    logging.info(f"NagyNet Weights: {net.num_params()}")

    optimizer = opt_class(
        filter(lambda p: p.requires_grad, net.parameters()),
    )
    criterion = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=3)

    # Training loop
    for epoch in range(EPOCHS):

        # Training

        logging.info(f"Starting epoch {epoch+1} of {EPOCHS}")

        accum_loss = 0.0
        n_obs = 0

        net.train()
        for i, data in enumerate(train_loader):

            optimizer.zero_grad()

            x = data["x"].to(device)
            mask = data["mask"].to(device)
            labels = data["y"].squeeze().to(device)
            weights = data["weight"].to(device)
            pred = net(x, mask)

            loss = criterion(
                torch.reshape(pred, [-1, OUTPUT_SIZE]),
                torch.reshape(labels, [-1]),
            )

            weighted_loss = torch.sum(loss.view([BATCH_SIZE, SEQ_LENGTH]) * weights)

            weighted_loss.backward()
            optimizer.step()

            accum_loss += weighted_loss.item()
            n_obs += x.shape[0] * x.shape[1]

            if i % WRITE_EVERY_N == WRITE_EVERY_N - 1:

                logging.info(
                    f"Epoch {epoch+1} | Batch {i+1} | Loss: {accum_loss / n_obs :.5f}"
                )
                accum_loss = 0.0
                n_obs = 0.0

        # Validation

        logging.info(f"Starting validation for epoch {epoch+1}")

        accum_loss = 0.0
        n_obs = 0

        correct = torch.zeros([3]).to(device)
        total = torch.zeros([3]).to(device)

        net.eval()
        with torch.no_grad():

            for i, data in enumerate(valid_loader):

                x = data["x"].to(device)
                mask = data["mask"].to(device)
                labels = data["y"].squeeze().to(device)
                weights = data["weight"].to(device)
                pred = net(x, mask)

                loss = criterion(
                    torch.reshape(pred, [-1, OUTPUT_SIZE]),
                    torch.reshape(labels, [-1]),
                )

                weighted_loss = torch.sum(loss.view([BATCH_SIZE, SEQ_LENGTH]) * weights)

                pc = torch.max(pred, dim=2)[1]

                for i in range(3):

                    correct[i] += torch.sum(pc[labels == i] == labels[labels == i])
                    total[i] += torch.sum(labels == i)

                accum_loss += weighted_loss.item()
                n_obs += x.shape[0]

                if i % WRITE_EVERY_N == WRITE_EVERY_N - 1:

                    logging.info(
                        f"Validation: Epoch {epoch+1} | Batch {i+1} | Loss: {accum_loss / n_obs :.5f}"
                    )
                    accum_loss = 0.0
                    n_obs = 0.0

            logging.info(
                f"Valid accuracy for A22: {(correct[0]/total[0]).item() :.5f} | {total[0] - correct[0]} total errors"
            )
            logging.info(
                f"Valid accuracy for EZ: {(correct[1]/total[1]).item() :.5f} | {total[1] - correct[1]} total errors"
            )
            logging.info(
                f"Valid accuracy for blank: {(correct[2]/total[2]).item() :.5f} | {total[2] - correct[2]} total errors"
            )

    # We don't have any fancy way to save checkpoints, or stop early or anything.

    # Create the output path if necessary.
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    # Save the two models and the parameters.
    torch.save(net.state_dict(), f"{MODEL_DIR}/{MODEL_NAME}_transformer.pt")
    with open(f"{MODEL_DIR}/{MODEL_NAME}_transformer_model_params.json", "w") as f:
        json.dump(
            {
                # Model params
                "n_heads": N_ATTENTION_HEADS,
                "ff_size": FF_SIZE,
                "n_layers": N_LAYERS,
                "network_size": NETWORK_SIZE,
                "output_size": OUTPUT_SIZE,
                "dropout": DROPOUT,
                # Training params
                "data_size": DATA_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
            },
            f,
        )
