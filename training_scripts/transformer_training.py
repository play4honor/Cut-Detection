from frameID.net import LovieNet
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
N_LAYERS = 2
INPUT_SIZE = 32
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1
SEQ_LENGTH = 256

# Training Details
BATCH_SIZE = 128
DROPOUT = 0.1
EPOCHS = 7
WRITE_EVERY_N = 500
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

    net = LovieNet(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    )

    net.to(device)
    logging.info(f"{type(net).__name__} Weights: {net.num_params()}")

    optimizer = opt_class(
        filter(lambda p: p.requires_grad, net.parameters()),
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

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
            # mask = data["mask"].to(device)
            labels = data["y"].squeeze().to(device).reshape([-1, OUTPUT_SIZE])
            # weights = data["weight"].to(device)
            pred = net(x).reshape([-1, OUTPUT_SIZE])

            # mask padding
            mask = labels <= 1

            loss = criterion(pred[mask], labels[mask])

            weighted_loss = torch.sum(loss)

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

        correct = torch.zeros([max(OUTPUT_SIZE, 2)]).to(device)
        total = torch.zeros([max(OUTPUT_SIZE, 2)]).to(device)

        net.eval()
        with torch.no_grad():

            for i, data in enumerate(valid_loader):

                x = data["x"].to(device)
                # mask = data["mask"].to(device)
                labels = data["y"].squeeze().to(device)
                # weights = data["weight"].to(device)
                pred = net(x)

                loss = criterion(
                    torch.reshape(pred, [-1, OUTPUT_SIZE]),
                    torch.reshape(labels, [-1, OUTPUT_SIZE]),
                )

                weighted_loss = torch.sum(loss.view([BATCH_SIZE, SEQ_LENGTH]) * 1)

                pc = pred > 0

                for i in range(max(OUTPUT_SIZE, 2)):

                    correct[i] += torch.sum(pc[labels == i] == i)
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
                f"Valid accuracy for A22: {(correct[0]/total[0]).item() :.5f} | {((total[0] - correct[0]) / 256).item()} total errors"
            )
            logging.info(
                f"Valid accuracy for EZ: {(correct[1]/total[1]).item() :.5f} | {((total[1] - correct[1]) / 256).item()} total errors"
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
                "n_layers": N_LAYERS,
                "input_size": INPUT_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "output_size": OUTPUT_SIZE,
                "dropout": DROPOUT,
                # Training params
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
            },
            f,
        )
