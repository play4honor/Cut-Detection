from frameID.net import load_and_glue_nets

import logging

import torch

# Logging and cuda
logging.basicConfig(
    level="INFO",
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
)

# Create dataset and network
MODEL_DIR = "./models"
MODEL_NAME = "init_model"

net, params = load_and_glue_nets(
    f"{MODEL_DIR}/{MODEL_NAME}_model_params.json",
    f"{MODEL_DIR}/{MODEL_NAME}_classifier_conv.pt",
    f"{MODEL_DIR}/{MODEL_NAME}_classifier_linear.pt",
)

net.eval()

example_data = torch.randn([1, 3, 144, 256])

traced_net = torch.jit.trace(net, example_data)

logging.info(traced_net.graph)

logging.info(f"Normal: {net(example_data)}")
logging.info(f"Traced: {traced_net(example_data)}")

traced_net.save(f"{MODEL_DIR}/saved_model_trace.pt")
