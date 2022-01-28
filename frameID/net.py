import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import json
import os

package_directory = os.path.dirname(os.path.abspath(__file__))


class CNNLayer(nn.Module):
    """
    Small module for building convolutional networks.
    """

    def __init__(
        self, conv_args: dict, max_pool_args: dict, activation=nn.ReLU, batch_norm=True
    ):

        super(CNNLayer, self).__init__()

        self.batch_norm = batch_norm

        self.conv = nn.Conv2d(**conv_args)
        self.activation = activation()
        self.max_pool = nn.MaxPool2d(**max_pool_args)

        if self.batch_norm:
            self.bn = nn.BatchNorm2d(conv_args["out_channels"])
        else:
            self.bn = nn.Identity()

    def forward(self, x):

        x = self.conv(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = self.bn(x)

        return x


class FCLayer(nn.Module):
    """
    Small module for building fully-connected networks.
    """

    def __init__(self, linear_args: dict, activation=nn.ReLU, batch_norm=True):

        super(FCLayer, self).__init__()

        self.batch_norm = batch_norm

        self.linear = nn.Linear(**linear_args)
        self.activation = activation()

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(linear_args["out_features"])
        else:
            self.bn = nn.Identity()

    def forward(self, x):

        x = self.linear(x)
        x = self.activation(x)
        x = self.bn(x)

        return x


class FrameConvNet(nn.Module):
    """
    Generic convolutional network. This will be pre-trained on the constrastive
    learning task, then trained more once we have labels.
    """

    def __init__(
        self,
        input_channels=3,
        hidden_channels=32,
        n_conv_layers=3,
        average_pool_size=1,
        output_size=32,
    ):

        super(FrameConvNet, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.n_conv_layers = n_conv_layers

        self.conv_layers = nn.ModuleList()
        self.average_pool = nn.AdaptiveAvgPool2d(average_pool_size)

        # Add the initial layer.
        self.conv_layers.append(
            CNNLayer(
                conv_args={
                    "in_channels": self.input_channels,
                    "out_channels": self.hidden_channels,
                    "kernel_size": 3,
                    "padding": 1,
                },
                max_pool_args={"kernel_size": 3},
                activation=nn.ReLU,
                batch_norm=True,
            )
        )

        # Add the remaining convolutional layers.
        for _ in range(self.n_conv_layers - 1):

            self.conv_layers.append(
                CNNLayer(
                    conv_args={
                        "in_channels": self.hidden_channels,
                        "out_channels": self.hidden_channels,
                        "kernel_size": 3,
                        "padding": 1,
                    },
                    max_pool_args={"kernel_size": 3},
                    activation=nn.ReLU,
                    batch_norm=True,
                )
            )

        # Add a fully-connected layer at the bottom to produce a reasonably-sized
        # latent representation.
        self.mapping_layer = nn.Linear(
            (average_pool_size ** 2) * self.hidden_channels, output_size
        )

    def forward(self, x):

        # Run through convolutions
        for layer in self.conv_layers:

            x = layer(x)

        # Average pool (or possibly not)
        x = self.average_pool(x)
        x = torch.reshape(x, [x.shape[0], -1])
        x = self.mapping_layer(x)
        x = F.relu(x)

        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FrameLinearNet(nn.Module):
    """
    Small all-FC network designed to sit on top of the conv net. Used either
    for a projection head for constrastive learning, or for getting class probabilities
    for the final model.
    """

    def __init__(
        self,
        n_layers: int = 3,
        input_size: int = 32,
        hidden_size: int = 32,
        output_size: int = 8,
    ):

        super(FrameLinearNet, self).__init__()

        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.ModuleList()

        # Add the FC layers. This sucks.
        input_sizes = [self.input_size] + [self.hidden_size] * (self.n_layers - 1)
        output_sizes = [self.hidden_size] * (self.n_layers - 1) + [self.output_size]
        activations = [nn.ReLU] * (self.n_layers - 1) + [nn.Identity]
        bnorm = [True] * (self.n_layers - 1) + [False]

        for in_size, out_size, act, bn in zip(
            input_sizes, output_sizes, activations, bnorm
        ):
            self.layers.append(
                FCLayer(
                    linear_args={"in_features": in_size, "out_features": out_size},
                    activation=act,
                    batch_norm=bn,
                )
            )

    def forward(self, x):

        for layer in self.layers:

            x = layer(x)

        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class NagyNet(nn.Module):
    def __init__(
        self,
        net_size: int,
        output_size: int,
        n_layers: int,
        dropout: float,
        layer_args: dict,
    ):

        super().__init__()

        self.position_encoding = PositionalEncoding(net_size, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            net_size, dropout=dropout, **layer_args, batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.transformer_activation = nn.ReLU()
        self.fc1 = nn.Linear(net_size, layer_args["dim_feedforward"])
        self.fc_activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(layer_args["dim_feedforward"])
        self.output_layer = nn.Linear(layer_args["dim_feedforward"], output_size)

    def forward(self, x, mask):

        x = self.position_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.transformer_activation(x)
        x = self.fc1(x)
        x = self.fc_activation(x)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)

        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Function to slightly simplify loading networks and making a single callable.
# Optionally only include the conv net, which might be a little confusing.
def load_and_glue_nets(param_file, conv_file, linear_file=None):

    with open(param_file, "r") as f:
        model_params = json.load(f)

    conv_net = FrameConvNet(
        hidden_channels=model_params["conv_channels"],
        n_conv_layers=model_params["conv_layers"],
        average_pool_size=model_params["avg_pool_size"],
        output_size=model_params["conv_output_size"],
    )
    conv_state_dict = torch.load(conv_file, map_location="cpu")
    conv_net.load_state_dict(conv_state_dict)

    if linear_file is not None:

        linear_net = FrameLinearNet(
            n_layers=model_params["linear_layers"],
            input_size=model_params["conv_output_size"],
            hidden_size=model_params["linear_size"],
            output_size=model_params["linear_output_size"],
        )
        linear_state_dict = torch.load(linear_file, map_location="cpu")
        linear_net.load_state_dict(linear_state_dict)

        net = nn.Sequential(conv_net, linear_net)

    else:
        net = conv_net

    return net, model_params


# Load a specific network that's included in the module.
def load_default_net():

    conv_file = os.path.join(
        package_directory, "prod_net", "init_model_classifier_conv.pt"
    )
    linear_file = os.path.join(
        package_directory, "prod_net", "init_model_classifier_linear.pt"
    )
    params_file = os.path.join(
        package_directory, "prod_net", "init_model_model_params.json"
    )

    return load_and_glue_nets(params_file, conv_file, linear_file)


def load_transformer(
    conv_config_path, conv_model_path, transformer_config_path, transformer_model_path
):

    with open(conv_config_path, "r") as f:
        conv_config = json.load(f)

    conv_net = FrameConvNet(
        hidden_channels=conv_config["conv_channels"],
        n_conv_layers=conv_config["conv_layers"],
        average_pool_size=conv_config["avg_pool_size"],
        output_size=conv_config["conv_output_size"],
    )
    conv_state_dict = torch.load(conv_model_path, map_location="cpu")
    conv_net.load_state_dict(conv_state_dict)

    with open(transformer_config_path, "r") as f:
        transformer_config = json.load(f)

    transformer = NagyNet(
        net_size=transformer_config["network_size"],
        output_size=transformer_config["output_size"],
        n_layers=transformer_config["n_layers"],
        dropout=transformer_config["dropout"],
        layer_args={
            "nhead": transformer_config["n_heads"],
            "dim_feedforward": transformer_config["ff_size"],
        },
    )
    transformer_state_dict = torch.load(transformer_model_path, map_location="cpu")
    transformer.load_state_dict(transformer_state_dict)

    return conv_net.eval(), transformer.eval()


if __name__ == "__main__":

    pass
