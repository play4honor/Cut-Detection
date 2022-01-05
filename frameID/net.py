import torch
import torch.nn as nn
import torch.nn.functional as F

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
        sequences=False,
    ):

        super(FrameConvNet, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.n_conv_layers = n_conv_layers

        self.conv_layers = nn.ModuleList()
        self.average_pool = nn.AdaptiveAvgPool2d(average_pool_size)

        self.sequences = sequences

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

    def forward(self, x):

        # Run through convolutions
        for layer in self.conv_layers:

            x = layer(x)

        # Average pool (or possibly not)
        x = self.average_pool(x)

        if self.sequences:
            x = torch.reshape(x, [x.shape[0], x.shape[1], -1])
        else:
            x = torch.reshape(x, [x.shape[0], -1])

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


class FeedForward(nn.Module):
    def __init__(self, input_dims, ff_dims, n_classes):

        super().__init__()

        self.fc1 = FCLayer(
            linear_args={"in_features": input_dims, "out_features": ff_dims},
            batch_norm=False,
        )
        self.fc2 = FCLayer(
            linear_args={"in_features": ff_dims, "out_features": n_classes},
            batch_norm=False,
            activation=nn.Identity,
        )

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)

        return x


class FunNameModel(nn.Module):
    def __init__(
        self,
        conv_net,
        tr_encoder,
        n_enc_layers,
        output_net,
    ):

        super().__init__()

        self.conv_net = conv_net
        self.tr_encoder = tr_encoder
        self.n_enc_layers = (n_enc_layers,)
        self.output_net = output_net

        self.transformer_encoder = nn.TransformerEncoder(self.tr_encoder, n_enc_layers)

    def forward(self, x, mask):

        input_shape = x.shape

        x = torch.reshape(x, [-1, input_shape[2], input_shape[3], input_shape[4]])
        x = self.conv_net(x)
        x = torch.reshape(x, [input_shape[0], input_shape[1], -1])
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.output_net(x)

        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Function to slightly simplify loading networks and making a single callable.
def load_and_glue_nets(param_file, conv_file, linear_file):

    with open(param_file, "r") as f:
        model_params = json.load(f)

    conv_net = FrameConvNet(
        hidden_channels=model_params["conv_channels"],
        n_conv_layers=model_params["conv_layers"],
        average_pool_size=model_params["avg_pool_size"],
    )
    conv_state_dict = torch.load(conv_file, map_location="cpu")
    conv_net.load_state_dict(conv_state_dict)

    linear_net = FrameLinearNet(
        n_layers=model_params["linear_layers"],
        input_size=model_params["conv_channels"] * (model_params["avg_pool_size"] ** 2),
        hidden_size=model_params["linear_size"],
        output_size=model_params["linear_output_size"],
    )
    linear_state_dict = torch.load(linear_file, map_location="cpu")
    linear_net.load_state_dict(linear_state_dict)

    net = nn.Sequential(conv_net, linear_net)

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


if __name__ == "__main__":

    conv_net = FrameConvNet(average_pool_size=4, sequences=True)

    enc_layer = nn.TransformerEncoderLayer(16 * 32, 4, 256, batch_first=True)
    output_layer = FeedForward(16 * 32, 128, 3)

    net = FunNameModel(conv_net, enc_layer, 6, output_layer)

    print(net.num_params())

    sample = torch.randn([4, 128, 3, 144, 256])
    mask = torch.zeros([4, 128]).bool()

    out = net(sample, mask)

    print(out.shape)
