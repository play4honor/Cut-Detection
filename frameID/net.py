import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLayer(nn.Module):
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


class FrameNet(nn.Module):
    def __init__(
        self,
        input_channels=3,
        hidden_channels=32,
        conv_layers=3,
        fc_size=32,
        output_size=8,
    ):

        super(FrameNet, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.conv_layers = conv_layers
        self.fc_size = fc_size
        self.output_size = output_size

        self.conv_layers = nn.ModuleList()
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.ModuleList()

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
        for _ in range(conv_layers - 1):

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

        # Add the FC layers. This sucks.
        input_sizes = [self.hidden_channels, self.fc_size, self.fc_size]
        output_sizes = [self.fc_size, self.fc_size, self.output_size]
        activations = [nn.ReLU, nn.ReLU, nn.Identity]
        bnorm = [True, True, False]

        for in_size, out_size, act, bn in zip(
            input_sizes, output_sizes, activations, bnorm
        ):
            self.fc_layers.append(
                FCLayer(
                    linear_args={"in_features": in_size, "out_features": out_size},
                    activation=act,
                    batch_norm=bn,
                )
            )

    def forward(self, x):

        # Run through convolutions
        for layer in self.conv_layers:

            x = layer(x)

        # Average pool
        x = self.average_pool(x)
        x = torch.reshape(x, [x.shape[0], x.shape[1]])

        # Through linear layers

        for layer in self.fc_layers:

            x = layer(x)

        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":

    net = FrameNet()

    # Simulate a batch of images.
    batch = torch.randn([32, 3, 144, 256])

    output = net(batch)

    print(output.shape)
