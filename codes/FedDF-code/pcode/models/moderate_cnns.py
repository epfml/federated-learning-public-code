# -*- coding: utf-8 -*-
import torch.nn as nn


__all__ = ["moderate_cnn"]


class ModerateCNN(nn.Module):
    def __init__(self, w_conv_bias=False, w_fc_bias=True, save_activations=True):
        super(ModerateCNN, self).__init__()

        # Conv Layer block 1
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=w_conv_bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=w_conv_bias,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Conv Layer block 2
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=w_conv_bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=w_conv_bias,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
        )

        # Conv Layer block 3
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=w_conv_bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=w_conv_bias,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 512, bias=w_fc_bias),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=w_fc_bias),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10, bias=w_fc_bias),
        )

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def forward(self, x):
        x = self.conv_layer1(x)
        activation1 = x
        x = self.conv_layer2(x)
        activation2 = x
        x = self.conv_layer3(x)
        activation3 = x

        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        if self.save_activations:
            self.activations = [activation1, activation2, activation3]
        return x


def moderate_cnn(conf):
    dataset = conf.data

    if "cifar" in dataset or dataset == "svhn":
        return ModerateCNN(w_conv_bias=conf.w_conv_bias, w_fc_bias=conf.w_fc_bias)
    else:
        raise NotImplementedError(f"not supported yet.")


if __name__ == "__main__":
    import torch

    print("cifar10")
    net = ModerateCNN()
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.shape)
