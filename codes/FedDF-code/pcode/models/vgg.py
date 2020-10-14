# -*- coding: utf-8 -*-
import math

import torch.nn as nn


__all__ = ["vgg"]


ARCHITECTURES = {
    "O": [4, "M", 8, "M", 16, 16, "M", 32, 32, "M", 32, 32, "M"],
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, nn_arch, dataset, use_bn=True):
        super(VGG, self).__init__()

        # init parameters.
        self.use_bn = use_bn
        self.nn_arch = nn_arch
        self.dataset = dataset
        self.num_classes = self._decide_num_classes()

        # init models.
        self.features = self._make_layers()
        self.intermediate_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
        )
        self.classifier = nn.Linear(512, self.num_classes)

        # weight initialization.
        self._weight_initialization()

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _decide_num_classes(self):
        if self.dataset == "cifar10" or self.dataset == "svhn":
            return 10
        elif self.dataset == "cifar100":
            return 100
        else:
            raise ValueError("not allowed dataset.")

    def _make_layers(self):
        layers = []
        in_channels = 3
        for v in ARCHITECTURES[self.nn_arch]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.use_bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.intermediate_classifier(x)
        x = self.classifier(x)
        return x


class VGG_S(nn.Module):
    def __init__(self, nn_arch, dataset, width=1, use_bn=True, save_activations=False):
        super(VGG_S, self).__init__()

        # init parameters.
        self.use_bn = use_bn
        self.nn_arch = nn_arch
        self.width = width
        self.dataset = dataset
        self.num_classes = self._decide_num_classes()

        # init models.
        self.features = self._make_layers()
        self.classifier = nn.Linear(int(32 * width), self.num_classes)

        # weight initialization.
        self._weight_initialization()

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _decide_num_classes(self):
        if self.dataset == "cifar10" or self.dataset == "svhn":
            return 10
        elif self.dataset == "cifar100":
            return 100
        else:
            raise ValueError("not allowed dataset.")

    def _make_layers(self):
        layers = []
        in_channels = 3
        for v in ARCHITECTURES[self.nn_arch]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_planes = int(v * self.width)
                conv2d = nn.Conv2d(in_channels, out_planes, kernel_size=3, padding=1)
                if self.use_bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg(conf):
    use_bn = "bn" in conf.arch
    dataset = conf.data

    if conf.vgg_scaling is not None:
        return VGG_S(
            nn_arch="O", dataset=dataset, width=conf.vgg_scaling, use_bn=use_bn
        )
    else:
        if "11" in conf.arch:
            return VGG(nn_arch="A", dataset=dataset, use_bn=use_bn)
        elif "13" in conf.arch:
            return VGG(nn_arch="B", dataset=dataset, use_bn=use_bn)
        elif "16" in conf.arch:
            return VGG(nn_arch="D", dataset=dataset, use_bn=use_bn)
        elif "19" in conf.arch:
            return VGG(nn_arch="E", dataset=dataset, use_bn=use_bn)
        else:
            raise NotImplementedError


if __name__ == "__main__":

    def get_n_model_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    width = 8
    net = VGG_S(nn_arch="O", dataset="cifar10", width=width, use_bn=False)
    print(f"VGG with width={width} has n_params={get_n_model_params(net)}M.")

    # x = torch.randn(1, 3, 32, 32)
    # y = net(x)
    # print(y.shape)
