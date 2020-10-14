# -*- coding: utf-8 -*-
import math
import functools

import torch
import torch.nn as nn


__all__ = ["resnet_evonorm"]


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (
            sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i))
        )


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


@torch.jit.script
def instance_std(x, eps):
    var = torch.var(x, dim=(2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)


@torch.jit.script
def group_std(x, eps):
    N, C, H, W = x.size()
    groups = 32
    groups = C if groups > C else groups
    x = x.view(N, groups, C // groups, H, W)
    var = torch.var(x, dim=(2, 3, 4), keepdim=True).expand_as(x)
    return torch.sqrt(var.add(eps)).view(N, C, H, W)


class EvoNorm2D(nn.Module):
    def __init__(
        self,
        input,
        non_linear=True,
        version="S0",
        efficient=True,
        affine=True,
        momentum=0.9,
        eps=1e-5,
        groups=32,
        training=True,
    ):
        super(EvoNorm2D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.efficient = efficient
        if self.version == "S0":
            self.swish = MemoryEfficientSwish()
        self.groups = groups
        self.eps = nn.Parameter(torch.FloatTensor([eps]), requires_grad=False)
        if self.version not in ["B0", "S0"]:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
            if self.non_linear:
                self.v = nn.Parameter(torch.ones(1, self.insize, 1, 1))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
            self.register_buffer("v", None)
        self.register_buffer("running_var", torch.ones(1, self.insize, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.version == "S0":
            if self.non_linear:
                if not self.efficient:
                    num = x * torch.sigmoid(
                        self.v * x
                    )  # Original Swish Implementation, however memory intensive.
                else:
                    num = self.swish(
                        x
                    )  # Experimental Memory Efficient Variant of Swish
                return num / group_std(x, eps=self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == "B0":
            if self.training:
                var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                den = torch.max(
                    (var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps)
                )
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        version="S0",
        norm_layer=None,
        use_bn_stat=False,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = functools.partial(
                nn.BatchNorm2d, track_running_stats=use_bn_stat
            )
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.evo = EvoNorm2D(planes, version=version)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.evo(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_cifar(nn.Module):
    def __init__(
        self,
        dataset,
        resnet_size,
        scaling=1,
        save_activations=False,
        use_bn_stat=False,
        version="S0",
    ):
        super(ResNet_cifar, self).__init__()
        self.dataset = dataset
        self.use_bn_stat = use_bn_stat
        self.version = "S0" if version is None else version

        # define model.
        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6

        if resnet_size >= 44:
            raise NotImplementedError("not supported yet.")
        else:
            block_fn = BasicBlock

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=int(16 * scaling),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.evo1 = EvoNorm2D(int(16 * scaling))
        # self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn, planes=int(16 * scaling), block_num=block_nums
        )
        self.layer2 = self._make_block(
            block_fn=block_fn, planes=int(32 * scaling), block_num=block_nums, stride=2
        )
        self.layer3 = self._make_block(
            block_fn=block_fn, planes=int(64 * scaling), block_num=block_nums, stride=2
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(
            in_features=int(64 * scaling * block_fn.expansion),
            out_features=self.num_classes,
        )

        # weight initialization based on layer type.
        self._weight_initialization()

        # a placeholder for activations in the intermediate layers.
        self.save_activations = save_activations
        self.activations = None

    def _decide_num_classes(self):
        if self.dataset == "cifar10" or self.dataset == "svhn":
            return 10
        elif self.dataset == "cifar100":
            return 100
        elif "imagenet" in self.dataset:
            return 1000
        elif "femnist" == self.dataset:
            return 62

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(mean=0, std=0.01)
            #     m.bias.data.zero_()

    def _make_block(self, block_fn, planes, block_num, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_fn.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    planes * block_fn.expansion, track_running_stats=self.use_bn_stat
                ),
            )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                use_bn_stat=self.use_bn_stat,
                version=self.version,
            )
        )
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(block_fn(in_planes=self.inplanes, planes=planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.evo1(x)

        x = self.layer1(x)
        activation1 = x.clone()
        x = self.layer2(x)
        activation2 = x.clone()
        x = self.layer3(x)
        activation3 = x.clone()
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if self.save_activations:
            self.activations = [activation1, activation2, activation3]
        return x


def resnet_evonorm(conf, arch=None):
    """Constructs a ResNet-18 model."""
    resnet_size = int(
        (arch if arch is not None else conf.arch).replace("resnet_evonorm", "")
    )
    dataset = conf.data

    if "cifar" in conf.data or "svhn" in conf.data:
        model = ResNet_cifar(
            dataset=dataset,
            resnet_size=resnet_size,
            scaling=conf.resnet_scaling,
            use_bn_stat=not conf.freeze_bn,
            version=conf.evonorm_version,
        )
    else:
        raise NotImplementedError
    return model


if __name__ == "__main__":

    print("cifar10")
    net = ResNet_cifar(dataset="cifar10", resnet_size=20, scaling=1, use_bn_stat=False)
    print(net)
    x = torch.randn(16, 3, 32, 32)
    y = net(x)
    print(y.shape)
