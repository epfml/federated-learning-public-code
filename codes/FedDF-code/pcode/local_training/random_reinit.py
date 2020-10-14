# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn


def random_reinit_model(conf, model):
    if conf.random_reinit_local_model is None:
        return
    else:
        assert "resnet" in conf.arch or "federated_averaging_cnn" in conf.arch
        names = [
            (_name, _module)
            for _name, _module in model.named_modules()
            if (
                len(list(_module.children())) == 0
                and "bn" not in _name
                and ("conv" in _name or "classifier" in _name)
            )
        ]

        if conf.random_reinit_local_model == "last":
            name_module = names[-1]
            weight_initialization(name_module[1])
        elif "random" in conf.random_reinit_local_model:
            name_module = names[conf.random_state.choice(len(names))]
            weight_initialization(name_module[1])
        else:
            raise NotImplementedError

        conf.logger.log(
            f"Worker-{conf.graph.worker_id} (client-{conf.graph.client_id}) received the model from Master and reinitialize layer={name_module[0]}."
        )


def weight_initialization(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0, std=0.01)
        # torch.nn.init.xavier_uniform(m.weight.data)
