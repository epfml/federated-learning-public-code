# -*- coding: utf-8 -*-
import torch.distributed as dist

import pcode.models as models


def define_model(
    conf,
    show_stat=True,
    to_consistent_model=True,
    use_complex_arch=True,
    client_id=None,
    arch=None,
):
    arch, model = define_cv_classification_model(
        conf, client_id, use_complex_arch, arch
    )

    # consistent the model.
    if to_consistent_model:
        consistent_model(conf, model)

    # get the model stat info.
    if show_stat:
        get_model_stat(conf, model, arch)
    return arch, model


"""define loaders for different models."""


def determine_arch(conf, client_id, use_complex_arch):
    # the client_id starts from 1.
    _id = client_id if client_id is not None else 0
    if use_complex_arch:
        if _id == 0:
            arch = conf.arch_info["master"]
        else:
            archs = conf.arch_info["worker"]
            if len(conf.arch_info["worker"]) == 1:
                arch = archs[0]
            else:
                assert "num_clients_per_model" in conf.arch_info
                assert (
                    conf.arch_info["num_clients_per_model"] * len(archs)
                    == conf.n_clients
                )
                arch = archs[int((_id - 1) / conf.arch_info["num_clients_per_model"])]
    else:
        arch = conf.arch
    return arch


def define_cv_classification_model(conf, client_id, use_complex_arch, arch):
    # determine the arch.
    arch = determine_arch(conf, client_id, use_complex_arch) if arch is None else arch
    # use the determined arch to init the model.
    if "wideresnet" in arch:
        model = models.__dict__["wideresnet"](conf)
    elif "resnet" in arch and "resnet_evonorm" not in arch:
        model = models.__dict__["resnet"](conf, arch=arch)
    elif "resnet_evonorm" in arch:
        model = models.__dict__["resnet_evonorm"](conf, arch=arch)
    elif "regnet" in arch.lower():
        model = models.__dict__["regnet"](conf, arch=arch)
    elif "densenet" in arch:
        model = models.__dict__["densenet"](conf)
    elif "vgg" in arch:
        model = models.__dict__["vgg"](conf)
    elif "mobilenetv2" in arch:
        model = models.__dict__["mobilenetv2"](conf)
    elif "shufflenetv2" in arch:
        model = models.__dict__["shufflenetv2"](conf, arch=arch)
    elif "efficientnet" in arch:
        model = models.__dict__["efficientnet"](conf)
    elif "federated_averaging_cnn" in arch:
        model = models.__dict__["simple_cnn"](conf)
    elif "moderate_cnn" in arch:
        model = models.__dict__["moderate_cnn"](conf)
    else:
        model = models.__dict__[arch](conf)
    return arch, model


"""some utilities functions."""


def get_model_stat(conf, model, arch):
    conf.logger.log(
        "\t=> {} created model '{}. Total params: {}M".format(
            "Master"
            if conf.graph.rank == 0
            else f"Worker-{conf.graph.worker_id} (client-{conf.graph.client_id})",
            arch,
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        )
    )


def consistent_model(conf, model):
    """it might because of MPI, the model for each process is not the same.

    This function is proposed to fix this issue,
    i.e., use the  model (rank=0) as the global model.
    """
    conf.logger.log("\tconsistent model for process (rank {})".format(conf.graph.rank))
    cur_rank = conf.graph.rank
    for param in model.parameters():
        param.data = param.data if cur_rank == 0 else param.data - param.data
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
