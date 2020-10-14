# -*- coding: utf-8 -*-
import torch

from pcode.utils.tensor_buffer import TensorBuffer


def aggregate(conf, master_model, fedavg_model, client_models, flatten_local_models):
    # perform the server Adam.
    # Following the setup in the paper, we use momentum of 0.9,
    # numerical stability constant epsilon to be 0.01,
    # the beta_2 is set to 0.99.
    # The suggested server_lr in the original paper is 0.1
    fl_aggregate = conf.fl_aggregate

    assert "server_lr" in fl_aggregate
    beta_2 = fl_aggregate["beta_2"] if "beta_2" in fl_aggregate else 0.99

    # start the server momentum acceleration.
    current_model_tb = TensorBuffer(list(fedavg_model.parameters()))
    previous_model_tb = TensorBuffer(list(master_model.parameters()))

    # get the update direction.
    update = previous_model_tb.buffer - current_model_tb.buffer

    # using server momentum for the update.
    if not hasattr(conf, "second_server_momentum_buffer"):
        conf.second_server_momentum_buffer = torch.zeros_like(update)
    conf.second_server_momentum_buffer.mul_(beta_2).add_((1 - beta_2) * (update ** 2))
    previous_model_tb.buffer.add_(
        -fl_aggregate["server_lr"]
        * update
        / (torch.sqrt(conf.second_server_momentum_buffer) + 0.01)
    )

    # update the master_model (but will use the bn stats from the fedavg_model)
    master_model = fedavg_model
    _model_param = list(master_model.parameters())
    previous_model_tb.unpack(_model_param)

    # free the memory.
    torch.cuda.empty_cache()

    # a temp hack (only for debug reason).
    client_models = dict(
        (used_client_arch, master_model.cpu())
        for used_client_arch in conf.used_client_archs
    )
    return client_models
