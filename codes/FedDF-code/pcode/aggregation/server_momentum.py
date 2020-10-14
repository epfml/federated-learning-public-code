# -*- coding: utf-8 -*-
import torch

from pcode.utils.tensor_buffer import TensorBuffer


def aggregate(conf, master_model, fedavg_model, client_models, flatten_local_models):
    # perform the server momentum (either heavy-ball momentum or nesterov momentum)
    fl_aggregate = conf.fl_aggregate

    assert "server_momentum_factor" in fl_aggregate

    # start the server momentum acceleration.
    current_model_tb = TensorBuffer(list(fedavg_model.parameters()))
    previous_model_tb = TensorBuffer(list(master_model.parameters()))

    # get the update direction.
    update = previous_model_tb.buffer - current_model_tb.buffer

    # using server momentum for the update.
    if not hasattr(conf, "server_momentum_buffer"):
        conf.server_momentum_buffer = torch.zeros_like(update)
    conf.server_momentum_buffer.mul_(fl_aggregate["server_momentum_factor"]).add_(
        update
    )
    previous_model_tb.buffer.add_(-conf.server_momentum_buffer)

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
