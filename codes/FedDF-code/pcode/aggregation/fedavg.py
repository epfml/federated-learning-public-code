# -*- coding: utf-8 -*-
import copy

import torch

from pcode.utils.module_state import ModuleState
import pcode.master_utils as master_utils
import pcode.aggregation.utils as agg_utils


def _fedavg(clientid2arch, n_selected_clients, flatten_local_models, client_models):
    weights = [
        torch.FloatTensor([1.0 / n_selected_clients]) for _ in range(n_selected_clients)
    ]

    # NOTE: the arch for different local models needs to be the same as the master model.
    # retrieve the local models.
    local_models = {}
    for client_idx, flatten_local_model in flatten_local_models.items():
        _arch = clientid2arch[client_idx]
        _model = copy.deepcopy(client_models[_arch])
        _model_state_dict = client_models[_arch].state_dict()
        flatten_local_model.unpack(_model_state_dict.values())
        _model.load_state_dict(_model_state_dict)
        local_models[client_idx] = _model

    # uniformly average the local models.
    # assume we use the runtime stat from the last model.
    _model = copy.deepcopy(_model)
    local_states = [
        ModuleState(copy.deepcopy(local_model.state_dict()))
        for _, local_model in local_models.items()
    ]
    model_state = local_states[0] * weights[0]
    for idx in range(1, len(local_states)):
        model_state += local_states[idx] * weights[idx]
    model_state.copy_to_module(_model)
    return _model


def fedavg(
    conf,
    clientid2arch,
    n_selected_clients,
    flatten_local_models,
    client_models,
    criterion,
    metrics,
    val_data_loader,
):
    if (
        "server_teaching_scheme" not in conf.fl_aggregate
        or "drop" not in conf.fl_aggregate["server_teaching_scheme"]
    ):
        # directly averaging.
        conf.logger.log(f"No indices to be removed.")
        return _fedavg(
            clientid2arch, n_selected_clients, flatten_local_models, client_models
        )
    else:
        # we will first perform the evaluation.
        # recover the models on the computation device.
        _, local_models = agg_utils.recover_models(
            conf, client_models, flatten_local_models
        )

        # get the weights from the validation performance.
        weights = []
        relationship = {}
        indices_to_remove = []
        random_guess_perf = agg_utils.get_random_guess_perf(conf)
        for idx, (client_id, local_model) in enumerate(local_models.items()):
            relationship[idx] = client_id
            validated_perfs = validate(
                conf,
                model=local_model,
                criterion=criterion,
                metrics=metrics,
                data_loader=val_data_loader,
            )
            perf = validated_perfs["top1"]
            weights.append(perf)

            # check the perf.
            if perf < random_guess_perf:
                indices_to_remove.append(idx)

        # update client_teacher.
        conf.logger.log(
            f"Indices to be removed for FedAvg: {indices_to_remove}; the original performance is: {weights}."
        )
        for index in indices_to_remove[::-1]:
            flatten_local_models.pop(relationship[index])
        return _fedavg(
            clientid2arch,
            n_selected_clients - len(indices_to_remove),
            flatten_local_models,
            client_models,
        )


def validate(conf, model, data_loader, criterion, metrics):
    val_perf = master_utils.validate(
        conf=conf,
        coordinator=None,
        model=model,
        criterion=criterion,
        metrics=metrics,
        data_loader=data_loader,
        label=None,
        display=False,
    )
    del model
    return val_perf
