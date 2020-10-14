# -*- coding: utf-8 -*-
import json
from copy import deepcopy

import torch

from pcode.utils.module_state import ModuleState
import pcode.aggregation.utils as agg_utils


def _get_init_agg_weights(conf, model, num_models):
    def _get_agg_weight_template():
        return torch.tensor(
            [0.5] * num_models,
            requires_grad=True,
            device="cuda" if conf.graph.on_cuda else "cpu",
        )

    # build a list of agg_weight params
    is_layerwise = False
    if (
        "layerwise" not in conf.fl_aggregate
        or conf.fl_aggregate["layerwise"] is False
        or conf.fl_aggregate["layerwise"] == "False"
    ):
        agg_weights = _get_agg_weight_template()
    elif (
        conf.fl_aggregate["layerwise"] is True
        or conf.fl_aggregate["layerwise"] == "True"
    ):
        is_layerwise = True
        agg_weights = dict()
        module_state = ModuleState(deepcopy(model.state_dict()))

        for name, _module in model.named_modules():
            for key in _module._parameters:
                param_name = f"{name}.{key}"
                if param_name in module_state.keys:
                    agg_weights[param_name] = _get_agg_weight_template()
    else:
        raise NotImplementedError("not supported scheme for learning to aggregate.")

    optimizer = torch.optim.Adam(
        [agg_weights] if not is_layerwise else list(agg_weights.values()),
        lr=conf.fl_aggregate["optim_lr"],
        betas=(conf.adam_beta_1, conf.adam_beta_2),
        eps=conf.adam_eps,
    )
    return agg_weights, optimizer, is_layerwise


def learning2aggregate(
    conf, fedavg_model, client_models, flatten_local_models, criterion, data_loader
):
    # init the local models.
    num_models, local_models = agg_utils.recover_models(
        conf, client_models, flatten_local_models
    )

    # init the agg_weights
    fedavg_model = fedavg_model.cuda() if conf.graph.on_cuda else fedavg_model
    agg_weights, optimizer, is_layerwise = _get_init_agg_weights(
        conf, fedavg_model, num_models
    )

    # learning the aggregation weights.
    for _ in range(int(conf.fl_aggregate["epochs"])):
        for _ind, (_input, _target) in enumerate(data_loader):
            # place model and data.
            if conf.graph.on_cuda:
                _input, _target = _input.cuda(), _target.cuda()

            # get mixed model.
            mixed_model = get_mixed_model(
                conf=conf,
                model=fedavg_model,
                local_models=local_models,
                agg_weights=agg_weights,
                is_layerwise=is_layerwise,
            )

            # inference and update alpha
            mixed_model.train()
            optimizer.zero_grad()
            loss = criterion(mixed_model(_input), _target)
            loss.backward()
            optimizer.step()

    # extract the final agg_weights.
    weighted_avg_model = get_mixed_model(
        conf=conf,
        model=fedavg_model,
        local_models=local_models,
        agg_weights=agg_weights,
        is_layerwise=is_layerwise,
        display_agg_weights=True,
    )
    del local_models
    return weighted_avg_model.cpu()


def get_mixed_model(
    conf, model, local_models, agg_weights, is_layerwise, display_agg_weights=False
):
    _model = deepcopy(model)
    local_states = [
        ModuleState(deepcopy(local_model.state_dict()))
        for _, local_model in local_models.items()
    ]

    # normalize the aggregation weights and then return an aggregated model.
    agg_weights_info = {}
    if not is_layerwise:
        # get agg_weights.
        agg_weights = torch.nn.functional.softmax(agg_weights, dim=0)
        if display_agg_weights:
            agg_weights_info["globalwise"] = agg_weights.detach().cpu().numpy().tolist()

        # aggregate local models by weights.
        model_state = local_states[0] * agg_weights[0]
        for idx in range(1, len(local_states)):
            model_state += local_states[idx] * agg_weights[idx]
        model_state.copy_to_module(_model)
    else:
        model_state = local_states[0] * 0.0
        for key, _agg_weights in agg_weights.items():
            _agg_weights = torch.nn.functional.softmax(_agg_weights, dim=0)
            if display_agg_weights:
                agg_weights_info[key] = _agg_weights.detach().cpu().numpy().tolist()

            # aggregate local models by weights.
            for idx in range(0, len(local_states)):
                model_state += local_states[idx].mul_by_key(
                    factor=_agg_weights[idx], by_key=key
                )
        model_state.copy_to_module(_model)

    if display_agg_weights:
        conf.logger.log(f"The aggregation weights={json.dumps(agg_weights_info)}")
    return _model
