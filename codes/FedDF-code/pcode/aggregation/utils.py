# -*- coding: utf-8 -*-
import copy
from copy import deepcopy
import collections

import numpy as np


def recover_models(conf, client_models, flatten_local_models, use_cuda=True):
    # init the local models.
    num_models = len(flatten_local_models)
    local_models = {}

    for client_idx, flatten_local_model in flatten_local_models.items():
        arch = conf.clientid2arch[client_idx]
        _model = deepcopy(client_models[arch])
        _model_state_dict = _model.state_dict()
        flatten_local_model.unpack(_model_state_dict.values())
        _model.load_state_dict(_model_state_dict)
        local_models[client_idx] = _model.cuda() if conf.graph.on_cuda else _model

        # turn off the grad for local models.
        for param in local_models[client_idx].parameters():
            param.requires_grad = False
    return num_models, local_models


def modify_model_trainable_status(conf, model, trainable):
    _model = deepcopy(model)
    if conf.graph.on_cuda:
        _model = _model.cuda()

    for _, _param in _model.named_parameters():
        _param.requires_grad = trainable
    return _model


def check_trainable(conf, model):
    _model = deepcopy(model)
    if conf.graph.on_cuda:
        _model = _model.cuda()

    trainable_params = []
    is_complete = True
    for _name, _param in _model.named_parameters():
        if _param.requires_grad is True:
            trainable_params.append(_name)
        else:
            is_complete = False
    print(f"\tthe trainable model parameters is complete={is_complete}")
    return _model


def include_previous_models(conf, local_models):
    if hasattr(conf, "previous_local_models"):
        local_models.update(collections.ChainMap(*conf.previous_local_models.values()))
    return local_models


def update_previous_models(conf, client_models):
    if not hasattr(conf, "previous_local_models"):
        conf.previous_local_models = collections.defaultdict(dict)

    for arch, model in client_models.items():
        conf.previous_local_models[arch][-conf.graph.comm_round] = model.cpu()
        # we use reverse order here.
        conf.previous_local_models[arch] = dict(
            list(sorted(conf.previous_local_models[arch].items(), key=lambda x: -x[0]))[
                -int(conf.fl_aggregate["include_previous_models"]) :
            ]
        )


def filter_models_by_weights(normalized_weights, detect_fn_name=None):
    remained_indices_weights = detect_outlier_and_remain(
        normalized_weights, fn_name=detect_fn_name
    )
    remained_weights = [weight for index, weight in remained_indices_weights]
    whole_indices = list(range(len(normalized_weights)))
    indices_to_remove = sorted(
        list(set(whole_indices) - set(index for index, _ in remained_indices_weights))
    )
    return indices_to_remove, remained_weights


def detect_outlier_and_remain(values, fn_name=None):
    if fn_name is None:
        return detect_outlier_and_remain_v1(values)
    else:
        return eval(fn_name)(values)


def detect_outlier_and_remain_v1(values):
    _values = copy.deepcopy(values)
    _values.remove(max(values))

    # calculate summary statistics
    data_mean, data_std = np.mean(_values), np.std(_values)
    # identify outliers
    cut_off = data_std * 1.5
    lower, upper = data_mean - cut_off, data_mean + cut_off
    return [(idx, value) for idx, value in enumerate(values) if lower <= value]


def detect_outlier_and_remain_v2(values):
    _values = copy.deepcopy(values)
    _values.remove(max(values))
    q25, q75 = np.quantile(_values, 0.25), np.quantile(_values, 0.75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    return [(idx, value) for idx, value in enumerate(values) if lower <= value]


SCALING_FACTOR = 1.2


def get_random_guess_perf(conf):
    if conf.data == "cifar10":
        return 1 / 10 * 100 * SCALING_FACTOR
    elif conf.data == "cifar100":
        return 1 / 100 * 100 * SCALING_FACTOR
    elif "imagenet" in conf.data:
        return 1 / 1000 * SCALING_FACTOR
    else:
        raise NotImplementedError
