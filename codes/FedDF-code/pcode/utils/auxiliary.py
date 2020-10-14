# -*- coding: utf-8 -*-
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch


def deepcopy_model(conf, model):
    # a dirty hack....
    tmp_model = deepcopy(model)
    if conf.track_model_aggregation:
        for tmp_para, para in zip(tmp_model.parameters(), model.parameters()):
            tmp_para.grad = para.grad.clone()
    return tmp_model


def get_model_difference(model1, model2):
    list_of_tensors = []
    for weight1, weight2 in zip(model1.parameters(), model2.parameters()):
        tensor = get_diff_weights(weight1, weight2)
        list_of_tensors.append(tensor)
    return list_to_vec(list_of_tensors).norm().item()


def get_diff_weights(weights1, weights2):
    """ Produce a direction from 'weights1' to 'weights2'."""
    if isinstance(weights1, list) and isinstance(weights2, list):
        return [w2 - w1 for (w1, w2) in zip(weights1, weights2)]
    elif isinstance(weights1, torch.Tensor) and isinstance(weights2, torch.Tensor):
        return weights2 - weights1
    else:
        raise NotImplementedError


def get_diff_states(states1, states2):
    """ Produce a direction from 'states1' to 'states2'."""
    return [v2 - v1 for (k1, v1), (k2, v2) in zip(states1.items(), states2.items())]


def list_to_vec(weights):
    """Concatnate a numpy list of weights of all layers into one torch vector."""
    v = []
    direction = [d * np.float64(1.0) for d in weights]
    for w in direction:
        if isinstance(w, np.ndarray):
            w = torch.tensor(w)
        else:
            w = w.clone().detach()
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def str2time(string, pattern):
    """convert the string to the datetime."""
    return datetime.strptime(string, pattern)


def get_fullname(o):
    """get the full name of the class."""
    return "%s.%s" % (o.__module__, o.__class__.__name__)


def is_float(value):
    try:
        float(value)
        return True
    except:
        return False


class dict2obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)
