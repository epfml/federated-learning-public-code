# -*- coding: utf-8 -*-
import os
import time
import torch

import shutil
import json
from os.path import join


def init_checkpoint(conf):
    # init checkpoint dir.
    invalid = True
    while invalid:
        time_id = str(int(time.time()))
        conf.time_stamp_ = f"{time_id}_task-{conf.task}_scheme-{conf.fl_aggregate['scheme']}_s-{conf.manual_seed}"
        conf.checkpoint_root = os.path.join(
            conf.checkpoint,
            conf.task,
            conf.experiment if conf.experiment is not None else "",
            conf.time_stamp_,
        )

        # if the directory does not exists, create them and break the loop.
        if not os.path.exists(conf.checkpoint_root) and build_dirs(
            conf.checkpoint_root
        ):
            invalid = False

    conf.pretrained_weight_path = os.path.join(conf.data_path, "pretrained_weights")
    print(conf.checkpoint_root)
    assert len(os.path.abspath(conf.checkpoint_root)) < 255
    return conf.checkpoint_root


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = os.path.join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def build_dirs(path):
    try:
        os.makedirs(path)
        return True
    except Exception as e:
        print(" encounter error: {}".format(e))
        return False


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def save_arguments(conf):
    # save the configure file to the checkpoint.
    # write_pickle(conf, path=join(conf.checkpoint_root, "arguments.pickle"))
    with open(join(conf.checkpoint_root, "arguments.json"), "w") as fp:
        json.dump(
            dict(
                [
                    (k, v)
                    for k, v in conf.__dict__.items()
                    if is_jsonable(v) and type(v) is not torch.Tensor
                ]
            ),
            fp,
            indent=" ",
        )
