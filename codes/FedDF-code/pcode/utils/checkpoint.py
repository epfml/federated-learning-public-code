# -*- coding: utf-8 -*-
import time
import shutil
import json
from os.path import join

import torch

from pcode.utils.op_paths import build_dirs
from pcode.utils.op_files import is_jsonable


def get_checkpoint_folder_name(conf):
    # get optimizer info.
    optim_info = "{}".format(conf.optimizer)

    # get n_participated
    conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)

    # concat them together.
    return "_l2-{}_lr-{}_n_comm_rounds-{}_local_n_epochs-{}_batchsize-{}_n_clients_{}_n_participated-{}_optim-{}_agg_scheme-{}".format(
        conf.weight_decay,
        conf.lr,
        conf.n_comm_rounds,
        conf.local_n_epochs,
        conf.batch_size,
        conf.n_clients,
        conf.n_participated,
        optim_info,
        conf.fl_aggregate_scheme,
    )


def init_checkpoint(conf, rank=None):
    # init checkpoint_root for the main process.
    conf.checkpoint_root = join(
        conf.checkpoint,
        conf.data,
        conf.arch,
        conf.experiment,
        conf.timestamp + get_checkpoint_folder_name(conf),
    )
    if conf.save_some_models is not None:
        conf.save_some_models = conf.save_some_models.split(",")

    if rank is None:
        # if the directory does not exists, create them.
        build_dirs(conf.checkpoint_root)
    else:
        conf.checkpoint_dir = join(conf.checkpoint_root, rank)
        build_dirs(conf.checkpoint_dir)


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


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


def save_to_checkpoint(conf, state, is_best, dirname, filename, save_all=False):
    # save full state.
    checkpoint_path = _save_to_checkpoint(state, dirname, filename)
    best_model_path = join(dirname, "model_best.pth.tar")
    if is_best:
        shutil.copyfile(checkpoint_path, best_model_path)
    if save_all:
        shutil.copyfile(
            checkpoint_path,
            join(
                dirname, "checkpoint_c_round_%s.pth.tar" % state["current_comm_round"]
            ),
        )
    elif conf.save_some_models is not None:
        if str(state["current_comm_round"]) in conf.save_some_models:
            shutil.copyfile(
                checkpoint_path,
                join(
                    dirname,
                    "checkpoint_c_round_%s.pth.tar" % state["current_comm_round"],
                ),
            )
