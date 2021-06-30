# -*- coding: utf-8 -*-
import os
import copy
import torch
import numpy as np

from parameters import get_args
import configs.task_configs as task_configs
import utils.checkpoint as checkpoint
import utils.logging as logging
import utils.param_parser as param_parser
from pcode.master import Master


def main(conf):
    # general init.
    init_config(conf)

    # start federated learning.
    process = Master(conf)
    process.run()


def init_config(conf):
    # configure the training device.
    assert conf.world is not None, "Please specify the gpu ids."
    conf.world = (
        [int(x) for x in conf.world.split(",")]
        if "," in conf.world
        else [int(conf.world)]
    )
    conf.n_sub_process = len(conf.world)

    # init the model arch info.
    conf.model_info = param_parser.dict_parser(conf.model_info)
    conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)

    # parse the fl_aggregate scheme.
    conf._fl_aggregate = conf.fl_aggregate
    conf.fl_aggregate = (
        param_parser.dict_parser(conf.fl_aggregate)
        if conf.fl_aggregate is not None
        else conf.fl_aggregate
    )
    [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]

    # re-configure batch_size if sub_process > 1.
    if conf.n_sub_process > 1:
        conf.batch_size = conf.batch_size * conf.n_sub_process

    # configure cuda related.
    assert torch.cuda.is_available()
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)
    torch.cuda.manual_seed(conf.manual_seed)
    torch.cuda.set_device(conf.world[0])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True if conf.train_fast else False

    # define checkpoint for logging.
    checkpoint.init_checkpoint(conf)

    # display the arguments' info.
    logging.display_args(conf)

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_root)


if __name__ == "__main__":
    conf = get_args()

    main(conf)
