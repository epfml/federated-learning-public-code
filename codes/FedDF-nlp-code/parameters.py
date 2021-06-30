# -*- coding: utf-8 -*-
from os.path import join
import argparse
import time

import utils.checkpoint as checkpoint


def get_args():
    ROOT_DIRECTORY = "./"
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, "data/")
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, "checkpoint")

    # feed them to the parser.
    parser = argparse.ArgumentParser()

    # task.
    parser.add_argument("--ptl", type=str, default="bert")
    parser.add_argument("--model_info", type=str, default="model=distilbert,ptl=")
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--max_seq_len", type=int, default=128)

    # training and learning scheme
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pin_memory", default=True, type=str2bool)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=5e-3)

    parser.add_argument("--adam_beta_1", default=0.9, type=float)
    parser.add_argument("--adam_beta_2", default=0.999, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)

    parser.add_argument("--momentum_factor", default=0.9, type=float)
    parser.add_argument("--use_nesterov", default=False, type=str2bool)

    parser.add_argument(
        "--weight_decay", default=0, type=float, help="weight decay (default: 1e-4)"
    )
    parser.add_argument("--drop_rate", default=0.0, type=float)

    # federated learning
    parser.add_argument("--n_comm_rounds", type=int, default=90)
    parser.add_argument(
        "--target_perf", type=float, default=None, help="it is between [0, 100]."
    )
    parser.add_argument("--early_stopping_rounds", type=int, default=0)
    parser.add_argument("--local_n_epochs", type=int, default=1)
    parser.add_argument("--min_local_epochs", type=int, default=None)
    parser.add_argument("--reshuffle_per_epoch", default=False, type=str2bool)
    parser.add_argument(
        "--n_clients",
        default=1,
        type=int,
        help="# of the clients for federated learning.",
    )
    parser.add_argument(
        "--partition_data",
        default=None,
        type=str,
        help="decide if each worker will access to all data.",
    )
    parser.add_argument(
        "--participation_ratio",
        default=0.1,
        type=float,
        help="number of participated ratio per communication rounds",
    )
    parser.add_argument(
        "--train_data_ratio", type=float, default=0, help="after the train/val split."
    )
    parser.add_argument("--fl_aggregate", default=None, type=str)
    parser.add_argument("--non_iid_alpha", default=0, type=float)

    # miscs
    parser.add_argument("--data_path", default=RAW_DATA_DIRECTORY, type=str)
    parser.add_argument("--checkpoint", default=TRAINING_DIRECTORY, type=str)
    parser.add_argument("--manual_seed", type=int, default=7, help="manual seed")
    parser.add_argument("--eval_every_batch", default=60, type=int)
    parser.add_argument("--summary_freq", default=100, type=int)
    parser.add_argument("--time_stamp", default=None, type=str)
    parser.add_argument("--train_fast", default=True, type=str2bool)
    parser.add_argument("--track_time", default=True, type=str2bool)
    parser.add_argument("--early_stop", default=None, type=float)
    parser.add_argument("--early_stopping_epochs", default=100, type=int)

    """meta info."""
    parser.add_argument("--experiment", type=str, default="debug")
    parser.add_argument("--job_id", type=str, default="/tmp/jobrun_logs")
    parser.add_argument("--script_path", default="exp/", type=str)
    parser.add_argument("--script_class_name", default=None, type=str)
    parser.add_argument("--num_jobs_per_node", default=1, type=int)

    # device
    parser.add_argument(
        "--python_path", type=str, default="$HOME/conda/envs/pytorch-py3.6/bin/python"
    )
    parser.add_argument(
        "-j",
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument("--world", default="0", type=str)

    # parse conf.
    conf = parser.parse_args()
    return conf


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    args = get_args()
