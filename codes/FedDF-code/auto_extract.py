# -*- coding: utf-8 -*-
import os
import argparse

import pcode.utils.op_files as op_files
from pcode.tools.show_results import load_raw_info_from_experiments

"""parse and define arguments for different tasks."""


def get_args():
    # feed them to the parser.
    parser = argparse.ArgumentParser(description="Extract results.")

    # add arguments.
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--out_name", type=str, default="summary.pickle")

    # parse aˇˇrgs.
    args = parser.parse_args()

    # an argument safety check.
    check_args(args)
    return args


def check_args(args):
    assert args.in_dir is not None

    # define out path.
    args.out_path = os.path.join(args.in_dir, args.out_name)


"""write the results to path."""


def main(args):
    # save the parsed results to path.
    op_files.write_pickle(load_raw_info_from_experiments(args.in_dir), args.out_path)


if __name__ == "__main__":
    args = get_args()

    main(args)
