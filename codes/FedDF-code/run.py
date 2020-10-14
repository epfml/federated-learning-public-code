# -*- coding: utf-8 -*-
import re
import os
import time

import pcode.utils.op_files as op_files
import parameters as para


def read_hostfile(file_path):
    def _parse(line):
        matched_line = re.findall(r"^(.*?) slots=(.*?)$", line, re.DOTALL)
        matched_line = [x.strip() for x in matched_line[0]]
        return matched_line

    # read file
    lines = op_files.read_txt(file_path)

    # use regex to parse the file.
    ip2slots = dict(_parse(line) for line in lines)
    return ip2slots


def map_slot(ip2slots):
    ip_slot = []
    for ip, slots in ip2slots.items():
        for _ in range(int(slots)):
            ip_slot += [ip]
    return ip_slot


def run_cmd(conf, cmd):
    # run the cmd.
    print("\nRun the following cmd:\n" + cmd)
    os.system(cmd)


def build_mpi_script(conf, replacement=None):
    # get the n_participated clients per communication round.
    conf.n_participated = int(conf.n_clients * conf.participation_ratio + 0.5)
    conf.timestamp = str(int(time.time()))
    assert conf.n_participated > 0

    # get prefix_cmd.
    if conf.n_participated >= 1:
        prefix_cmd = f"mpirun -n {conf.n_participated + 1} --hostfile {conf.hostfile} --mca orte_base_help_aggregate 0 --mca btl_tcp_if_exclude docker0,lo --prefix {conf.mpi_path} "
        prefix_cmd += (
            f" -x {conf.mpi_env}"
            if conf.mpi_env is not None and len(conf.mpi_env) > 0
            else ""
        )
    else:
        prefix_cmd = ""

    # build complete script.
    cmd = " {} main.py ".format(conf.python_path)
    for k, v in conf.__dict__.items():
        if replacement is not None and k in replacement:
            cmd += " --{} {} ".format(k, replacement[k])
        elif v is not None:
            cmd += " --{} {} ".format(k, v)
    return prefix_cmd + cmd


def main_mpi(conf, ip2slot):
    cmd = build_mpi_script(conf)

    # run cmd.
    run_cmd(conf, cmd)


if __name__ == "__main__":
    # parse the arguments.
    conf = para.get_args()

    # get ip and the corresponding # of slots.
    ip2slots = read_hostfile(conf.hostfile)
    ip2slot = map_slot(ip2slots)

    # run the main script.
    main_mpi(conf, ip2slot)
