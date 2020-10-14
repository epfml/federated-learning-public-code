# -*- coding: utf-8 -*-
from __future__ import division
import os
import json
import functools
import numbers
import pandas as pd


from pcode.utils.op_paths import list_files
from pcode.utils.op_files import load_pickle, read_json
from pcode.utils.auxiliary import str2time


"""load data from pickled file."""


def get_pickle_info(root_data_path, experiments):
    file_paths = []
    for experiment in experiments:
        file_paths += [
            os.path.join(root_data_path, experiment, file)
            for file in os.listdir(os.path.join(root_data_path, experiment))
            if "pickle" in file
        ]

    results = dict((path, load_pickle(path)) for path in file_paths)
    info = functools.reduce(lambda a, b: a + b, list(results.values()))
    return info


"""load the raw results"""


def load_raw_info_from_experiments(root_path):
    """load experiments.
    root_path: a directory with a list of different trials.
    """
    exp_folder_paths = [
        folder_path
        for folder_path in list_files(root_path)
        if "pickle" not in folder_path
    ]

    info = []
    for folder_path in exp_folder_paths:
        try:
            element_of_info = _get_info_from_the_folder(folder_path)
            info.append(element_of_info)
        except Exception as e:
            print("error: {}".format(e))
    return info


def _get_info_from_the_folder(folder_path):
    print("process the folder: {}".format(folder_path))
    arguments_path = os.path.join(folder_path, "arguments.json")

    # collect runtime json info for one rank.
    sub_folder_paths = sorted(
        [
            sub_folder_path
            for sub_folder_path in list_files(folder_path)
            if ".tar" not in sub_folder_path and "pickle" not in sub_folder_path
        ]
    )

    # return the information.
    return (
        folder_path,
        {
            "arguments": read_json(arguments_path),
            "single_records": _parse_runtime_infos(sub_folder_paths[0]),
        },
    )


def _parse_runtime_infos(file_folder):
    existing_json_files = [file for file in os.listdir(file_folder) if "json" in file]

    if "log.json" in existing_json_files:
        # old logging fashion.
        return _parse_runtime_info(os.path.join(file_folder, "log.json"))
    else:
        # new logging fashion.
        lines = []
        for idx in range(1, 1 + len(existing_json_files)):
            _lines = _parse_runtime_info(
                os.path.join(file_folder, "log-{}.json".format(idx))
            )
            lines.append(_lines)

        return functools.reduce(
            lambda a, b: [a[idx] + b[idx] for idx in range(len(a))], lines
        )


def _parse_runtime_info(json_file_path):
    with open(json_file_path) as json_file:
        lines = json.load(json_file)

        # distinguish lines to different types.
        tr_lines, aggregated_test_lines, fedavg_test_lines, ensemble_test_lines = (
            [],
            [],
            [],
            [],
        )

        for line in lines:
            if line["measurement"] != "runtime":
                continue

            try:
                _time = str2time(line["time"], "%Y-%m-%d %H:%M:%S")
            except RuntimeError:
                _time = None
            line["time"] = _time

            if line["split"] == "train":
                tr_lines.append(line)
            elif line["split"] == "test":
                if line["type"] == "aggregated_test_loader-0":
                    aggregated_test_lines.append(line)
                elif line["type"] == "fedag_test_loader-0":
                    fedavg_test_lines.append(line)
                elif line["type"] == "ensemble_test_loader":
                    ensemble_test_lines.append(line)
    return tr_lines, fedavg_test_lines, aggregated_test_lines, ensemble_test_lines


"""extract the results based on the condition."""


def _is_same(items):
    return len(set(items)) == 1


def is_meet_conditions(args, conditions, threshold=1e-8):
    if conditions is None:
        return True

    # get condition values and have a safety check.
    condition_names = list(conditions.keys())
    condition_values = list(conditions.values())
    assert _is_same([len(values) for values in condition_values]) is True

    # re-build conditions.
    num_condition = len(condition_values)
    num_condition_value = len(condition_values[0])
    condition_values = [
        [condition_values[ind_cond][ind_value] for ind_cond in range(num_condition)]
        for ind_value in range(num_condition_value)
    ]

    # check re-built condition.
    g_flag = False
    try:
        for cond_values in condition_values:
            l_flag = True
            for ind, cond_value in enumerate(cond_values):
                _cond = cond_value == (
                    args[condition_names[ind]] if condition_names[ind] in args else None
                )

                if isinstance(cond_value, numbers.Number):
                    _cond = (
                        _cond
                        or abs(cond_value - args[condition_names[ind]]) <= threshold
                    )

                l_flag = l_flag and _cond
            g_flag = g_flag or l_flag
        return g_flag
    except:
        return False


def reorganize_records(records):
    def _parse(lines):
        time, step, loss, top1, top5 = [], [], [], [], []

        for line in lines:
            time.append(line["time"])
            step.append(line["comm_round"])
            loss.append(line["loss"])
            top1.append(line["top1"] if "top1" in line else 0)
            top5.append(line["top5"] if "top5" in line else 0)
        return time, step, loss, top1, top5

    # deal with single records.
    tr_records, te_fedavg_records, te_aggregated_records, te_ensemble_records = records[
        "single_records"
    ]
    if len(te_fedavg_records) == 0:
        te_time, te_epoch, te_loss, te_top1, te_top5 = _parse(te_aggregated_records)
        return {
            "te_time": te_time,
            "te_step": te_epoch,
            "te_loss": te_loss,
            "te_top1": te_top1,
            "te_top5": te_top5,
        }
    else:
        (
            te_fedavg_time,
            te_fedavg_epoch,
            te_fedavg_loss,
            te_fedavg_top1,
            te_fedavg_top5,
        ) = _parse(te_fedavg_records)
        _, _, te_aggreg_loss, te_aggreg_top1, te_aggreg_top5 = _parse(
            te_aggregated_records
        )

        if len(te_ensemble_records) > 0:
            _, _, te_ensemble_loss, te_ensemble_top1, te_ensemble_top5 = _parse(
                te_ensemble_records
            )
        else:
            te_ensemble_loss, te_ensemble_top1, te_ensemble_top5 = [], [], []
        return {
            "te_time": te_fedavg_time,
            "te_step": te_fedavg_epoch,
            "te_avg_loss": te_fedavg_loss,
            "te_avg_top1": te_fedavg_top1,
            "te_avg_top5": te_fedavg_top5,
            "te_loss": te_aggreg_loss,
            "te_top1": te_aggreg_top1,
            "te_top5": te_aggreg_top5,
            "te_ensemble_loss": te_ensemble_loss,
            "te_ensemble_top1": te_ensemble_top1,
            "te_ensemble_top5": te_ensemble_top5,
        }


def extract_list_of_records(list_of_records, conditions):
    # load and filter data.
    records = []

    for path, raw_records in list_of_records:
        # check conditions.
        if not is_meet_conditions(raw_records["arguments"], conditions):
            continue

        # get parsed records
        records += [(raw_records["arguments"], reorganize_records(raw_records))]

    print("we have {}/{} records.".format(len(records), len(list_of_records)))
    return records


"""summary the results."""


def _summarize_info(
    record, arg_names, be_groupby, larger_is_better, avg_count=100, get_final=False
):
    args, info = record

    if "te" in be_groupby or "tr" in be_groupby:
        if not get_final:
            sorted_info = sorted(info[be_groupby], reverse=False)
            if "te" in be_groupby:
                performance = sorted_info[-1] if larger_is_better else sorted_info[0]
            elif "tr" in be_groupby:
                performance = (
                    sum(sorted_info[-avg_count:]) / avg_count
                    if larger_is_better
                    else sum(sorted_info[avg_count]) / avg_count
                )
        else:
            performance = info[be_groupby][-1]
    else:
        performance = args[be_groupby] if be_groupby in args else -1
    return [args[arg_name] if arg_name in args else None for arg_name in arg_names] + [
        performance
    ]


def reorder_records(records, based_on):
    # records is in the form of <args, info>
    conditions = based_on.split(",")
    list_of_args = [
        (ind, [args[condition] for condition in conditions])
        for ind, (args, info) in enumerate(records)
    ]
    sorted_list_of_args = sorted(list_of_args, key=lambda x: x[1:])
    return [records[ind] for ind, args in sorted_list_of_args]


def summarize_info(
    records, arg_names, be_groupby="te_top1", larger_is_better=True, get_final=False
):
    # note that 'get_final' has higher priority than 'larger_is_better'.
    # define header.
    headers = arg_names + [be_groupby]
    # reorder records
    records = reorder_records(records, based_on="n_clients")
    # extract test records
    test_records = [
        _summarize_info(
            record, arg_names, be_groupby, larger_is_better, get_final=get_final
        )
        for record in records
    ]
    # aggregate test records
    aggregated_records = pd.DataFrame(test_records, columns=headers)
    # average test records
    averaged_records = (
        aggregated_records.fillna(-1)
        .groupby(headers[:-1], as_index=False)
        .agg({be_groupby: ["mean", "std", "max", "min", "count"]})
        .sort_values((be_groupby, "mean"), ascending=not larger_is_better)
    )
    return aggregated_records, averaged_records
