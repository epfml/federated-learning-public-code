# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.lines import Line2D
from itertools import groupby

import seaborn as sns

"""operate x and y."""


def smoothing_func(x, y, smooth_length=10):
    def smoothing(end_index):
        # print(end_index)
        if end_index - smooth_length < 0:
            start_index = 0
        else:
            start_index = end_index - smooth_length

        data = y[start_index:end_index]
        if len(data) == 0:
            return y[start_index]
        else:
            return 1.0 * sum(data) / len(data)

    if smooth_length == 0:
        _min_length = min(len(x), len(y))
        return x[:_min_length], y[:_min_length]

    # smooth curve
    x_, y_ = [], []

    for end_ind in range(0, len(x)):
        x_.append(x[end_ind])
        y_.append(smoothing(end_ind))
    return x_, y_


def reject_outliers(data, threshold=3):
    return data[abs(data - np.mean(data)) < threshold * np.std(data)]


def groupby_indices(results, grouper):
    """group by indices and select the subset parameters"""
    out = []
    for key, group in groupby(sorted(results, key=grouper), grouper):
        group_item = list(group)
        out += [(key, group_item)]
    return out


def find_same_num_sync(num_update_steps_and_local_step):
    list_of_num_sync = [
        num_update_steps // local_step
        for num_update_steps, local_step in num_update_steps_and_local_step
    ]
    return min(list_of_num_sync)


def sample_from_records(x, y, local_step, max_same_num_sync):
    # cut the records.
    if max_same_num_sync is not None:
        x = x[: local_step * max_same_num_sync]
        y = y[: local_step * max_same_num_sync]
    return x[::local_step], y[::local_step]


def drop_first_few(x, y, num_drop):
    return x[num_drop:], y[num_drop:]


def rebuild_runtime_record(times):
    times = [(time - times[0]).seconds + 1 for time in times]
    return times


def add_communication_delay(times, local_step, delay_factor):
    """add communication delay to original time."""
    return [
        time + delay_factor * ((ind + 1) // local_step)
        for ind, time in enumerate(times)
    ]


"""plot style related."""


def determine_color_and_lines(num_rows, num_cols, ind):
    line_styles = ["-", "--", "-.", ":"]
    color_styles = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]

    num_line_styles = len(line_styles)
    num_color_styles = len(color_styles)
    total_num_combs = num_line_styles * num_color_styles

    assert total_num_combs > num_rows * num_cols

    if max(num_rows, num_cols) > max(num_line_styles, num_color_styles):
        row = ind // num_line_styles
        col = ind % num_line_styles
        # print('plot {}. case 1, row: {}, col: {}'.format(ind, row, col))
        return line_styles[row], color_styles[col], Line2D.filled_markers[ind]

    denominator = max(num_rows, num_cols)
    row = ind // denominator
    col = ind % denominator
    # print('plot {}. case 2, row: {}, col: {}'.format(ind, row, col))
    return line_styles[row], color_styles[col], Line2D.filled_markers[ind]


def configure_figure(
    ax,
    xlabel,
    ylabel,
    title=None,
    has_legend=True,
    legend_loc="lower right",
    legend_ncol=2,
    bbox_to_anchor=[0, 0],
):
    if has_legend:
        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=legend_ncol,
            shadow=True,
            fancybox=True,
            fontsize=20,
        )

    ax.set_xlabel(xlabel, fontsize=24, labelpad=18)
    ax.set_ylabel(ylabel, fontsize=24, labelpad=18)

    if title is not None:
        ax.set_title(title, fontsize=24)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)
    return ax


def plot_one_case(
    ax,
    label,
    line_style,
    color_style,
    mark_style,
    line_width=2.0,
    mark_every=5000,
    x=None,
    y=None,
    sns_plot=None,
    remove_duplicate=False,
):
    if sns_plot is not None and not remove_duplicate:
        ax = sns.lineplot(
            x="x",
            y="y",
            data=sns_plot,
            label=label,
            linewidth=line_width,
            linestyle=line_style,
            color=color_style,
            marker=mark_style,
            markevery=mark_every,
            markersize=16,
            ax=ax,
        )
    elif sns_plot is not None and remove_duplicate:
        ax = sns.lineplot(
            x="x",
            y="y",
            data=sns_plot,
            label=label,
            linewidth=line_width,
            linestyle=line_style,
            color=color_style,
            marker=mark_style,
            markevery=mark_every,
            markersize=16,
            ax=ax,
            estimator=None,
        )
    else:
        ax.plot(
            x,
            y,
            label=label,
            linewidth=line_width,
            linestyle=line_style,
            color=color_style,
            marker=mark_style,
            markevery=mark_every,
            markersize=16,
        )
    return ax


def build_legend(args, legend):
    legend = legend.split(",")

    my_legend = []
    for _legend in legend:
        _legend_content = args[_legend] if _legend in args else -1
        my_legend += [
            "{}={}".format(
                _legend,
                list(_legend_content)[0]
                if "pandas" in str(type(_legend_content))
                else _legend_content,
            )
        ]
    return ", ".join(my_legend)
