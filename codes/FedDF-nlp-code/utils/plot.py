# -*- coding: utf-8 -*-
import numpy as np
from operator import itemgetter
from matplotlib.lines import Line2D
from itertools import groupby

import seaborn as sns


"""plot functions."""


def plot_curve_wrt_time(
    ax,
    records,
    x_wrt_sth,
    y_wrt_sth,
    xlabel,
    ylabel,
    title=None,
    markevery_list=None,
    is_smooth=True,
    smooth_space=100,
    l_subset=0.0,
    r_subset=1.0,
    reorder_record_item=None,
    remove_duplicate=True,
    legend=None,
    legend_loc="lower right",
    legend_ncol=2,
    bbox_to_anchor=[0, 0],
    ylimit_bottom=None,
    ylimit_top=None,
    use_log=False,
    num_cols=3,
):
    """Each info consists of
        ['tr_loss', 'tr_top1', 'tr_time', 'te_top1', 'te_step', 'te_time'].
    """
    # parse a list of records.
    num_records = len(records)
    distinct_conf_set = set()

    for ind, (args, info) in enumerate(records):
        # build legend.
        _legend = build_legend(args, legend)
        if _legend in distinct_conf_set and remove_duplicate:
            continue
        else:
            distinct_conf_set.add(_legend)

        # determine the style of line, color and marker.
        line_style, color_style, mark_style = determine_color_and_lines(
            num_rows=num_records // num_cols, num_cols=num_cols, ind=ind
        )

        if markevery_list is not None:
            mark_every = markevery_list[ind]
        else:
            mark_style = None
            mark_every = None

        # determine if we want to smooth the curve.
        assert x_wrt_sth in info, "x-axis does not exist"
        assert y_wrt_sth in info, "y-axis does not exist"
        assert (
            x_wrt_sth in info
            and y_wrt_sth in info
            and len(info[x_wrt_sth]) == len(info[y_wrt_sth])
        ), ""
        x = info[x_wrt_sth]
        if "time" in x_wrt_sth:
            x = [(time - x[0]).seconds + 1 for time in x]
        y = info[y_wrt_sth]

        # sort x and reorganize y.
        _indices = sorted(range(len(x)), key=lambda k: x[k])
        x, y = [x[i] for i in _indices], [y[i] for i in _indices]

        # smoothing.
        if is_smooth:
            x, y = smoothing_func(x, y, smooth_space)

        # only plot subtset.
        _l_subset, _r_subset = int(len(x) * l_subset), int(len(x) * r_subset)
        _x = x[_l_subset:_r_subset]
        _y = y[_l_subset:_r_subset]

        # use log scale for y
        if use_log:
            _y = np.log(_y)

        # plot
        ax = plot_one_case(
            ax,
            x=_x,
            y=_y,
            label=_legend,
            line_style=line_style,
            color_style=color_style,
            mark_style=mark_style,
            mark_every=mark_every,
            remove_duplicate=remove_duplicate,
        )

    ax.set_ylim(bottom=ylimit_bottom, top=ylimit_top)
    ax = configure_figure(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        has_legend=legend is not None,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        bbox_to_anchor=bbox_to_anchor,
    )
    return ax


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

    # smooth curve
    x_, y_ = [], []

    for end_ind in range(0, len(x)):
        x_.append(x[end_ind])
        y_.append(smoothing(end_ind))
    return x_, y_


def reject_outliers(data, threshold=3):
    return data[abs(data - np.mean(data)) < threshold * np.std(data)]


def groupby_indices(results, grouper):
    """group by indices and select the subset parameters
    """
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
        _legend_content = args[_legend]
        my_legend += [
            "{}={}".format(
                _legend,
                list(_legend_content)[0]
                if "pandas" in str(type(_legend_content))
                else _legend_content,
            )
        ]
    return ", ".join(my_legend)
