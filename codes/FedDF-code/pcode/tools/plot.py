# -*- coding: utf-8 -*-
from operator import itemgetter
import numpy as np

from pcode.tools.show_results import reorder_records
from pcode.tools.plot_utils import (
    determine_color_and_lines,
    plot_one_case,
    smoothing_func,
    configure_figure,
    build_legend,
    groupby_indices,
)


"""plot the curve in terms of time."""


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

    # re-order the records.
    if reorder_record_item is not None:
        records = reorder_records(records, based_on=reorder_record_item)

    count = 0
    for ind, (args, info) in enumerate(records):
        # build legend.
        _legend = build_legend(args, legend)
        if _legend in distinct_conf_set and remove_duplicate:
            continue
        else:
            distinct_conf_set.add(_legend)

        # split the y_wrt_sth if it can be splitted.
        if ";" in y_wrt_sth:
            has_multiple_y = True
            list_of_y_wrt_sth = y_wrt_sth.split(";")
        else:
            has_multiple_y = False
            list_of_y_wrt_sth = [y_wrt_sth]

        for _y_wrt_sth in list_of_y_wrt_sth:
            # determine the style of line, color and marker.
            line_style, color_style, mark_style = determine_color_and_lines(
                num_rows=num_records // num_cols, num_cols=num_cols, ind=count
            )
            if markevery_list is not None:
                mark_every = markevery_list[count]
            else:
                mark_style = None
                mark_every = None

            # update the counter.
            count += 1

            # determine if we want to smooth the curve.
            if "tr_step" in x_wrt_sth or "tr_epoch" in x_wrt_sth:
                info["tr_step"] = list(range(1, 1 + len(info["tr_loss"])))
            if "tr_epoch" == x_wrt_sth:
                x = info["tr_step"]
                x = [
                    1.0 * _x / args["num_batches_train_per_device_per_epoch"]
                    for _x in x
                ]
            else:
                x = info[x_wrt_sth]
                if "time" in x_wrt_sth:
                    x = [(time - x[0]).seconds + 1 for time in x]
            y = info[_y_wrt_sth]

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
                label=_legend if not has_multiple_y else _legend + f", {_y_wrt_sth}",
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
