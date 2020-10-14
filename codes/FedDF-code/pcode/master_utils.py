# -*- coding: utf-8 -*-
import functools

import torch
import torch.nn.functional as F

import pcode.datasets.mixup_data as mixup
import pcode.create_dataset as create_dataset
import pcode.utils.checkpoint as checkpoint
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.utils.logging import display_test_stat, dispaly_best_test_stat
from pcode.utils.mathdict import MathDict


def inference(
    conf, model, criterion, metrics, data_batch, tracker=None, is_training=True
):
    """Inference on the given model and get loss and accuracy."""
    # do the forward pass and get the output.
    output = model(data_batch["input"])

    # evaluate the output and get the loss, performance.
    if conf.use_mixup and is_training:
        loss = mixup.mixup_criterion(
            criterion,
            output,
            data_batch["target_a"],
            data_batch["target_b"],
            data_batch["mixup_lambda"],
        )

        performance_a = metrics.evaluate(loss, output, data_batch["target_a"])
        performance_b = metrics.evaluate(loss, output, data_batch["target_b"])
        performance = [
            data_batch["mixup_lambda"] * _a + (1 - data_batch["mixup_lambda"]) * _b
            for _a, _b in zip(performance_a, performance_b)
        ]
    else:
        loss = criterion(output, data_batch["target"])
        performance = metrics.evaluate(loss, output, data_batch["target"])

    # update tracker.
    if tracker is not None:
        tracker.update_metrics(
            [loss.item()] + performance, n_samples=data_batch["input"].size(0)
        )
    return loss, output


def do_validation(
    conf,
    coordinator,
    model,
    criterion,
    metrics,
    data_loaders,
    performance=None,
    label=None,
):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function, and then evaluate.
    conf.logger.log(f"Master enters the validation phase.")
    if performance is None:
        performance = get_avg_perf_on_dataloaders(
            conf, coordinator, model, criterion, metrics, data_loaders, label
        )

    # remember best performance and display the val info.
    coordinator.update_perf(performance)
    dispaly_best_test_stat(conf, coordinator)

    # save to the checkpoint.
    conf.logger.log(f"Master finished the validation.")
    if not conf.train_fast:
        checkpoint.save_to_checkpoint(
            conf,
            {
                "arch": conf.arch,
                "current_comm_round": conf.graph.comm_round,
                "best_perf": coordinator.best_trackers["top1"].best_perf,
                "state_dict": model.state_dict(),
            },
            coordinator.best_trackers["top1"].is_best,
            dirname=conf.checkpoint_root,
            filename="checkpoint.pth.tar",
            save_all=conf.save_all_models,
        )
        conf.logger.log(f"Master saved to checkpoint.")


def get_avg_perf_on_dataloaders(
    conf, coordinator, model, criterion, metrics, data_loaders, label
):
    print(f"\tGet averaged performance from {len(data_loaders)} data_loaders.")
    performance = []

    for idx, data_loader in enumerate(data_loaders):
        _performance = validate(
            conf,
            coordinator,
            model,
            criterion,
            metrics,
            data_loader,
            label=f"{label}-{idx}" if label is not None else "test_loader",
        )
        performance.append(MathDict(_performance))
    performance = functools.reduce(lambda a, b: a + b, performance) / len(performance)
    return performance


def validate(
    conf,
    coordinator,
    model,
    criterion,
    metrics,
    data_loader,
    label="test_loader",
    display=True,
):
    """A function for model evaluation."""
    if data_loader is None:
        return None

    # switch to evaluation mode.
    model.eval()

    # place the model to the device.
    if conf.graph.on_cuda:
        model = model.cuda()

    # evaluate on test_loader.
    tracker_te = RuntimeTracker(metrics_to_track=metrics.metric_names)

    for _input, _target in data_loader:
        # load data and check performance.
        data_batch = create_dataset.load_data_batch(
            conf, _input, _target, is_training=False
        )

        with torch.no_grad():
            inference(
                conf,
                model,
                criterion,
                metrics,
                data_batch,
                tracker_te,
                is_training=False,
            )

    # place back model to the cpu.
    if conf.graph.on_cuda:
        model = model.cpu()

    # display the test stat.
    perf = tracker_te()
    if label is not None:
        display_test_stat(conf, coordinator, tracker_te, label)
    if display:
        conf.logger.log(f"The validation performance = {perf}.")
    return perf


def ensembled_validate(
    conf,
    coordinator,
    models,
    criterion,
    metrics,
    data_loader,
    label="test_loader",
    ensemble_scheme=None,
):
    """A function for model evaluation."""
    if data_loader is None:
        return None

    # switch to evaluation mode.
    for model in models:
        model.eval()

        # place the model to the device.
        if conf.graph.on_cuda:
            model = model.cuda()

    # evaluate on test_loader.
    tracker_te = RuntimeTracker(metrics_to_track=metrics.metric_names)

    for _input, _target in data_loader:
        # load data and check performance.
        data_batch = create_dataset.load_data_batch(
            conf, _input, _target, is_training=False
        )

        with torch.no_grad():
            # ensemble.
            if (
                ensemble_scheme is None
                or ensemble_scheme == "avg_losses"
                or ensemble_scheme == "avg_logits"
            ):
                outputs = []
                for model in models:
                    outputs.append(model(data_batch["input"]))
                output = sum(outputs) / len(outputs)
            elif ensemble_scheme == "avg_probs":
                outputs = []
                for model in models:
                    outputs.append(F.softmax(model(data_batch["input"])))
                output = sum(outputs) / len(outputs)

            # eval the performance.
            loss = torch.FloatTensor([0])
            performance = metrics.evaluate(loss, output, data_batch["target"])

        # update the tracker.
        tracker_te.update_metrics(
            [loss.item()] + performance, n_samples=data_batch["input"].size(0)
        )

    # place back model to the cpu.
    for model in models:
        if conf.graph.on_cuda:
            model = model.cpu()

    # display the test stat.
    if label is not None:
        display_test_stat(conf, coordinator, tracker_te, label)
    perf = tracker_te()
    conf.logger.log(f"The performance of the ensenmbled model: {perf}.")
    return perf
