# -*- coding: utf-8 -*-
import functools
import copy
import torch

import pcode.create_dataset as create_dataset
import pcode.aggregation.utils as agg_utils
from pcode.utils.misc import to_one_hot
from pcode.utils.stat_tracker import RuntimeTracker


def aggregate(
    conf,
    fedavg_model,
    client_models,
    criterion,
    metrics,
    data_info,
    flatten_local_models,
    fa_val_perf,
):
    if fa_val_perf["top1"] > conf.fl_aggregate["top1_starting_threshold"]:
        # recover the models.
        _, local_models = agg_utils.recover_models(
            conf, client_models, flatten_local_models
        )

        # create the virtual labels.
        dataset, labels, eps = create_virtual_labels(
            conf, fedavg_model, local_models, data_info
        )
        conf.logger.log(f"the used label smoothing={eps}")

        # train the model on the server with the created virtual model
        data_loaders = rebuild_dataset(conf, data_info, dataset, labels)
        fedavg_model = training(conf, fedavg_model, criterion, data_loaders, eps)

        # free the memory.
        del local_models
    else:
        conf.logger.log(f"skip and directly return the model.")
    return fedavg_model


def create_virtual_labels(conf, model, local_models, data_info):
    soft_label_scheme = conf.fl_aggregate["soft_label_scheme"]
    if soft_label_scheme == "avg":
        # # average over all labels.
        # _soft_labels = functools.reduce(
        #     lambda a, b: a + b, list(dict_of_output.values())
        # ) / dict_of_output[idx].size(1)
        # soft_labels = _soft_labels / torch.sum(_soft_labels, dim=1).unsqueeze(dim=1)
        raise NotImplementedError(
            "the implementation of the current scheme is incorrect."
        )
    elif soft_label_scheme == "majority_vote":
        # get soft_labels.
        dataset, dict_of_labels = extract_labels_from_local_models(
            conf, local_models, data_info
        )
        dict_of_onehot_labels = [
            to_one_hot(v, n_dims=model.num_classes) for v in dict_of_labels.values()
        ]
        summed_onehot_labels = functools.reduce(
            lambda a, b: a + b, dict_of_onehot_labels
        )
        labels = torch.max(summed_onehot_labels, dim=1).indices.cpu()

        # estimate eps for label smoothing.
        eps = estimate_eps(conf, model, dict_of_labels, labels, summed_onehot_labels)
    elif soft_label_scheme == "runtime_avg":
        # use the averaged model to estimate the current labels.
        dataset, _current_estimated_labels = extract_labels_from_local_models(
            conf, {0: model}, data_info
        )
        current_estimated_labels = _current_estimated_labels[0]

        # construct the virtual labels.
        runtime_summed_labels = (
            conf.runtime_labels if hasattr(conf, "runtime_labels") else 0
        )
        runtime_summed_labels = (
            runtime_summed_labels * conf.fl_aggregate["running_avg"]
            + to_one_hot(current_estimated_labels, n_dims=model.num_classes).cpu()
        )
        conf.runtime_labels = runtime_summed_labels
        labels = torch.max(conf.runtime_labels, dim=1).indices.cpu()

        # estimate eps for label smoothing.
        eps = conf.fl_aggregate["max_eps"] if "max_eps" in conf.fl_aggregate else 0.1
    else:
        raise NotImplementedError(
            f"this soft_label_scheme={soft_label_scheme} has not been supported yet."
        )
    print(labels[:20])
    return dataset, labels, eps


def extract_labels_from_local_models(conf, local_models, data_info):
    # init the basic data-loader.
    dataset = data_info["sampler"].use_indices()
    basic_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
    )

    # extract the predictions from local models.
    dict_of_labels = {}
    for idx, _model in local_models.items():
        _model.eval()
        _list_of_labels = []
        if conf.graph.on_cuda:
            _model = _model.cuda()

        for _input, _target in basic_data_loader:
            _data_batch = create_dataset.load_data_batch(
                conf, _input, _target, is_training=False
            )
            _output = _model(_data_batch["input"])
            # extract the exact label for the current prediction.
            _list_of_labels.append(
                torch.max(torch.softmax(_output, dim=1), dim=1).indices
            )

        # concatenate list_of_output
        dict_of_labels[idx] = torch.cat(_list_of_labels)

        # free the memory.
        if conf.graph.on_cuda:
            _model = _model.cpu()
    return dataset, dict_of_labels


def rebuild_dataset(conf, data_info, dataset, labels):
    # rebuild dataset.
    prediction_correctness = dataset.update_replaced_targets(labels.tolist())
    conf.logger.log(
        f"the correctness of the label predictions for the onserver data={prediction_correctness*100:.3f}%."
    )

    # init to construct the data_loader for the on-server training/validation.
    assert (
        "val_percentage" in conf.fl_aggregate
        and 0 <= conf.fl_aggregate["val_percentage"] <= 0.5
    )
    sampled_indices = data_info["sampler"].sampled_indices
    num_sampled_indices = len(sampled_indices)
    num_indices_to_validate = int(
        num_sampled_indices * conf.fl_aggregate["val_percentage"]
    )
    will_validate = conf.fl_aggregate["val_percentage"] > 0

    # construct the data_loader for the on-server training/validation.
    tr_dataset = copy.deepcopy(dataset)
    tr_dataset.indices = dataset.indices[num_indices_to_validate:]
    tr_dataset.replaced_targets = dataset.replaced_targets[num_indices_to_validate:]
    tr_data_loader = torch.utils.data.DataLoader(
        tr_dataset,
        batch_size=conf.batch_size
        if "batch_size" not in conf.fl_aggregate
        else conf.fl_aggregate["batch_size"],
        shuffle=True,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
    )

    if will_validate:
        val_dataset = copy.deepcopy(dataset)
        val_dataset.indices = dataset.indices[:num_indices_to_validate]
        val_dataset.replaced_targets = dataset.replaced_targets[
            :num_indices_to_validate
        ]
        val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=conf.batch_size
            if "batch_size" not in conf.fl_aggregate
            else conf.fl_aggregate["batch_size"],
            shuffle=True,
            num_workers=conf.num_workers,
            pin_memory=conf.pin_memory,
            drop_last=False,
        )
        conf.logger.log(
            f"the number of on-server training mini-batches={len(tr_data_loader)}; the number of on-server validation mini-batches={len(val_data_loader)} (the mini-batch size={conf.batch_size})."
        )
    else:
        val_data_loader = None
        conf.logger.log(
            f"the number of on-server training mini-batches={len(tr_data_loader)} (the mini-batch size={conf.batch_size})."
        )
    return {"tr_data_loader": tr_data_loader, "val_data_loader": val_data_loader}


def estimate_eps(conf, model, dict_of_labels, labels, summed_onehot_labels):
    # evaluate the kl divergence loss.
    normalized_labels = summed_onehot_labels / torch.sum(
        summed_onehot_labels, dim=1, keepdim=True
    )
    onehot_labels = to_one_hot(labels, n_dims=model.num_classes)
    criterion = torch.nn.KLDivLoss(
        reduction="mean"
    )  # 'none' will return an ouput (with the same shape as the input).

    # As with NLLLoss, the input given is expected to contain log-probabilities and is not restricted to a 2D Tensor.
    # The targets are given as probabilities (i.e. without taking the logarithm).
    # https://pytorch.org/docs/stable/nn.html?highlight=kldivloss#torch.nn.KLDivLoss.
    kl = criterion(normalized_labels.log(), onehot_labels)
    kl_max = criterion(
        torch.Tensor([1.0 / model.num_classes] * model.num_classes).log(),
        onehot_labels[0],
    )

    # rescle the eps.
    max_eps = conf.fl_aggregate["max_eps"] if "max_eps" in conf.fl_aggregate else 0.5
    return kl / kl_max * (max_eps - 0.1) + 0.1


def training(conf, model, criterion, data_loaders, eps):
    # place the model on gpu
    if conf.graph.on_cuda:
        model = model.cuda()

    # then train the averaged model on the created virtual model.
    optimizer = create_optimizer(conf, model)

    # init the training setup.
    epoch_count = 0
    final_model = copy.deepcopy(model)

    # init the recording status.
    if data_loaders["val_data_loader"] is not None:
        tracker_val = RuntimeTracker(metrics_to_track=[])
        for _ind, (_input, _target) in enumerate(data_loaders["val_data_loader"]):
            # place model and data.
            if conf.graph.on_cuda:
                _input, _target = _input.cuda(), _target.cuda()

            # inference and evaluate.
            model.eval()
            loss = criterion(model(_input), _target)
            tracker_val.update_metrics([loss.item()], n_samples=_input.size(0))
            tracking = {
                "tr_loss_last_epoch": float("inf"),
                "val_loss_last_epoch": tracker_val.stat["loss"].avg,
            }
    else:
        tracking = {
            "tr_loss_last_epoch": float("inf"),
            "val_loss_last_epoch": float("inf"),
        }
    conf.logger.log(
        f"finish {epoch_count} epoch on-server training: train={tracking['tr_loss_last_epoch']}, val={tracking['val_loss_last_epoch']}."
    )

    # on server training and validation.
    while True:
        epoch_count += 1
        tracker_tr = RuntimeTracker(metrics_to_track=[])
        tracker_val = RuntimeTracker(metrics_to_track=[])

        # train on the tr_data_loader.
        for _ind, (_input, _target) in enumerate(data_loaders["tr_data_loader"]):
            # place model and data.
            if conf.graph.on_cuda:
                _input, _target = _input.cuda(), _target.cuda()

            # inference and update alpha
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(_input), _target, smooth_eps=eps)
            tracker_tr.update_metrics([loss.item()], n_samples=_input.size(0))
            loss.backward()
            optimizer.step()

        # validate on the val_data_loader.
        if data_loaders["val_data_loader"] is not None:
            for _ind, (_input, _target) in enumerate(data_loaders["val_data_loader"]):
                # place model and data.
                if conf.graph.on_cuda:
                    _input, _target = _input.cuda(), _target.cuda()

                # inference and evaluate.
                model.eval()
                loss = criterion(model(_input), _target)
                tracker_val.update_metrics([loss.item()], n_samples=_input.size(0))

            # check the condition.
            if (
                tracker_tr.stat["loss"].avg < tracking["tr_loss_last_epoch"]
                and tracker_val.stat["loss"].avg < tracking["val_loss_last_epoch"]
            ):
                conf.logger.log(
                    f"finish {epoch_count} epoch on-server training: train={tracker_tr()}, val={tracker_val()}: will continue training."
                )
                final_model = copy.deepcopy(model)
            else:
                conf.logger.log(
                    f"finish {epoch_count} epoch on-server training: train={tracker_tr()}, val={tracker_val()}: will end training."
                )
                if conf.graph.on_cuda:
                    final_model = final_model.cpu()
                del model
                return final_model
        else:
            conf.logger.log(
                f"finish {epoch_count} epoch on-server training: {tracker_tr()}"
            )
            assert conf.fl_aggregate["epochs"] == "plateau"
            assert "epochs_max" in conf.fl_aggregate
            if (
                tracking["tr_loss_last_epoch"] - tracker_tr.stat["loss"].avg
                <= conf.fl_aggregate["plateau_tol"]
            ) or epoch_count >= conf.fl_aggregate["epochs_max"]:
                if conf.graph.on_cuda:
                    model = model.cpu()
                return model

        # update the tracking records.
        tracking = {
            "tr_loss_last_epoch": tracker_tr.stat["loss"].avg,
            "val_loss_last_epoch": tracker_val.stat["loss"].avg,
        }


def create_optimizer(conf, model):
    if conf.fl_aggregate["optim_name"] == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=conf.fl_aggregate["optim_lr"],
            betas=(conf.adam_beta_1, conf.adam_beta_2),
            eps=conf.adam_eps,
        )
    elif conf.fl_aggregate["optim_name"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=conf.fl_aggregate["optim_lr"],
            momentum=conf.momentum_factor,
            nesterov=conf.use_nesterov,
        )
    else:
        raise NotImplementedError("not supported yet.")
