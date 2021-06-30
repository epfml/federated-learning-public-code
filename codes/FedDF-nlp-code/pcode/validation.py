import time
import collections

import numpy as np
import torch
import torch.nn.functional as F

import trainers.base as base
from utils.eval_meters import accuracy
import pcode.create_dataset as create_dataset


def evaluate(
    conf,
    model,
    dataloader,
    criterion,
    back_to_cpu=True,
    label="test_loader",
    save_jason=True,
):
    if not next(model.parameters()).is_cuda:
        model = base.parallel_to_device(conf, model)
    model.eval()

    all_losses, all_golds, all_preds = [], [], []
    for batched in dataloader:
        # golds is used for compute loss, _golds used for i2t convertion
        uids, golds, batched, _golds = create_dataset.seqcls_batch_to_device(batched)
        with torch.no_grad():
            logits, *_ = model(**batched)
            loss = criterion(logits, golds).mean().item()
            preds = torch.argmax(logits, dim=-1, keepdim=False)
            all_losses.append(loss)
            all_preds.extend(preds.detach().cpu().numpy())
            all_golds.extend(golds.detach().cpu().numpy())

    eval_res = accuracy(all_preds, all_golds)
    loss = np.mean(all_losses)

    conf.logger.log(
        f"[INFO]: gold distribution on accuracy: {collections.Counter(all_golds)}"
    )

    # logging.
    conf.logger.log(f"[INFO] Finished evaluation on {label}")
    if save_jason:
        conf.logger.log_metric(
            name="test",
            values={
                "label": label,
                "comm_round": conf.comm_round,
                "loss": loss,
                "accuracy": eval_res,
            },
            tags={"split": "test"},
            display=True,
        )
        conf.logger.save_json()

    if back_to_cpu:
        model.cpu()

    model.train()
    return eval_res, loss


def ensembled_validate(conf, models, data_loader, ensemble_scheme=None):
    """A function for model evaluation."""
    if data_loader is None:
        return None

    # switch to evaluation mode.
    for model in models:
        model.eval()
        model = base.parallel_to_device(conf, model)

    all_losses, all_golds, all_preds = [], [], []

    # evaluate on test_loader.
    for batched in data_loader:
        # golds is used for compute loss, _golds used for i2t convertion
        uids, golds, batched, _golds = create_dataset.seqcls_batch_to_device(batched)

        with torch.no_grad():
            # ensemble.
            if (
                ensemble_scheme is None
                or ensemble_scheme == "avg_losses"
                or ensemble_scheme == "avg_logits"
            ):
                outputs = []
                for model in models:
                    outputs.append(model(**batched)[0])
                output = sum(outputs) / len(outputs)
            elif ensemble_scheme == "avg_probs":
                outputs = []
                for model in models:
                    outputs.append(F.softmax(model(**batched)[0]))
                output = sum(outputs) / len(outputs)

            # eval the performance.
            preds = torch.argmax(output, dim=-1, keepdim=False)
            all_preds.extend(preds.detach().cpu().numpy())
            all_golds.extend(golds.detach().cpu().numpy())

    eval_res = accuracy(all_preds, all_golds)

    # place back model to the cpu.
    for model in models:
        model = model.cpu()
        model.train()

    # logging.
    conf.logger.log(f"[INFO] Finished Ensemble evaluation:")
    conf.logger.log_metric(
        name="test",
        values={
            "label": "ensemble evaluation",
            "comm_round": conf.comm_round,
            "accuracy": eval_res,
        },
        tags={"split": "test"},
        display=True,
    )
    conf.logger.save_json()

    return eval_res
