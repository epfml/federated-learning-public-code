# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torch.nn as nn

import pcode.create_metrics as create_metrics
from .base import BaseTrainer
from utils.stat_tracker import RuntimeTracker
from utils.timer import Timer
import torch.optim as optim
import pcode.validation as validation
from utils.early_stopping import EarlyStoppingTracker


class BertFinetuner(BaseTrainer):
    def __init__(self, conf, logger, data_iter, data_partitioner, val_dl, tst_dl):
        super(BertFinetuner, self).__init__(
            conf, logger, data_partitioner=data_partitioner
        )
        self.dataset = data_iter
        self.trn_dl, self.val_dl, self.tst_dl = None, val_dl, tst_dl
        self.task_metrics = data_iter.metrics

        # logging tools.
        self.timer = Timer(
            verbosity_level=1 if conf.track_time else 0,
            log_fn=self.log_fn_json,
            on_cuda=True,
        )
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_epochs
        )
        self.model_ptl = conf.ptl

    def batch_to_device(self, batched):
        uids = batched[0]
        input_ids, golds, attention_mask, token_type_ids = map(
            lambda x: x.cuda(), batched[1:]
        )
        return (
            uids,
            golds,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                # "token_type_ids": token_type_ids,
            },
            None,
        )

    def train(self, model, client_id, local_n_epochs):
        # init the model and dataloaders for the training.
        model, opt = self._init_training(model, client_id)
        model.train()

        # init the hook for the training.
        metrics = create_metrics.Metrics(model, task="classification")
        tracker = RuntimeTracker(metrics_to_track=metrics.metric_names)

        num_batch = len(self.trn_dl)
        self.log_fn(f"[INFO]: start training for task: {self.conf.task}")
        for epoch in range(1, self.conf.local_n_epochs + 1):
            for batched in self.trn_dl:
                self._batch_step += 1
                self._epoch_step = self._batch_step / num_batch

                with self.timer("load_data", epoch=self._epoch_step):
                    uids, golds, batched, _ = self.batch_to_device(batched)

                # forward for "pretrained model+classifier".
                with self.timer("forward_pass", epoch=self._epoch_step):
                    logits, *_ = model(**batched)
                    # the cross entropy by default uses reduction='mean'
                    loss = self.criterion(logits, golds)
                    tracker.update_metrics(
                        metric_stat=[loss.item()]
                        + metrics.evaluate(loss, logits, golds),
                        n_samples=len(logits),
                    )

                # backward for "pretrained model+classifier".
                with self.timer("backward_pass", epoch=self._epoch_step):
                    loss.backward()

                with self.timer("perform_update", epoch=self._epoch_step):
                    opt.step()
                    opt.zero_grad()

                # logging.
                self.log_fn(
                    "Round {} epoch {} for client-{}: Loss={}.".format(
                        self.conf.comm_round,
                        self._epoch_step,
                        client_id,
                        tracker.stat["loss"].avg,
                    )
                )

                # display the timer info.
                if (
                    self.conf.track_time
                    and self._batch_step % self.conf.summary_freq == 0
                ):
                    print(self.timer.summary())

            if self.conf.early_stopping_epochs < self.conf.local_n_epochs:
                # evaluation
                val_acc, _ = validation.evaluate(
                    conf=self.conf,
                    model=model,
                    dataloader=self.val_dl,
                    criterion=self.criterion,
                    back_to_cpu=False,
                    label=f"(client-{client_id})_validation_epoch_{self._epoch_step}",
                )

                # early stopping
                if self.early_stopping_tracker(val_acc):
                    self.log_fn_json(
                        name="test",
                        values={
                            "label": "early stopping at epoch {}: run out of patience.".format(
                                self._epoch_step
                            ),
                            "accuracy": val_acc,
                        },
                        tags={"split": "test"},
                        display=True,
                    )
                    break

            self._epoch_step += 1
            tracker.reset()

        acc, _ = validation.evaluate(
            conf=self.conf,
            model=model,
            dataloader=self.tst_dl,
            criterion=self.criterion,
            back_to_cpu=True,
            label=f"(client-{client_id})",
        )

        self._batch_step = 0

        return model, acc

    def _init_training(self, model, client_id):
        # init data loaders
        self._wrap_datasplits(self.dataset, client_id)

        model = self._parallel_to_device(model)

        # define the param to optimize.
        params = [
            {
                "params": [value],
                "name": key,
                "weight_decay": self.conf.weight_decay,
                "param_size": value.size(),
                "nelement": value.nelement(),
                "lr": self.conf.lr,
            }
            for key, value in model.named_parameters()
            if value.requires_grad
        ]

        # create the optimizer.
        if self.conf.optimizer == "adam":
            opt = optim.Adam(
                params,
                lr=self.conf.lr,
                betas=(self.conf.adam_beta_1, self.conf.adam_beta_2),
                eps=self.conf.adam_eps,
                weight_decay=self.conf.weight_decay,
            )
        elif self.conf.optimizer == "sgd":
            opt = optim.SGD(
                params,
                lr=self.conf.lr,
                momentum=self.conf.momentum_factor,
                weight_decay=self.conf.weight_decay,
                nesterov=self.conf.use_nesterov,
            )
        else:
            raise NotImplementedError("this optimizer is not supported yet.")
        opt.zero_grad()
        model.zero_grad()
        self.log_fn(f"Initialize the optimizer: {self.conf.optimizer}")
        return model, opt
