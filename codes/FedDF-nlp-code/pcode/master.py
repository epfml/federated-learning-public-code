# -*- coding: utf-8 -*-
import os
import copy

import numpy as np
import torch
import copy

import utils.cross_entropy as cross_entropy
from utils.early_stopping import EarlyStoppingTracker
import utils.param_parser as param_parser
import trainers.finetuner as finetuner
import configs.task_configs as task_configs
import pcode.create_coordinator as create_coordinator
import pcode.create_aggregator as create_aggregator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import utils.checkpoint as checkpoint
from utils.tensor_buffer import TensorBuffer
import pcode.validation as validation
import pcode.predictors.linear_predictors as linear_predictors


class Master(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.client_ids = list(range(1, 1 + conf.n_clients))

        # init the task.
        self.master_model, self.dataset = init_task(conf)

        # partitioner
        _, self.data_partitioner, agg_dataset, val_dl = create_dataset.define_data_loader(
            self.conf,
            dataset=self.dataset.trn_dl,
            localdata_id=0,  # random id here.
            is_train=True,
            data_partitioner=None,
            agg_data_ratio=self.conf.fl_aggregate["agg_data_ratio"],
            return_val=False if hasattr(self.dataset, "tst_dl") else True,
        )

        if val_dl:
            # assign the original validation set as the test set
            self.test_loader = create_dataset.define_data_loader(
                conf, self.dataset.val_dl, is_train=False
            )

            # assign the train set split as the validation set
            self.val_loader = create_dataset.define_data_loader(
                conf, val_dl, is_train=False
            )
        else:
            self.val_loader = create_dataset.define_data_loader(
                conf, self.dataset.val_dl, is_train=False
            )
            self.test_loader = create_dataset.define_data_loader(
                conf, self.dataset.tst_dl, is_train=False
            )

        # define the criterion and metrics.
        self.criterion = cross_entropy.CrossEntropyLoss(reduction="mean")
        self.metrics = create_metrics.Metrics(self.master_model, task="classification")
        self.conf.perfs = {}

        # define the aggregators.
        self.aggregator = create_aggregator.Aggregator(
            conf,
            model=self.master_model,
            criterion=self.criterion,
            metrics=self.metrics,
            agg_dataset=agg_dataset,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
        )
        self.coordinator = create_coordinator.Coordinator(conf, self.metrics)

        # define the local trainer
        self.local_trainer = finetuner.BertFinetuner(
            conf,
            logger=conf.logger,
            data_iter=self.dataset,
            data_partitioner=self.data_partitioner,
            val_dl=self.val_loader,
            tst_dl=self.test_loader,
        )

        conf.logger.log(
            f"Master initialized model/dataset/aggregator/criterion/metrics/worker."
        )

        # define early_stopping_tracker.
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )

        # save arguments to disk.
        conf.is_finished = False
        checkpoint.save_arguments(conf)

    def run(self):
        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            self.conf.comm_round = comm_round
            self.conf.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # get random n_local_epochs.
            list_of_local_n_epochs = get_n_local_epoch(
                conf=self.conf, n_participated=self.conf.n_participated
            )

            # random select clients from a pool.
            selected_client_ids = self._random_select_clients()

            # detect early stopping.
            self._check_early_stopping()
            if self.conf.is_finished:
                self.conf.logger.log(
                    f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                )
                return

            # perform local training for each client sequentially
            client_models = []
            client_perfs = []
            for client_id, local_n_epochs in zip(
                selected_client_ids, list_of_local_n_epochs
            ):
                _cli_model, _cli_perf = self.local_trainer.train(
                    model=copy.deepcopy(self.master_model),
                    client_id=client_id,
                    local_n_epochs=local_n_epochs,
                )

                client_models.append(_cli_model)
                client_perfs.append(_cli_perf)
            self.conf.perfs["client_avg"] = np.mean(client_perfs)

            # aggregate the local models and evaluate on the validation dataset.
            self._aggregate_model_and_evaluate(client_models)

            # evaluate the aggregated model.
            self.conf.logger.log(f"Master finished one round of federated learning.\n")

        # formally stop the training (the master has finished all communication rounds).
        self._finishing()

    def _random_select_clients(self):
        selected_client_ids = self.conf.random_state.choice(
            self.client_ids, self.conf.n_participated, replace=False
        ).tolist()
        selected_client_ids.sort()
        self.conf.logger.log(
            f"Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
        )
        return selected_client_ids

    def _aggregate_model_and_evaluate(self, client_models):
        # right now we only support to aggregate the model with same arch.
        fedavg_model = self.aggregator.aggregate(
            master_model=self.master_model,
            client_models=client_models,
            aggregate_fn_name="_s1_federated_average",
        )

        self.conf.perfs["fedavg"], _ = validation.evaluate(
            conf=self.conf,
            model=fedavg_model,
            dataloader=self.test_loader,
            criterion=self.criterion,
            label="fedavg_model_on_test_loader",
            back_to_cpu=True,
        )

        # (smarter) aggregate the model from clients.
        # note that: if conf.fl_aggregate["scheme"] == "federated_average",
        #            then self.aggregator.aggregate_fn = None.
        if self.aggregator.aggregate_fn is not None:
            # aggregate the local models.
            master_model = self.aggregator.aggregate(
                # self.master_model here will be the global model
                # that we sent out at the beginning of this comm. round.
                master_model=self.master_model,
                client_models=client_models,
                fedavg_model=fedavg_model,
                performance=None,
            )

            self.master_model.load_state_dict(master_model.state_dict())
            del master_model, fedavg_model

            self.conf.perfs["fusion_model"], _ = validation.evaluate(
                conf=self.conf,
                model=self.master_model,
                dataloader=self.test_loader,
                criterion=self.criterion,
                label="fused_model_on_test_loader",
                back_to_cpu=True,
            )

        else:
            # only update self.client_models in place.
            self.master_model.load_state_dict(fedavg_model.state_dict())
            del fedavg_model

        torch.cuda.empty_cache()

        self.conf.logger.log(f"[INFO] Performance Summary")
        self.conf.logger.log_metric(
            name="test",
            values={
                "comm_round": self.conf.comm_round,
                "perf": copy.deepcopy(self.conf.perfs),
            },
            tags={"split": "test"},
            display=True,
        )
        self.conf.logger.save_json()

    def _check_early_stopping(self):
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                self.coordinator.key_metric.cur_perf is not None
                and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.conf.logger.log("Master early stopping: meet target perf.")
                self.conf.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.conf.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True
        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.conf.comm_round - 1
            self.conf.comm_round = -1
            self._finishing(_comm_round)

    def _finishing(self, _comm_round=None):
        self.conf.logger.save_json()
        self.conf.logger.log(f"Master finished the federated learning.")
        self.conf.is_finished = True
        self.conf.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")


def get_n_local_epoch(conf, n_participated):
    if conf.min_local_epochs is None:
        return [conf.local_n_epochs] * n_participated
    else:
        # here we only consider to (uniformly) randomly sample the local epochs.
        assert conf.min_local_epochs > 1.0
        random_local_n_epochs = conf.random_state.uniform(
            low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
        )
        return random_local_n_epochs


def init_task(conf):
    # get model info for both the master and workers
    conf.logger.log(
        f"The master will use model={conf.model_info['model']} and pretrain weight={conf.model_info['ptl']}."
    )

    # initialize a client tokenizer
    tokenizer = linear_predictors.ptl2classes[
        conf.model_info["ptl"]
    ].tokenizer.from_pretrained(conf.model_info["model"])

    # initialize dataset
    data_iter = task_configs.task2dataiter[conf.task](
        conf.task, conf.model_info["model"], tokenizer, conf.max_seq_len, conf=conf
    )

    # initialize the master model and the client model template
    model = linear_predictors.ptl2classes[
        conf.model_info["ptl"]
    ].seqcls.from_pretrained(
        conf.model_info["model"],
        num_labels=data_iter.num_labels,
        cache_dir=conf.pretrained_weight_path,
    )

    conf.logger.log(
        f"Creating and loading pretrained {conf.model_info['ptl'].upper()} model."
    )

    return model, data_iter
