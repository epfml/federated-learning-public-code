# -*- coding: utf-8 -*-
import json
import copy
import torch

import pcode.aggregation.noise_knowledge_transfer as noise_knowledge_transfer
from utils.module_state import ModuleState
import pcode.predictors.linear_predictors as linear_predictors
import configs.task_configs as task_configs
import pcode.create_dataset as create_dataset


class Aggregator(object):
    def __init__(
        self, conf, model, criterion, metrics, val_loader, test_loader, agg_dataset=None
    ):
        self.conf = conf
        self.logger = conf.logger
        self.model = copy.deepcopy(model)
        self.criterion = criterion
        self.metrics = metrics
        self.agg_dataset = agg_dataset
        self.val_loader = val_loader
        self.test_loader = test_loader

        # for the ease of aggregation.
        # some examples for the argparse.
        # 0. --fl_aggregate scheme=federated_average
        if conf.fl_aggregate is not None:
            if conf.fl_aggregate["scheme"] == "noise_knowledge_transfer":
                self.data_info = self._define_aggregation_data()

        # define the aggregation function.
        self._define_aggregate_fn()

    def _define_aggregate_fn(self):
        if (
            self.conf.fl_aggregate is None
            or self.conf.fl_aggregate["scheme"] == "federated_average"
        ):
            self.aggregate_fn = None
        elif self.conf.fl_aggregate["scheme"] == "noise_knowledge_transfer":
            self.aggregate_fn = self._s7_noise_knowledge_transfer()
        else:
            raise NotImplementedError

    def _s1_federated_average(self):
        # global-wise averaging scheme.
        def f(**kwargs):
            weights = [
                torch.FloatTensor([1.0 / kwargs["n_selected_clients"]])
                for _ in range(kwargs["n_selected_clients"])
            ]

            # NOTE: the arch for different local models needs to be the same as the master model
            # uniformly average the local models.
            # assume we use the runtime stat from the last model.
            _model = copy.deepcopy(kwargs["master_model"])
            local_states = [
                ModuleState(copy.deepcopy(local_model.state_dict()))
                for local_model in kwargs["client_models"]
            ]
            model_state = local_states[0] * weights[0]
            for idx in range(1, len(local_states)):
                model_state += local_states[idx] * weights[idx]
            model_state.copy_to_module(_model)
            return _model

        return f

    def _s7_noise_knowledge_transfer(self):
        def f(**kwargs):
            _master_model = noise_knowledge_transfer.aggregate(
                conf=self.conf,
                master_model=kwargs["master_model"],
                fedavg_model=kwargs["fedavg_model"],
                client_models=kwargs["client_models"],
                criterion=self.criterion,
                metrics=self.metrics,
                fa_val_perf=kwargs["performance"],
                distillation_data_loader=self.data_info["data_loader"],
                val_data_loader=self.data_info["self_val_data_loader"],
                test_data_loader=self.test_loader,
            )
            return _master_model

        return f

    def aggregate(self, master_model, client_models, aggregate_fn_name=None, **kwargs):
        n_selected_clients = len(client_models)

        # apply advanced aggregate_fn.
        self.logger.log(
            f"Aggregator via {aggregate_fn_name if aggregate_fn_name is not None else self.conf.fl_aggregate['scheme']}: {f'scheme={json.dumps(self.conf.fl_aggregate)}' if self.conf.fl_aggregate is not None else ''}"
        )
        _aggregate_fn = (
            self.aggregate_fn
            if aggregate_fn_name is None
            else getattr(self, aggregate_fn_name)()
        )
        return _aggregate_fn(
            master_model=master_model,
            client_models=client_models,
            n_selected_clients=n_selected_clients,
            **kwargs,
        )

    def _define_aggregation_data(self):
        # init.
        fl_aggregate = self.conf.fl_aggregate

        # prepare the data.
        if self.val_loader is not None:
            # define things to return.
            things_to_return = {"self_val_data_loader": self.val_loader}
        else:
            things_to_return = {}

        # whether to aggregate with the training data or not
        if "data_source" in fl_aggregate and "other" in fl_aggregate["data_source"]:
            assert (
                "data_name" in fl_aggregate
                and "data_name" != self.conf.task
                and "agg_data_ratio" in fl_aggregate
            )
            # initialize a client tokenizer
            tokenizer = linear_predictors.ptl2classes[
                self.conf.model_info["ptl"]
            ].tokenizer.from_pretrained(self.conf.model_info["model"])

            # initialize dataset
            dataset = task_configs.task2dataiter[fl_aggregate["data_name"]](
                fl_aggregate["data_name"],
                self.conf.model_info["model"],
                tokenizer,
                self.conf.max_seq_len,
            )
            _, agg_dataset = create_dataset.split_train_dataset(
                conf=self.conf,
                train_dataset=dataset.trn_dl,
                agg_data_ratio=fl_aggregate["agg_data_ratio"],
            )

        else:
            agg_dataset = self.agg_dataset

        self.logger.log(
            f"created data from the training set for aggregation with size {len(agg_dataset)}."
        )

        data_loader = torch.utils.data.DataLoader(
            agg_dataset,
            batch_size=self.conf.batch_size,
            shuffle=fl_aggregate["randomness"]
            if "randomness" in fl_aggregate
            else True,
            num_workers=self.conf.num_workers,
            pin_memory=self.conf.pin_memory,
            drop_last=False,
        )
        things_to_return.update({"data_loader": data_loader})

        return things_to_return
