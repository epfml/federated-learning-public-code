# -*- coding: utf-8 -*-
import json
import copy
import torch

import pcode.datasets.prepare_data as prepare_data
import pcode.datasets.partition_data as partition_data

import pcode.aggregation.fedavg as fedavg
import pcode.aggregation.knowledge_transfer as knowledge_transfer
import pcode.aggregation.swa_knowledge_transfer as swa_knowledge_transfer
import pcode.aggregation.adv_knowledge_transfer as adv_knowledge_transfer
import pcode.aggregation.noise_knowledge_transfer as noise_knowledge_transfer
import pcode.aggregation.server_momentum as server_momentum
import pcode.aggregation.server_adaptive as server_adaptive


class Aggregator(object):
    def __init__(
        self, conf, model, criterion, metrics, dataset, test_loaders, clientid2arch
    ):
        self.conf = conf
        self.logger = conf.logger
        self.model = copy.deepcopy(model)
        self.criterion = criterion
        self.metrics = metrics
        self.dataset = dataset
        self.test_loaders = test_loaders
        self.clientid2arch = clientid2arch

        # for the ease of aggregation.
        # some examples for the argparse.
        # 0. --fl_aggregate scheme=federated_average
        # 1. --fl_aggregate scheme=learning_to_aggregate,layerwise=False,data_name=cifar10,data_type=train,data_scheme=random_sampling,data_percentage=0.1,optim_name=adam,optim_lr=0.1,epochs=3
        # 2. --fl_aggregate scheme=optimal_transport
        # 3. --fl_aggregate scheme=unlabeled_train_on_server,soft_label_scheme=majority_vote,top1_starting_threshold=50,data_name=cifar10,data_type=train,data_scheme=random_sampling,data_percentage=0.1,val_percentage=0.1,optim_name=sgd,optim_lr=0.1,epochs_max=16,epochs=plateau,plateau_tol=1e-3,max_eps=0.1 \
        # 4. --fl_aggregate scheme=knowledge_transfer,construct_scheme=class_similarity_dirichlet,top1_starting_threshold=50,outer_iters=10,inner_epochs=256,kt_batch_size_per_class=250,kt_g_batch_size_per_class=128,kt_scaling_factors=1:0.1,kt_data_generate_iters=1024,step_size=0.001,optim_name=adam,optim_lr=0.001
        # 5. --fl_aggregate scheme=adv_knowledge_transfer,top1_starting_threshold=50,data_name=cifar10,total_n_pseudo_batches=1000,generator_learning_rate=1e-4,student_learning_rate=2e-4
        if conf.fl_aggregate is not None:
            if conf.fl_aggregate["scheme"] == "adv_knowledge_transfer":
                self.data_info = self._define_aggregation_data(return_loader=True)
            elif conf.fl_aggregate["scheme"] == "noise_knowledge_transfer":
                self.data_info = self._define_aggregation_data(return_loader=True)
            elif (
                conf.fl_aggregate["scheme"] == "federated_average"
                or conf.fl_aggregate["scheme"] == "server_momentum"
            ) and (
                "server_teaching_scheme" not in conf.fl_aggregate
                or "drop" in conf.fl_aggregate["server_teaching_scheme"]
            ):
                self.data_info = self._define_aggregation_data(return_loader=True)

        # define the aggregation function.
        self._define_aggregate_fn()

    def _define_aggregate_fn(self):
        if (
            self.conf.fl_aggregate is None
            or self.conf.fl_aggregate["scheme"] == "federated_average"
        ):
            self.aggregate_fn = None
        elif self.conf.fl_aggregate["scheme"] == "knowledge_transfer":
            self.aggregate_fn = self._s5_knowledge_transfer()
        elif self.conf.fl_aggregate["scheme"] == "adv_knowledge_transfer":
            self.aggregate_fn = self._s6_adv_knowledge_transfer()
        elif self.conf.fl_aggregate["scheme"] == "noise_knowledge_transfer":
            self.aggregate_fn = self._s7_noise_knowledge_transfer()
        elif self.conf.fl_aggregate["scheme"] == "server_momentum":
            self.aggregate_fn = self._s8_server_momentum()
        elif self.conf.fl_aggregate["scheme"] == "server_adaptive":
            self.aggregate_fn = self._s9_server_adaptive()
        else:
            raise NotImplementedError

    def _s1_federated_average(self):
        # global-wise averaging scheme.
        def f(**kwargs):
            return fedavg.fedavg(
                conf=self.conf,
                clientid2arch=self.clientid2arch,
                n_selected_clients=kwargs["n_selected_clients"],
                flatten_local_models=kwargs["flatten_local_models"],
                client_models=kwargs["client_models"],
                criterion=self.criterion,
                metrics=self.metrics,
                val_data_loader=self.data_info["self_val_data_loader"],
            )

        return f

    def _s5_knowledge_transfer(self):
        def f(**kwargs):
            _client_models = knowledge_transfer.aggregate(
                conf=self.conf,
                fedavg_model=kwargs["fedavg_model"],
                client_models=kwargs["client_models"],
                criterion=self.criterion,
                metrics=self.metrics,
                flatten_local_models=kwargs["flatten_local_models"],
                fa_val_perf=kwargs["performance"],
            )
            return _client_models

        return f

    def _s6_adv_knowledge_transfer(self):
        def f(**kwargs):
            _client_models = adv_knowledge_transfer.aggregate(
                conf=self.conf,
                fedavg_models=kwargs["fedavg_models"],
                client_models=kwargs["client_models"],
                criterion=self.criterion,
                metrics=self.metrics,
                flatten_local_models=kwargs["flatten_local_models"],
                fa_val_perf=kwargs["performance"],
                val_data_loader=self.data_info["self_val_data_loader"],
            )
            return _client_models

        return f

    def _s7_noise_knowledge_transfer(self):
        def f(**kwargs):
            _client_models = noise_knowledge_transfer.aggregate(
                conf=self.conf,
                fedavg_models=kwargs["fedavg_models"],
                client_models=kwargs["client_models"],
                criterion=self.criterion,
                metrics=self.metrics,
                flatten_local_models=kwargs["flatten_local_models"],
                fa_val_perf=kwargs["performance"],
                distillation_sampler=self.data_info["sampler"],
                distillation_data_loader=self.data_info["data_loader"],
                val_data_loader=self.data_info["self_val_data_loader"],
                test_data_loader=self.test_loaders[0],
            )
            return _client_models

        return f

    def _s8_server_momentum(self):
        def f(**kwargs):
            _client_models = server_momentum.aggregate(
                conf=self.conf,
                master_model=kwargs["master_model"],
                fedavg_model=kwargs["fedavg_model"],
                client_models=kwargs["client_models"],
                flatten_local_models=kwargs["flatten_local_models"],
            )
            return _client_models

        return f

    def _s9_server_adaptive(self):
        def f(**kwargs):
            _client_models = server_adaptive.aggregate(
                conf=self.conf,
                master_model=kwargs["master_model"],
                fedavg_model=kwargs["fedavg_model"],
                client_models=kwargs["client_models"],
                flatten_local_models=kwargs["flatten_local_models"],
            )
            return _client_models

        return f

    def _s10_swa_knowledge_transfer(self):
        def f(**kwargs):
            _client_models = swa_knowledge_transfer.aggregate(
                conf=self.conf,
                fedavg_models=kwargs["fedavg_models"],
                client_models=kwargs["client_models"],
                criterion=self.criterion,
                metrics=self.metrics,
                flatten_local_models=kwargs["flatten_local_models"],
                fa_val_perf=kwargs["performance"],
                distillation_sampler=self.data_info["sampler"],
                distillation_data_loader=self.data_info["data_loader"],
                val_data_loader=self.data_info["self_val_data_loader"],
                test_data_loader=self.test_loaders[0],
            )
            return _client_models

        return f

    def aggregate(
        self,
        master_model,
        client_models,
        flatten_local_models,
        aggregate_fn_name=None,
        **kwargs,
    ):
        n_selected_clients = len(flatten_local_models)

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
            flatten_local_models=flatten_local_models,
            n_selected_clients=n_selected_clients,
            **kwargs,
        )

    def _define_aggregation_data(self, return_loader=True):
        # init.
        fl_aggregate = self.conf.fl_aggregate

        # prepare the data.
        if self.dataset["val"] is not None:
            # prepare the dataloader.
            data_loader = torch.utils.data.DataLoader(
                self.dataset["val"],
                batch_size=self.conf.batch_size,
                shuffle=False,
                num_workers=self.conf.num_workers,
                pin_memory=self.conf.pin_memory,
                drop_last=False,
            )
            # define things to return.
            things_to_return = {"self_val_data_loader": data_loader}
        else:
            things_to_return = {}

        if "data_source" in fl_aggregate and "other" in fl_aggregate["data_source"]:
            assert (
                "data_name" in fl_aggregate
                and "data_scheme" in fl_aggregate
                and "data_type" in fl_aggregate
            )

            # create dataset.
            self.logger.log(f'create data={fl_aggregate["data_name"]} for aggregation.')
            dataset = prepare_data.get_dataset(
                self.conf,
                fl_aggregate["data_name"],
                datasets_path=self.conf.data_dir
                if "data_dir" not in fl_aggregate
                else fl_aggregate["data_dir"],
                split="train",
            )
            self.logger.log(
                f'created data={fl_aggregate["data_name"]} for aggregation with size {len(dataset)}.'
            )

            # sample the indices from the dataset.
            if fl_aggregate["data_scheme"] == "random_sampling":
                assert "data_percentage" in fl_aggregate
                sampler = partition_data.DataSampler(
                    self.conf,
                    data=dataset,
                    data_scheme=fl_aggregate["data_scheme"],
                    data_percentage=fl_aggregate["data_percentage"],
                )
            elif fl_aggregate["data_scheme"] == "class_selection":
                assert "num_overlap_class" in fl_aggregate
                assert "num_total_class" in fl_aggregate
                assert self.conf.data == "cifar100"
                assert "imagenet" in self.conf.fl_aggregate["data_name"]

                #
                selected_imagenet_classes = partition_data.get_imagenet1k_classes(
                    num_overlap_classes=int(fl_aggregate["num_overlap_class"]),
                    random_state=self.conf.random_state,
                    num_total_classes=int(
                        fl_aggregate["num_total_class"]
                    ),  # for cifar-100
                )
                sampler = partition_data.DataSampler(
                    self.conf,
                    data=dataset,
                    data_scheme=fl_aggregate["data_scheme"],
                    data_percentage=fl_aggregate["data_percentage"]
                    if "data_percentage" in fl_aggregate
                    else None,
                    selected_classes=selected_imagenet_classes,
                )
            else:
                raise NotImplementedError("invalid data_scheme")

            sampler.sample_indices()

            # define things to return.
            things_to_return.update({"sampler": sampler})

            if return_loader:
                data_loader = torch.utils.data.DataLoader(
                    sampler.use_indices(),
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
