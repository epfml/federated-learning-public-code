# -*- coding: utf-8 -*-
import copy
import collections

import torch
import torchcontrib
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


from pcode.aggregation.adv_knowledge_transfer import BaseKTSolver
import pcode.aggregation.utils as agg_utils
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.master_utils as master_utils
import pcode.aggregation.swag_utils.swag as swag


def get_unlabeled_data(fl_aggregate, distillation_data_loader):
    if (
        "use_data_scheme" not in fl_aggregate
        or fl_aggregate["use_data_scheme"] == "real_data"
    ):
        return distillation_data_loader
    else:
        return None


def sample_from_swag(conf, model, client_models, loader=None):
    swag_model_root = swag.SWAG(
        model.cuda() if conf.graph.on_cuda else model,
        no_cov_mat=False,
        max_num_models=len(client_models),
    )

    # collect models.
    for _id, client_model in client_models.items():
        swag_model_root.collect_model(client_model)

    # sample from swag_model.
    conf.logger.log("sampling model from swag.")
    sampled_models = {}
    for idx in range(int(conf.fl_aggregate["swag_n_sampled_model"])):
        # sample.
        swag_model_root.sample(scale=conf.fl_aggregate["swag_sample_scale"], cov=True)

        # update bn.
        if loader is not None:
            swag.bn_update(swag_model_root, loader)
        sampled_models[idx] = copy.deepcopy(swag_model_root)

    # include client models or not.
    if (
        "include_client_models" in conf.fl_aggregate
        and conf.fl_aggregate["include_client_models"]
    ):
        for _id, client_model in client_models.items():
            sampled_models[idx + _id + 1] = copy.deepcopy(client_model)

    conf.logger.log(f"sampled {len(sampled_models)} model from SWAG.")
    return sampled_models


def aggregate(
    conf,
    fedavg_models,
    client_models,
    criterion,
    metrics,
    flatten_local_models,
    fa_val_perf,
    distillation_sampler,
    distillation_data_loader,
    val_data_loader,
    test_data_loader,
):
    fl_aggregate = conf.fl_aggregate

    # recover the models on the computation device.
    _, local_models = agg_utils.recover_models(
        conf, client_models, flatten_local_models, use_cuda=conf.graph.on_cuda
    )

    # evaluate the local model on the test_loader
    if "eval_local" in fl_aggregate and fl_aggregate["eval_local"]:
        perfs = []
        for idx, local_model in enumerate(local_models.values()):
            conf.logger.log(f"Evaluate the local model-{idx}.")
            perf = master_utils.validate(
                conf,
                coordinator=None,
                model=local_model,
                criterion=criterion,
                metrics=metrics,
                data_loader=test_data_loader,
                label=None,
                display=False,
            )
            perfs.append(perf["top1"])
        conf.logger.log(
            f"The averaged test performance of the local models: {sum(perfs) / len(perfs)}; the details of the local performance: {perfs}."
        )

    # evaluate the ensemble of local models on the test_loader
    if "eval_ensemble" in fl_aggregate and fl_aggregate["eval_ensemble"]:
        master_utils.ensembled_validate(
            conf,
            coordinator=None,
            models=list(local_models.values()),
            criterion=criterion,
            metrics=metrics,
            data_loader=test_data_loader,
            label="ensemble_test_loader",
            ensemble_scheme=None
            if "update_student_scheme" not in fl_aggregate
            else fl_aggregate["update_student_scheme"],
        )

    # distillation.
    _client_models = {}
    for arch, fedavg_model in fedavg_models.items():
        conf.logger.log(
            f"Master: we have {len(local_models)} local models for noise distillation (use {arch} for the distillation)."
        )

        # sample models.
        assert len(fedavg_models) == 1, "right now, we only support homo-arch case."
        # TODO.
        sampled_models = sample_from_swag(
            conf, fedavg_model, local_models, loader=val_data_loader
        )

        # initialize knowledge distillation solver.
        kt = SWAKTSolver(
            conf=conf,
            teacher_models=list(sampled_models.values()),
            student_model=fedavg_model,
            criterion=criterion,
            metrics=metrics,
            batch_size=128
            if "batch_size" not in fl_aggregate
            else int(fl_aggregate["batch_size"]),
            total_n_server_pseudo_batches=250
            if "total_n_server_pseudo_batches" not in fl_aggregate
            else int(fl_aggregate["total_n_server_pseudo_batches"]),
            val_data_loader=val_data_loader,
            distillation_sampler=distillation_sampler,
            distillation_data_loader=get_unlabeled_data(
                fl_aggregate, distillation_data_loader
            ),
            student_learning_rate=1e-3
            if "student_learning_rate" not in fl_aggregate
            else fl_aggregate["student_learning_rate"],
            AT_beta=0 if "AT_beta" not in fl_aggregate else fl_aggregate["AT_beta"],
            KL_temperature=1
            if "temperature" not in fl_aggregate
            else fl_aggregate["temperature"],
            log_fn=conf.logger.log,
            eval_batches_freq=100
            if "eval_batches_freq" not in fl_aggregate
            else int(fl_aggregate["eval_batches_freq"]),
            update_student_scheme="avg_logits",
            server_teaching_scheme=None
            if "server_teaching_scheme" not in fl_aggregate
            else fl_aggregate["server_teaching_scheme"],
            optimizer="sgd"
            if "optimizer" not in fl_aggregate
            else fl_aggregate["optimizer"],
        )
        getattr(
            kt,
            "distillation"
            if "noise_kt_scheme" not in fl_aggregate
            else fl_aggregate["noise_kt_scheme"],
        )()
        _client_models[arch] = kt.server_student.cpu()

    # free the memory.
    del local_models, sampled_models, kt
    torch.cuda.empty_cache()
    return _client_models


class SWAKTSolver(object):
    """ Main solver class to transfer the knowledge through noise or unlabelled data."""

    def __init__(
        self,
        conf,
        teacher_models,
        student_model,
        criterion,
        metrics,
        batch_size,
        total_n_server_pseudo_batches=0,
        val_data_loader=None,
        distillation_sampler=None,
        distillation_data_loader=None,
        student_learning_rate=1e-3,
        AT_beta=0,
        KL_temperature=1,
        log_fn=print,
        eval_batches_freq=100,
        update_student_scheme="avg_logits",  # either avg_losses or avg_logits
        server_teaching_scheme=None,
        optimizer=None,
    ):
        # general init.
        self.conf = conf
        self.device = (
            torch.device("cuda") if conf.graph.on_cuda else torch.device("cpu")
        )

        # init the validation criterion and metrics.
        self.criterion = criterion
        self.metrics = metrics

        # init training logics and loaders.
        self.batch_size = batch_size
        self.total_n_server_pseudo_batches = total_n_server_pseudo_batches

        # init the fundamental solver.
        self.base_solver = BaseKTSolver(KL_temperature=KL_temperature, AT_beta=AT_beta)

        # init student and teacher nets.
        self.numb_teachers = len(teacher_models)
        self.server_student = self.base_solver.prepare_model(
            conf, student_model, self.device, _is_teacher=False
        )
        self.client_teachers = [
            self.base_solver.prepare_model(
                conf, _teacher, self.device, _is_teacher=True
            )
            for _teacher in teacher_models
        ]

        # init the loaders.
        self.val_data_loader = val_data_loader
        self.distillation_sampler = distillation_sampler
        self.distillation_data_loader = self.preprocess_unlabeled_real_data(
            distillation_sampler, distillation_data_loader
        )

        # init the optimizers.
        if optimizer is None or optimizer == "adam":
            self.optimizer_server_student = optim.Adam(
                self.server_student.parameters(), lr=student_learning_rate
            )
        elif optimizer == "sgd":
            self.optimizer_server_student = optim.SGD(
                self.server_student.parameters(), lr=student_learning_rate
            )
        else:
            raise NotImplementedError("incorrect optimizer for model fusion.")

        # init for SWA with the initialized optimizer.
        self.swa_optimizer = torchcontrib.optim.SWA(
            self.optimizer_server_student, swa_start=0, swa_freq=25, swa_lr=1e-3
        )

        # For distillation.
        self.AT_beta = AT_beta
        self.KL_temperature = KL_temperature

        # Set up & Resume
        self.log_fn = log_fn
        self.eval_batches_freq = eval_batches_freq
        self.update_student_scheme = update_student_scheme
        self.server_teaching_scheme = server_teaching_scheme
        print("\tFinished the initialization for NoiseKTSolver.")

    """related to distillation."""

    def distillation(self):
        # init the tracker.
        server_tracker = RuntimeTracker(
            metrics_to_track=["student_loss"], force_to_replace_metrics=True
        )

        # init the data iter.
        if self.distillation_data_loader is not None:
            data_iter = iter(self.distillation_data_loader)

        # get the client_weights from client's validation performance.
        client_weights = self._get_client_weights()

        # get the init server perf.
        init_perf_on_val = self.validate(
            model=self.server_student, data_loader=self.val_data_loader
        )

        # iterate over dataset
        n_pseudo_batches = 0
        self.log_fn(
            f"Batch {n_pseudo_batches}/{self.total_n_server_pseudo_batches}: Student Validation Acc={init_perf_on_val}."
        )
        while n_pseudo_batches < self.total_n_server_pseudo_batches:
            # get the inputs.
            if self.distillation_data_loader is not None:
                try:
                    pseudo_data = next(data_iter)[0].to(device=self.device)
                except StopIteration:
                    data_iter = iter(self.distillation_data_loader)
                    pseudo_data = next(data_iter)[0].to(device=self.device)
            else:
                if self.conf.fl_aggregate["use_data_scheme"] == "random_data":
                    pseudo_data = self._create_data_randomly()
                else:
                    raise NotImplementedError("incorrect use_data_scheme.")

            # get the logits.
            with torch.no_grad():
                teacher_logits = [
                    _teacher(pseudo_data) for _teacher in self.client_teachers
                ]

            # steps on the same pseudo data
            student_logits = self.server_student(pseudo_data)
            student_logits_activations = [
                (student_logits, self.server_student.activations)
            ] * self.numb_teachers

            stud_avg_loss = self.update_student(
                student_logits_activations=student_logits_activations,
                base_solver=self.base_solver,
                _student=self.server_student,
                _teachers=self.client_teachers,
                _opt_student=self.swa_optimizer,
                teacher_logits=teacher_logits,
                update_student_scheme=self.update_student_scheme,
                weights=client_weights,
            )

            # update the tracker after each batch.
            server_tracker.update_metrics([stud_avg_loss], n_samples=self.batch_size)

            if (n_pseudo_batches + 1) % self.eval_batches_freq == 0:
                validated_perf = self.validate(
                    model=self.server_student, data_loader=self.val_data_loader
                )
                self.log_fn(
                    f"Batch {n_pseudo_batches + 1}/{self.total_n_server_pseudo_batches}: Student Loss={server_tracker.stat['student_loss'].avg:02.5f}; Student Validation Acc={validated_perf}."
                )
                server_tracker.reset()

            n_pseudo_batches += 1

        # update the server model.
        self.swa_optimizer.swap_swa_sgd()
        self.server_student = self.server_student.cpu()

    def _get_client_weights(self):
        if self.server_teaching_scheme is not None:
            # get the perf for teachers.
            weights = []
            indices_to_remove = []
            for idx, client_teacher in enumerate(self.client_teachers):
                validated_perf = self.validate(
                    model=client_teacher, data_loader=self.val_data_loader
                )
                perf = validated_perf["top1"]

                # check the perf.
                if perf < agg_utils.get_random_guess_perf(self.conf):
                    indices_to_remove.append(idx)
                    weights.append(0)
                else:
                    weights.append(perf)

            # get the normalized weights.
            sum_weights = sum(weights)
            normalized_weights = [x / sum_weights for x in weights]

            if self.server_teaching_scheme == "weighted":
                return normalized_weights
            elif "drop_worst" in self.server_teaching_scheme:
                self.log_fn(f"normalized weights: {normalized_weights}.")
                if len(indices_to_remove) == 0:
                    indices_to_remove, remained_weights = agg_utils.filter_models_by_weights(
                        normalized_weights
                    )
                else:
                    indices_to_remove = sorted(indices_to_remove)
                    remained_weights = [
                        weight for weight in normalized_weights if weight > 0
                    ]
                summed_remained_weights = sum(remained_weights)

                # update client_teacher.
                self.log_fn(f"indices to be removed: {indices_to_remove}.")
                for index in indices_to_remove[::-1]:
                    self.client_teachers.pop(index)

                # update the normalized weights.
                normalized_weights = [
                    weight / summed_remained_weights for weight in remained_weights
                ]

                if "weighted" in self.server_teaching_scheme:
                    return normalized_weights
                else:
                    return None
        else:
            return None

    @staticmethod
    def update_student(
        student_logits_activations,
        base_solver,
        _student,
        _teachers,
        _opt_student,
        teacher_logits,
        update_student_scheme,
        weights=None,
    ):
        # get weights.
        weights = (
            weights if weights is not None else [1.0 / len(_teachers)] * len(_teachers)
        )

        if update_student_scheme == "avg_losses":
            student_losses = [
                base_solver.KT_loss_student(
                    _student_logits,
                    _student_activations,
                    _teacher_logits,
                    _teacher.activations,
                )
                for (
                    _student_logits,
                    _student_activations,
                ), _teacher_logits, _teacher in zip(
                    student_logits_activations, teacher_logits, _teachers
                )
            ]
            student_avg_loss = sum(
                [loss * weight for loss, weight in zip(student_losses, weights)]
            )
        elif update_student_scheme == "avg_logits":
            student_logits, _ = student_logits_activations[0]
            teacher_avg_logits = sum(
                [
                    teacher_logit * weight
                    for teacher_logit, weight in zip(teacher_logits, weights)
                ]
            )
            student_avg_loss = base_solver.divergence(
                student_logits, teacher_avg_logits
            )
        elif update_student_scheme == "avg_probs":
            student_logits, _ = student_logits_activations[0]
            teacher_avg_probs = sum(
                [
                    F.softmax(teacher_logit, dim=1) * weight
                    for teacher_logit, weight in zip(teacher_logits, weights)
                ]
            )
            student_avg_loss = base_solver.divergence(
                student_logits, teacher_avg_probs, use_teacher_logits=False
            )
        else:
            raise NotImplementedError(
                f"the update_student_scheme={update_student_scheme} is not supported yet."
            )
        _opt_student.zero_grad()
        student_avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(_student.parameters(), 5)
        _opt_student.step()
        return student_avg_loss.item()

    def validate(
        self, model, data_loader=None, criterion=None, metrics=None, device=None
    ):
        if data_loader is None:
            return -1
        else:
            val_perf = master_utils.validate(
                conf=self.conf,
                coordinator=None,
                model=model,
                criterion=self.criterion if criterion is None else criterion,
                metrics=self.metrics if metrics is None else metrics,
                data_loader=data_loader,
                label=None,
                display=False,
            )
            model = model.to(self.device if device is None else device)
            model.train()
            return val_perf

    """related to dataset used for distillation."""

    def preprocess_unlabeled_real_data(
        self, distillation_sampler, distillation_data_loader
    ):
        if (
            "noise_kt_preprocess" not in self.conf.fl_aggregate
            or not self.conf.fl_aggregate["noise_kt_preprocess"]
        ):
            return distillation_data_loader

        # preprocessing for noise_kt.
        self.log_fn(f"preprocessing the unlabeled data.")

        ## prepare the data_loader.
        data_loader = torch.utils.data.DataLoader(
            distillation_sampler.use_indices(),
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=self.conf.num_workers,
            pin_memory=self.conf.pin_memory,
            drop_last=False,
        )

        ## evaluate the entropy of each data point.
        outputs = []
        for _input, _ in data_loader:
            _outputs = [
                F.softmax(client_teacher(_input.to(self.device)), dim=1)
                for client_teacher in self.client_teachers
            ]
            _entropy = Categorical(sum(_outputs) / len(_outputs)).entropy()
            outputs.append(_entropy)
        entropy = torch.cat(outputs)

        ## we pick samples that have low entropy.
        assert "noise_kt_preprocess_size" in self.conf.fl_aggregate
        noise_kt_preprocess_size = int(
            self.conf.fl_aggregate["noise_kt_preprocess_size"]
        )
        _, indices = torch.topk(
            entropy,
            k=min(len(entropy), noise_kt_preprocess_size),
            largest=False,
            sorted=False,
        )
        distillation_sampler.sampled_indices = distillation_sampler.sampled_indices[
            indices.cpu()
        ]

        ## create the dataloader.
        return torch.utils.data.DataLoader(
            distillation_sampler.use_indices(),
            batch_size=self.conf.batch_size,
            shuffle=self.conf.fl_aggregate["randomness"]
            if "randomness" in self.conf.fl_aggregate
            else True,
            num_workers=self.conf.num_workers,
            pin_memory=self.conf.pin_memory,
            drop_last=False,
        )

    def _create_data_randomly(self):
        # create pseudo_data and map to [0, 1].
        pseudo_data = torch.randn(
            (self.batch_size, 3, self.conf.img_resolution, self.conf.img_resolution),
            requires_grad=False,
        ).to(device=self.device)
        pseudo_data = (pseudo_data - torch.min(pseudo_data)) / (
            torch.max(pseudo_data) - torch.min(pseudo_data)
        )

        # map values to [-1, 1] if necessary.
        if self.conf.pn_normalize:
            pseudo_data = (pseudo_data - 0.5) * 2
        return pseudo_data
