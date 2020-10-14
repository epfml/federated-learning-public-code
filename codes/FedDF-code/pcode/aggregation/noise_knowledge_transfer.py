# -*- coding: utf-8 -*-
import copy
import collections

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


from pcode.aggregation.adv_knowledge_transfer import BaseKTSolver
import pcode.aggregation.utils as agg_utils
from pcode.utils.stat_tracker import RuntimeTracker, BestPerf
import pcode.master_utils as master_utils


def get_unlabeled_data(fl_aggregate, distillation_data_loader):
    if (
        "use_data_scheme" not in fl_aggregate
        or fl_aggregate["use_data_scheme"] == "real_data"
    ):
        return distillation_data_loader
    else:
        return None


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
        conf, client_models, flatten_local_models
    )

    # include model from previous comm. round.
    if (
        "include_previous_models" in fl_aggregate
        and fl_aggregate["include_previous_models"] > 0
    ):
        local_models = agg_utils.include_previous_models(conf, local_models)

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
        kt = NoiseKTSolver(
            conf=conf,
            teacher_models=list(local_models.values()),
            student_model=fedavg_model
            if "use_fedavg_as_start" not in fl_aggregate
            else (
                fedavg_model
                if fl_aggregate["use_fedavg_as_start"]
                else copy.deepcopy(client_models[arch])
            ),
            criterion=criterion,
            metrics=metrics,
            batch_size=128
            if "batch_size" not in fl_aggregate
            else int(fl_aggregate["batch_size"]),
            total_n_server_pseudo_batches=1000 * 10
            if "total_n_server_pseudo_batches" not in fl_aggregate
            else int(fl_aggregate["total_n_server_pseudo_batches"]),
            server_local_steps=1
            if "server_local_steps" not in fl_aggregate
            else int(fl_aggregate["server_local_steps"]),
            val_data_loader=val_data_loader,
            distillation_sampler=distillation_sampler,
            distillation_data_loader=get_unlabeled_data(
                fl_aggregate, distillation_data_loader
            ),
            use_server_model_scheduler=True
            if "use_server_model_scheduler" not in fl_aggregate
            else fl_aggregate["use_server_model_scheduler"],
            same_noise=True
            if "same_noise" not in fl_aggregate
            else fl_aggregate["same_noise"],
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
            early_stopping_server_batches=2000
            if "early_stopping_server_batches" not in fl_aggregate
            else int(fl_aggregate["early_stopping_server_batches"]),
            update_student_scheme="avg_losses"
            if "update_student_scheme" not in fl_aggregate
            else fl_aggregate["update_student_scheme"],
            server_teaching_scheme=None
            if "server_teaching_scheme" not in fl_aggregate
            else fl_aggregate["server_teaching_scheme"],
            return_best_model_on_val=False
            if "return_best_model_on_val" not in fl_aggregate
            else fl_aggregate["return_best_model_on_val"],
        )
        getattr(
            kt,
            "distillation"
            if "noise_kt_scheme" not in fl_aggregate
            else fl_aggregate["noise_kt_scheme"],
        )()
        _client_models[arch] = kt.server_student.cpu()

    # update local models from the current comm. round.
    if (
        "include_previous_models" in fl_aggregate
        and fl_aggregate["include_previous_models"] > 0
    ):
        agg_utils.update_previous_models(conf, _client_models)

    # free the memory.
    del local_models, kt
    torch.cuda.empty_cache()
    return _client_models


class NoiseKTSolver(object):
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
        server_local_steps=1,
        val_data_loader=None,
        distillation_sampler=None,
        distillation_data_loader=None,
        use_server_model_scheduler=True,
        same_noise=True,
        student_learning_rate=1e-3,
        AT_beta=0,
        KL_temperature=1,
        log_fn=print,
        eval_batches_freq=100,
        early_stopping_server_batches=1000,
        update_student_scheme="avg_losses",  # either avg_losses or avg_logits
        server_teaching_scheme=None,
        return_best_model_on_val=False,
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
        self.same_noise = same_noise
        self.batch_size = batch_size
        self.total_n_server_pseudo_batches = total_n_server_pseudo_batches
        self.server_local_steps = server_local_steps

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
        self.return_best_model_on_val = return_best_model_on_val
        self.init_server_student = copy.deepcopy(self.server_student)

        # init the loaders.
        self.val_data_loader = val_data_loader
        self.distillation_sampler = distillation_sampler
        self.distillation_data_loader = self.preprocess_unlabeled_real_data(
            distillation_sampler, distillation_data_loader
        )

        # init the optimizers.
        self.optimizer_server_student = optim.Adam(
            self.server_student.parameters(), lr=student_learning_rate
        )

        # init the training scheduler.
        self.server_teaching_scheme = server_teaching_scheme
        self.use_server_model_scheduler = use_server_model_scheduler
        self.scheduler_server_student = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_server_student,
            self.total_n_server_pseudo_batches,
            last_epoch=-1,
        )

        # For distillation.
        self.AT_beta = AT_beta
        self.KL_temperature = KL_temperature

        # Set up & Resume
        self.log_fn = log_fn
        self.eval_batches_freq = eval_batches_freq
        self.early_stopping_server_batches = early_stopping_server_batches
        self.update_student_scheme = update_student_scheme
        self.validated_perfs = collections.defaultdict(list)
        print("\tFinished the initialization for NoiseKTSolver.")

    """related to distillation."""

    def distillation(self):
        # init the tracker.
        server_tracker = RuntimeTracker(
            metrics_to_track=["student_loss"], force_to_replace_metrics=True
        )
        server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)

        # update the server generator/student
        n_pseudo_batches = 0
        best_models = [None]

        # init the data iter.
        if self.distillation_data_loader is not None:
            data_iter = iter(self.distillation_data_loader)

        # get the client_weights from client's validation performance.
        client_weights = self._get_client_weights()

        # get the init server perf.
        init_perf_on_val = self.validate(
            model=self.init_server_student, data_loader=self.val_data_loader
        )
        self.log_fn(
            f"Batch {n_pseudo_batches}/{self.total_n_server_pseudo_batches}: Student Validation Acc={init_perf_on_val}."
        )

        # iterate over dataset
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
            for _ in range(self.server_local_steps):
                student_logits = self.server_student(pseudo_data)
                student_logits_activations = [
                    (student_logits, self.server_student.activations)
                ] * self.numb_teachers

                stud_avg_loss = self.update_student(
                    student_logits_activations=student_logits_activations,
                    base_solver=self.base_solver,
                    _student=self.server_student,
                    _teachers=self.client_teachers,
                    _opt_student=self.optimizer_server_student,
                    teacher_logits=teacher_logits,
                    update_student_scheme=self.update_student_scheme,
                    weights=client_weights,
                )

            # after each batch.
            if self.use_server_model_scheduler:
                self.scheduler_server_student.step()

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

                # check early stopping.
                if self.base_solver.check_early_stopping(
                    model=self.server_student,
                    model_ind=0,
                    best_tracker=server_best_tracker,
                    validated_perf=validated_perf,
                    validated_perfs=self.validated_perfs,
                    perf_index=n_pseudo_batches + 1,
                    early_stopping_batches=self.early_stopping_server_batches,
                    best_models=best_models,
                ):
                    break
            n_pseudo_batches += 1

        # recover the best server model
        use_init_server_model = False
        if self.return_best_model_on_val:
            use_init_server_model = (
                True
                if init_perf_on_val["top1"] > server_best_tracker.best_perf
                else False
            )

        # get the server model.
        if use_init_server_model:
            self.log_fn("use init server model instead.")
            best_server_dict = self.init_server_student.state_dict()
        else:
            best_server_dict = best_models[0].state_dict()

        # update the server model.
        self.server_student.load_state_dict(best_server_dict)
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
                    (
                        indices_to_remove,
                        remained_weights,
                    ) = agg_utils.filter_models_by_weights(normalized_weights)
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
