# -*- coding: utf-8 -*-
import math
import copy
import random
import collections
from contextlib import closing

import numpy as np
import torch.multiprocessing as tmp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid

import pcode.aggregation.utils as agg_utils
from pcode.utils.stat_tracker import RuntimeTracker, BestPerf
import pcode.master_utils as master_utils


def aggregate(
    conf,
    fedavg_models,
    client_models,
    criterion,
    metrics,
    flatten_local_models,
    fa_val_perf,
    val_data_loader,
):
    # recover the models on the computation device.
    _, local_models = agg_utils.recover_models(
        conf, client_models, flatten_local_models
    )

    # distillation.
    client_models = {}
    for arch, fedavg_model in fedavg_models.items():
        kt = ZeroShotKTSolver(
            conf=conf,
            teacher_models=list(local_models.values()),
            student_model=fedavg_model,
            criterion=criterion,
            metrics=metrics,
            z_dim=100,
            n_generator_iter=1
            if "n_generator_iter" not in conf.fl_aggregate
            else int(conf.fl_aggregate["n_generator_iter"]),
            n_student_iter=10
            if "n_student_iter" not in conf.fl_aggregate
            else int(conf.fl_aggregate["n_student_iter"]),
            dataset=conf.fl_aggregate["data_name"],
            batch_size=128
            if "batch_size" not in conf.fl_aggregate
            else int(conf.fl_aggregate["batch_size"]),
            total_n_pseudo_batches=1000
            if "total_n_pseudo_batches" not in conf.fl_aggregate
            else int(conf.fl_aggregate["total_n_pseudo_batches"]),
            total_n_server_pseudo_batches=1000 * 10
            if "total_n_server_pseudo_batches" not in conf.fl_aggregate
            else int(conf.fl_aggregate["total_n_server_pseudo_batches"]),
            server_local_steps=1
            if "server_local_steps" not in conf.fl_aggregate
            else int(conf.fl_aggregate["server_local_steps"]),
            val_data_loader=val_data_loader,
            same_noise=True
            if "same_noise" not in conf.fl_aggregate
            else conf.fl_aggregate["same_noise"],
            generator=conf.generators[arch] if hasattr(conf, "generators") else None,
            client_generators=None,
            generator_learning_rate=1e-3
            if "generator_learning_rate" not in conf.fl_aggregate
            else conf.fl_aggregate["generator_learning_rate"],
            student_learning_rate=2e-3
            if "student_learning_rate" not in conf.fl_aggregate
            else conf.fl_aggregate["student_learning_rate"],
            AT_beta=0
            if "AT_beta" not in conf.fl_aggregate
            else conf.fl_aggregate["AT_beta"],
            KL_temperature=1
            if "temperature" not in conf.fl_aggregate
            else conf.fl_aggregate["temperature"],
            log_fn=conf.logger.log,
            eval_batches_freq=100
            if "eval_batches_freq" not in conf.fl_aggregate
            else int(conf.fl_aggregate["eval_batches_freq"]),
            early_stopping_batches=200
            if "early_stopping_batches" not in conf.fl_aggregate
            else int(conf.fl_aggregate["early_stopping_batches"]),
            early_stopping_server_batches=2000
            if "early_stopping_server_batches" not in conf.fl_aggregate
            else int(conf.fl_aggregate["early_stopping_server_batches"]),
            n_parallel_comp=2
            if "n_parallel_comp" not in conf.fl_aggregate
            else int(conf.fl_aggregate["n_parallel_comp"]),
            scheme_of_next_generator="optimal_generator_based_on_teacher"
            if "scheme_of_next_generator" not in conf.fl_aggregate
            else conf.fl_aggregate["scheme_of_next_generator"],
            weighted_server_teaching=False
            if "weighted_server_teaching" not in conf.fl_aggregate
            else conf.fl_aggregate["weighted_server_teaching"],
            ensemble_teaching=True
            if ("adv_kt_scheme" in conf.fl_aggregate)
            and (conf.fl_aggregate["adv_kt_scheme"] == "ensemble")
            else False,
        )
        if not hasattr(conf, "generators"):
            conf.generators = {}
        conf.generators[arch] = getattr(
            kt,
            "alternative_clients_teaching_v3_parallel"
            if "adv_kt_scheme" not in conf.fl_aggregate
            else conf.fl_aggregate["adv_kt_scheme"],
        )()
        client_models[arch] = kt.server_student.cpu()

    # free the memory.
    del local_models, kt
    torch.cuda.empty_cache()
    return client_models


"""related to optimizer."""


def create_optimizer(conf, model, lr=None):
    if conf.fl_aggregate["optim_name"] == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=conf.fl_aggregate["optim_lr"] if lr is None else lr,
            betas=(conf.adam_beta_1, conf.adam_beta_2),
            eps=conf.adam_eps,
        )
    elif conf.fl_aggregate["optim_name"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=conf.fl_aggregate["optim_lr"] if lr is None else lr,
            momentum=conf.momentum_factor,
            nesterov=conf.use_nesterov,
        )
    else:
        raise NotImplementedError("not supported yet.")


"""define a class as well as the associated functions for the zero-shot on-server knowledge training."""


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(z_dim, 128 * 8 ** 2),
            View((-1, 128, 8, 8)),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.BatchNorm2d(3, affine=True),
        )

    def forward(self, z):
        return self.layers(z)

    def print_shape(self, x):
        """
        For debugging purposes
        """
        act = x
        for layer in self.layers:
            act = layer(act)
            print("\n", layer, "---->", act.shape)


def visualize(x_norm, dataset):
    """
    This un-normalizes for visualization purposes only.
    """
    if dataset == "SVHN":
        mean = torch.Tensor([0.4377, 0.4438, 0.4728]).view(1, 3, 1, 1).to(x_norm.device)
        std = torch.Tensor([0.1980, 0.2010, 0.1970]).view(1, 3, 1, 1).to(x_norm.device)
        x = x_norm * std + mean
    elif dataset == "CIFAR10":
        mean = torch.Tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x_norm.device)
        std = torch.Tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(x_norm.device)
        x = x_norm * std + mean
    else:
        raise NotImplementedError

    return x


class LearnableLoader(nn.Module):
    def __init__(self, z_dim, n_repeat_batch, dataset, batch_size, device, size=1):
        """Infinite loader, which contains a learnable generator."""

        super(LearnableLoader, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_repeat_batch = n_repeat_batch
        self.z_dim = z_dim
        self.generator = Generator(z_dim).to(device=device)
        self.device = device
        self.size = size
        self.z = None

    def __next__(self):
        return self.generate()

    def _sample_noises(self, same_noise):
        def _create():
            return torch.randn((self.batch_size, self.z_dim), requires_grad=False).to(
                device=self.device
            )

        if same_noise or self.size == 1:
            return _create()
        else:
            return [_create() for _ in range(self.size)]

    def generate(self, same_noise=True):
        assert self.z is not None

        if self.size == 1:
            images = [self.generator(self.z)]
        else:
            if same_noise:
                images = [self.generator(self.z) for _ in range(self.size)]
            else:
                images = [self.generator(_z) for _z in self.z]
        return images

    def reset_z(self, same_noise):
        self.z = self._sample_noises(same_noise)

    def samples(self, n, grid=True):
        """
        :return: if grid returns single grid image, else
        returns n images.
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn((n, self.z_dim)).to(device=self.device)
            images = visualize(self.generator(z), dataset=self.dataset).cpu()
            if grid:
                images = make_grid(images, nrow=round(math.sqrt(n)), normalize=True)

        self.generator.train()
        return images

    def __iter__(self):
        return self


class ZeroShotKTSolver(object):
    """ Main solver class to train and test the generator and student adversarially."""

    def __init__(
        self,
        conf,
        teacher_models,
        student_model,
        criterion,
        metrics,
        z_dim,
        n_generator_iter,
        n_student_iter,
        dataset,
        batch_size,
        total_n_pseudo_batches=0,
        total_n_server_pseudo_batches=0,
        server_local_steps=1,
        val_data_loader=None,
        generator=None,
        client_generators=None,
        use_server_model_scheduler=True,
        use_server_generator_scheduler=True,
        use_client_model_scheduler=True,
        use_client_generator_scheduler=True,
        same_noise=True,
        generator_learning_rate=1e-3,
        student_learning_rate=2e-3,
        AT_beta=250,
        KL_temperature=1,
        log_fn=print,
        eval_batches_freq=100,
        early_stopping_batches=100,
        early_stopping_server_batches=1000,
        n_parallel_comp=1,
        scheme_of_next_generator="optimal_generator_based_on_teacher",
        weighted_server_teaching=False,
        ensemble_teaching=False,
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
        self.n_generator_iter = n_generator_iter
        self.n_student_iter = n_student_iter
        self.n_repeat_batch = n_generator_iter + n_student_iter
        self.total_n_pseudo_batches = total_n_pseudo_batches
        self.total_n_server_pseudo_batches = total_n_server_pseudo_batches
        self.server_local_steps = server_local_steps
        self.n_parallel_comp = n_parallel_comp
        self.ensemble_teaching = ensemble_teaching

        self.val_data_loader = val_data_loader

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
        self.client_students = [
            self.base_solver.prepare_model(
                conf, student_model, self.device, _is_teacher=False
            )
            for _ in range(self.numb_teachers)
        ]

        # init the generators.
        # init the server generator.
        if generator is None:
            self.server_generator = LearnableLoader(
                z_dim=z_dim,
                n_repeat_batch=self.n_repeat_batch,
                dataset=dataset,
                batch_size=batch_size,
                device=self.device,
                size=1 if self.ensemble_teaching else self.numb_teachers,
            ).to(device=self.device)
        else:
            self.server_generator = generator.to(device=self.device)

        # init the local generators.
        if client_generators is None:
            self.client_generators = [
                LearnableLoader(
                    z_dim=z_dim,
                    n_repeat_batch=self.n_repeat_batch,
                    dataset=dataset,
                    batch_size=batch_size,
                    device=self.device,
                    size=1,
                ).to(device=self.device)
                for _ in range(self.numb_teachers)
            ]
            for _gen in self.client_generators:
                _gen.generator = copy.deepcopy(self.server_generator.generator).to(
                    device=self.device
                )
            self.use_inherited_generators = False
        else:
            self.client_generators = client_generators
            self.use_inherited_generators = True

        # init the optimizers.
        self.optimizer_server_generator = optim.Adam(
            self.server_generator.parameters(), lr=generator_learning_rate
        )
        self.optimizer_client_generators = [
            optim.Adam(_generator.parameters(), lr=generator_learning_rate)
            for _generator in self.client_generators
        ]
        self.optimizer_server_student = optim.Adam(
            self.server_student.parameters(), lr=student_learning_rate
        )
        self.optimizer_client_students = [
            optim.Adam(_student.parameters(), lr=student_learning_rate)
            for _student in self.client_students
        ]

        # init the training scheduler.
        self.weighted_server_teaching = weighted_server_teaching
        self.scheme_of_next_generator = scheme_of_next_generator
        self.use_client_model_scheduler = use_client_model_scheduler
        self.use_server_model_scheduler = use_server_model_scheduler
        self.use_server_generator_scheduler = use_server_generator_scheduler
        self.use_client_generator_scheduler = use_client_generator_scheduler
        self.scheduler_server_student = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_server_student,
            self.total_n_server_pseudo_batches,
            last_epoch=-1,
        )
        self.scheduler_client_students = [
            optim.lr_scheduler.CosineAnnealingLR(
                _optimizer_student, total_n_pseudo_batches, last_epoch=-1
            )
            for _optimizer_student in self.optimizer_client_students
        ]
        self.scheduler_server_generator = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_server_generator,
            self.total_n_server_pseudo_batches,
            last_epoch=-1,
        )
        self.scheduler_client_generators = [
            optim.lr_scheduler.CosineAnnealingLR(
                _optimizer_generator, total_n_pseudo_batches, last_epoch=-1
            )
            for _optimizer_generator in self.optimizer_client_generators
        ]

        # For distillation.
        self.AT_beta = AT_beta
        self.KL_temperature = KL_temperature

        # Set up & Resume
        self.log_fn = log_fn
        self.eval_batches_freq = eval_batches_freq
        self.early_stopping_batches = early_stopping_batches
        self.early_stopping_server_batches = early_stopping_server_batches
        self.validated_perfs = collections.defaultdict(list)
        print("\tFinished the initialization for ZeroShotKTSolver.")

    def alternative_clients_teaching_v3(self):
        client_trackers = dict(
            (
                _client_ind,
                RuntimeTracker(
                    metrics_to_track=["student_loss", "generator_loss"],
                    force_to_replace_metrics=True,
                ),
            )
            for _client_ind in range(self.numb_teachers)
        )
        client_best_trackers = dict(
            (_client_ind, BestPerf(best_perf=None, larger_is_better=True))
            for _client_ind in range(self.numb_teachers)
        )

        server_tracker = RuntimeTracker(
            metrics_to_track=["student_loss"], force_to_replace_metrics=True
        )
        server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)

        # store the best models according to the model index: client generators
        # are stored as the best_models[client_ind],
        # the server model is stored as best_models[self.numb_teachers]
        best_models = [None] * (self.numb_teachers + 1)

        # update each client student/generator pair
        if not self.use_inherited_generators:
            n_pseudo_batches = 0
            self.finished_clients = set()

            while n_pseudo_batches < self.total_n_pseudo_batches:
                # reset the noise.
                [x.reset_z(same_noise=self.same_noise) for x in self.client_generators]

                # update each client student/generator.
                for _client_ind in range(self.numb_teachers):
                    if _client_ind in self.finished_clients:
                        continue

                    # for local updates, each list contains only one element
                    _generator = [self.client_generators[_client_ind]]
                    _student = [self.client_students[_client_ind]]
                    _teacher = [self.client_teachers[_client_ind]]
                    _opt_generator = self.optimizer_client_generators[_client_ind]
                    _opt_student = self.optimizer_client_students[_client_ind]
                    _scheduler_generator = self.scheduler_client_generators[_client_ind]
                    _scheduler_student = self.scheduler_client_students[_client_ind]

                    # adv training the client generator and client student.
                    teacher_logits = None
                    for idx_pseudo in range(self.n_repeat_batch):
                        if idx_pseudo % self.n_repeat_batch < self.n_generator_iter:
                            generator_avg_loss = self.base_solver.update_generator(
                                same_noise=self.same_noise,
                                KT_loss_generator_fn=self.base_solver.KT_loss_generator,
                                _generator=_generator[0],
                                _teachers=_teacher,
                                _students=_student,
                                _opt_generator=_opt_generator,
                            )
                        # Take n_student_iter steps on student
                        elif idx_pseudo % self.n_repeat_batch < self.n_repeat_batch:
                            (
                                student_avg_loss,
                                teacher_logits,
                            ) = self.base_solver.update_student(
                                same_noise=self.same_noise,
                                KT_loss_student_fn=self.base_solver.KT_loss_student,
                                _student=_student[0],
                                _teachers=_teacher,
                                _generators=_generator,
                                _opt_student=_opt_student,
                                teacher_logits=teacher_logits,
                            )

                    # after each batch
                    if self.use_client_model_scheduler:
                        _scheduler_student.step()
                    if self.use_client_generator_scheduler:
                        _scheduler_generator.step()
                    client_trackers[_client_ind].update_metrics(
                        [student_avg_loss, generator_avg_loss],
                        n_samples=self.batch_size,
                    )
                    # eval on the validation dataset.
                    if (n_pseudo_batches + 1) % self.eval_batches_freq == 0:
                        validated_perf = self.validate(
                            model=_student[0], data_loader=self.val_data_loader
                        )
                        self.validated_perfs[_client_ind].append(validated_perf)
                        self.log_fn(
                            f"Batch {n_pseudo_batches + 1}/{self.total_n_pseudo_batches} for client-{_client_ind}: Generator Loss={client_trackers[_client_ind].stat['generator_loss'].avg:02.5f}; Student Loss={client_trackers[_client_ind].stat['student_loss'].avg:02.5f}; Student Validation Acc={validated_perf}."
                        )
                        client_trackers[_client_ind].reset()

                        # check early stopping.
                        if self.base_solver.check_early_stopping(
                            # we save the generator corresponding to the best perf.
                            model=_generator[0],
                            model_ind=_client_ind,
                            best_tracker=client_best_trackers[_client_ind],
                            validated_perf=validated_perf,
                            validated_perfs=self.validated_perfs,
                            perf_index=n_pseudo_batches + 1,
                            early_stopping_batches=self.early_stopping_batches,
                            log_fn=self.log_fn,
                            best_models=best_models,
                        ):
                            self.finished_clients.add(_client_ind)
                n_pseudo_batches += 1

            # recover the best client generators after the local training.
            self.client_generators = [
                x.to(self.device) for x in best_models[: self.numb_teachers]
            ]
        else:
            self.log_fn(f"Skip the trainnig for client generators.")

        # update the server generator/student
        n_pseudo_batches = 0
        while n_pseudo_batches < self.total_n_server_pseudo_batches:
            # update the noise in client_generators.
            [x.reset_z(same_noise=self.same_noise) for x in self.client_generators]

            # local update the server generator/student
            teacher_logits = None
            for _ in range(self.server_local_steps):
                stud_avg_loss, teacher_logits = self.base_solver.update_student(
                    same_noise=self.same_noise,
                    KT_loss_student_fn=self.base_solver.KT_loss_student,
                    _student=self.server_student,
                    _teachers=self.client_teachers,
                    _generators=self.client_generators,
                    _opt_student=self.optimizer_server_student,
                    teacher_logits=teacher_logits,
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
                    model_ind=self.numb_teachers,
                    best_tracker=server_best_tracker,
                    validated_perf=validated_perf,
                    validated_perfs=self.validated_perfs,
                    perf_index=n_pseudo_batches + 1,
                    early_stopping_batches=self.early_stopping_server_batches,
                    log_fn=self.log_fn,
                    best_models=best_models,
                ):
                    break
            n_pseudo_batches += 1

        # recover the best server model
        best_server_dict = best_models[self.numb_teachers].state_dict()
        self.server_student.load_state_dict(best_server_dict)
        self.server_student = self.server_student.cpu()

        # return the client generator
        if self.scheme_of_next_generator == "optimal_generator_based_on_teacher":
            # return the client generator whose client teacher achieves the highest val perf
            _perfs = []
            for _teacher in self.client_teachers:
                _perfs.append(
                    self.validate(model=_teacher, data_loader=self.val_data_loader)[
                        "top1"
                    ]
                )
            best_client_ind = np.argmax(_perfs)
            client_generator = self.client_generators[best_client_ind].cpu()
        else:
            # or random return one generator.
            client_generator = self.client_generators[
                random.randint(0, self.numb_teachers - 1)
            ].cpu()
        return copy.deepcopy(client_generator)

    @staticmethod
    def alternative_clients_teaching_v3_update_one_client(
        client_id,
        base_solver,
        batch_size,
        n_generator_iter,
        n_repeat_batch,
        total_n_pseudo_batches,
        same_noise,
        client_generator,
        client_student,
        client_teacher,
        opt_client_generator,
        opt_client_student,
        scheduler_client_generator,
        scheduler_client_student,
        use_client_model_scheduler,
        use_client_generator_scheduler,
        client_tracker,
        best_tracker,
        eval_batches_freq,
        log_fn,
        device,
        data_loader=None,
        criterion=None,
        metrics=None,
        validate_fn=None,
        early_stopping_batches=100,
    ):
        n_pseudo_batches = 0

        # place to the correct device.
        client_generator, client_student, client_teacher = (
            client_generator.to(device),
            client_student.to(device),
            client_teacher.to(device),
        )
        print(
            f"\tClient-{client_id} initialized the alternative_clients_teaching_v3_update_one_client."
        )

        # store the best model in the list. the size of the list will be 1 in this function.
        best_models = [None]

        while n_pseudo_batches < total_n_pseudo_batches:
            # reset noise
            client_generator.reset_z(same_noise=same_noise)

            # init the placehold for local updates.
            _generator = [client_generator]
            _student = [client_student]
            _teacher = [client_teacher]
            _scheduler_generator = scheduler_client_generator
            _scheduler_student = scheduler_client_student

            # adv training the client generator and client student.
            teacher_logits = None
            for idx_pseudo in range(n_repeat_batch):
                if idx_pseudo % n_repeat_batch < n_generator_iter:
                    generator_avg_loss = base_solver.update_generator(
                        same_noise=same_noise,
                        KT_loss_generator_fn=base_solver.KT_loss_generator,
                        _generator=_generator[0],
                        _teachers=_teacher,
                        _students=_student,
                        _opt_generator=opt_client_generator,
                    )
                # Take n_student_iter steps on student
                elif idx_pseudo % n_repeat_batch < n_repeat_batch:
                    student_avg_loss, teacher_logits = base_solver.update_student(
                        same_noise=same_noise,
                        KT_loss_student_fn=base_solver.KT_loss_student,
                        _student=_student[0],
                        _teachers=_teacher,
                        _generators=_generator,
                        _opt_student=opt_client_student,
                        teacher_logits=teacher_logits,
                    )

            # after each batch
            if use_client_model_scheduler:
                _scheduler_student.step()
            if use_client_generator_scheduler:
                _scheduler_generator.step()
            client_tracker.update_metrics(
                [student_avg_loss, generator_avg_loss], n_samples=batch_size
            )

            # eval on the validation dataset.
            if (n_pseudo_batches + 1) % eval_batches_freq == 0:
                if validate_fn is not None:
                    validated_perf = validate_fn(
                        model=_student[0],
                        data_loader=data_loader,
                        criterion=criterion,
                        metrics=metrics,
                        device=device,
                    )
                else:
                    validated_perf = None
                log_fn(
                    f"Batch {n_pseudo_batches + 1}/{total_n_pseudo_batches} for client-{client_id}: Generator Loss={client_tracker.stat['generator_loss'].avg:02.5f}; Student Loss={client_tracker.stat['student_loss'].avg:02.5f}; Student Validation Acc={validated_perf}."
                )
                client_tracker.reset()

                # check early stopping.
                if validated_perf is not None and base_solver.check_early_stopping(
                    # we save the generator corresponding to the best perf.
                    model=_generator[0],
                    # always put the best model on the first place.
                    model_ind=0,
                    best_tracker=best_tracker,
                    validated_perf=validated_perf,
                    validated_perfs=None,
                    perf_index=n_pseudo_batches + 1,
                    early_stopping_batches=early_stopping_batches,
                    best_models=best_models,
                ):
                    break
            n_pseudo_batches += 1

        # del object to free the memory.
        generator = copy.deepcopy(best_models[0].generator.cpu())
        del client_student, client_teacher, client_generator
        torch.cuda.empty_cache()
        return best_tracker, generator

    def alternative_clients_teaching_v3_parallel(self):
        client_trackers = dict(
            (
                _client_ind,
                RuntimeTracker(
                    metrics_to_track=["student_loss", "generator_loss"],
                    force_to_replace_metrics=True,
                ),
            )
            for _client_ind in range(self.numb_teachers)
        )
        client_best_trackers = dict(
            (_client_ind, BestPerf(best_perf=None, larger_is_better=True))
            for _client_ind in range(self.numb_teachers)
        )
        server_tracker = RuntimeTracker(
            metrics_to_track=["student_loss"], force_to_replace_metrics=True
        )
        server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)

        # update each client student/generator pair
        if not self.use_inherited_generators:
            tmp.set_start_method("spawn", force=True)
            tmp.set_sharing_strategy("file_system")

            # init the local computation objects.
            dummys = []
            for _c in range(self.numb_teachers):
                # place them to cpu to save the space.
                self.client_generators[_c] = self.client_generators[_c].cpu()
                self.client_students[_c] = self.client_students[_c].cpu()
                self.client_teachers[_c] = self.client_teachers[_c].cpu()

                # add to dummys for the parallel computation.
                dummys.append(
                    Dummy(
                        run_fn=self.alternative_clients_teaching_v3_update_one_client,
                        client_id=_c,
                        base_solver=self.base_solver,
                        batch_size=self.batch_size,
                        n_generator_iter=self.n_generator_iter,
                        n_repeat_batch=self.n_repeat_batch,
                        total_n_pseudo_batches=self.total_n_pseudo_batches,
                        same_noise=self.same_noise,
                        client_generator=self.client_generators[_c],
                        client_student=self.client_students[_c],
                        client_teacher=self.client_teachers[_c],
                        opt_client_generator=self.optimizer_client_generators[_c],
                        opt_client_student=self.optimizer_client_students[_c],
                        scheduler_client_generator=self.scheduler_client_generators[_c],
                        scheduler_client_student=self.scheduler_client_students[_c],
                        use_client_model_scheduler=self.use_client_model_scheduler,
                        use_client_generator_scheduler=self.use_client_generator_scheduler,
                        client_tracker=client_trackers[_c],
                        best_tracker=client_best_trackers[_c],
                        eval_batches_freq=self.eval_batches_freq,
                        log_fn=self.log_fn,
                        device=self.device,
                        data_loader=self.val_data_loader,
                        criterion=self.criterion,
                        metrics=self.metrics,
                        validate_fn=self.validate,
                        early_stopping_batches=self.early_stopping_batches,
                    )
                )

            with closing(tmp.Pool(processes=int(self.n_parallel_comp))) as pool:
                parallel_outputs = pool.map(run_steps, dummys)
            torch.cuda.empty_cache()
            self.log_fn("Finished local client/generator training.")

            # load the info/model from the parallel processes.
            best_local_training_perfs = []
            for c_id, (best_tracker, client_generator) in zip(
                range(self.numb_teachers), parallel_outputs
            ):
                self.client_generators[c_id].generator = client_generator.to(
                    self.device
                )
                self.client_teachers[c_id] = self.client_teachers[c_id].to(self.device)
                best_local_training_perfs.append(best_tracker.best_perf)
            # get the client weights by using the local training performance.
            sumed_perfs = sum(best_local_training_perfs)
            client_weights = [x / sumed_perfs for x in best_local_training_perfs]

        # update the server generator/student
        n_pseudo_batches = 0
        # store the best model in the list. the size of the list will be 1 in this function.
        best_models = [None]
        while n_pseudo_batches < self.total_n_server_pseudo_batches:
            # update the noise in client_generators.
            [x.reset_z(same_noise=self.same_noise) for x in self.client_generators]

            # local update the server generator/student
            teacher_logits = None
            for _ in range(self.server_local_steps):
                stud_avg_loss, teacher_logits = self.base_solver.update_student(
                    same_noise=self.same_noise,
                    KT_loss_student_fn=self.base_solver.KT_loss_student,
                    _student=self.server_student,
                    _teachers=self.client_teachers,
                    _generators=self.client_generators,
                    _opt_student=self.optimizer_server_student,
                    teacher_logits=teacher_logits,
                    weights=client_weights if self.weighted_server_teaching else None,
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
                    # always put the best model on the first place.
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
        best_server_dict = best_models[0].state_dict()
        self.server_student.load_state_dict(best_server_dict)
        self.server_student = self.server_student.cpu()

        # return the client generator
        if self.scheme_of_next_generator == "optimal_generator_based_on_teacher":
            # return the client generator whose client teacher achieves the highest val perf
            _perfs = []
            for _teacher in self.client_teachers:
                _perfs.append(
                    self.validate(model=_teacher, data_loader=self.val_data_loader)[
                        "top1"
                    ]
                )
            best_client_ind = np.argmax(_perfs)
            client_generator = self.client_generators[best_client_ind].cpu()
        else:
            # or random return one generator.
            client_generator = self.client_generators[
                random.randint(0, self.numb_teachers - 1)
            ].cpu()
        return copy.deepcopy(client_generator)

    def alternative_clients_teaching_v4_parallel(self):
        client_trackers = dict(
            (
                _client_ind,
                RuntimeTracker(
                    metrics_to_track=["student_loss", "generator_loss"],
                    force_to_replace_metrics=True,
                ),
            )
            for _client_ind in range(self.numb_teachers)
        )
        client_best_trackers = dict(
            (_client_ind, BestPerf(best_perf=None, larger_is_better=True))
            for _client_ind in range(self.numb_teachers)
        )
        server_tracker = RuntimeTracker(
            metrics_to_track=["student_loss", "generator_loss"],
            force_to_replace_metrics=True,
        )
        server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)

        # update each client student/generator pair
        if not self.use_inherited_generators:
            tmp.set_start_method("spawn", force=True)
            tmp.set_sharing_strategy("file_system")

            # init the local computation objects.
            dummys = []
            for _c in range(self.numb_teachers):
                # place them to cpu to save the space.
                self.client_generators[_c] = self.client_generators[_c].cpu()
                self.client_students[_c] = self.client_students[_c].cpu()
                self.client_teachers[_c] = self.client_teachers[_c].cpu()

                # add to dummys for the parallel computation.
                dummys.append(
                    Dummy(
                        run_fn=self.alternative_clients_teaching_v3_update_one_client,
                        client_id=_c,
                        base_solver=self.base_solver,
                        batch_size=self.batch_size,
                        n_generator_iter=self.n_generator_iter,
                        n_repeat_batch=self.n_repeat_batch,
                        total_n_pseudo_batches=self.total_n_pseudo_batches,
                        same_noise=self.same_noise,
                        client_generator=self.client_generators[_c],
                        client_student=self.client_students[_c],
                        client_teacher=self.client_teachers[_c],
                        opt_client_generator=self.optimizer_client_generators[_c],
                        opt_client_student=self.optimizer_client_students[_c],
                        scheduler_client_generator=self.scheduler_client_generators[_c],
                        scheduler_client_student=self.scheduler_client_students[_c],
                        use_client_model_scheduler=self.use_client_model_scheduler,
                        use_client_generator_scheduler=self.use_client_generator_scheduler,
                        client_tracker=client_trackers[_c],
                        best_tracker=client_best_trackers[_c],
                        eval_batches_freq=self.eval_batches_freq,
                        log_fn=self.log_fn,
                        device=self.device,
                        data_loader=self.val_data_loader,
                        criterion=self.criterion,
                        metrics=self.metrics,
                        validate_fn=self.validate,
                        early_stopping_batches=self.early_stopping_batches,
                    )
                )

            with closing(tmp.Pool(processes=int(self.n_parallel_comp))) as pool:
                parallel_outputs = pool.map(run_steps, dummys)
            torch.cuda.empty_cache()
            self.log_fn("Finished local client/generator training.")

            # load the info/model from the parallel processes.
            best_local_training_perfs = []
            for c_id, (best_tracker, client_generator) in zip(
                range(self.numb_teachers), parallel_outputs
            ):
                self.client_generators[c_id].generator = client_generator.to(
                    self.device
                )
                self.client_teachers[c_id] = self.client_teachers[c_id].to(self.device)
                best_local_training_perfs.append(best_tracker.best_perf)
            # get the client weights by using the local training performance.
            sumed_perfs = sum(best_local_training_perfs)
            client_weights = [x / sumed_perfs for x in best_local_training_perfs]

        # update the server generator/student
        n_pseudo_batches = 0
        # store the best model in the list. the size of the list will be 1 in this function.
        best_models = [None]
        while n_pseudo_batches < self.total_n_server_pseudo_batches:
            # update the noise in client_generators.
            [x.reset_z(same_noise=self.same_noise) for x in self.client_generators]

            # local update the server generator/student
            teacher_logits = None
            for idx_pseudo in range(self.n_repeat_batch):
                if idx_pseudo % self.n_repeat_batch < self.n_generator_iter:
                    # update each client generator.
                    gen_losses = []
                    for _client_ind in range(self.numb_teachers):
                        # for local updates, each list contains only one element
                        _student = [self.server_student]
                        _generator = [self.client_generators[_client_ind]]
                        _teacher = [self.client_teachers[_client_ind]]
                        _opt_generator = self.optimizer_client_generators[_client_ind]

                        # adv training the client generator and client student.
                        generator_avg_loss = self.base_solver.update_generator(
                            same_noise=self.same_noise,
                            KT_loss_generator_fn=self.base_solver.KT_loss_generator,
                            _generator=_generator[0],
                            _teachers=_teacher,
                            _students=_student,
                            _opt_generator=_opt_generator,
                        )
                        gen_losses.append(generator_avg_loss)
                # update the server student
                elif idx_pseudo % self.n_repeat_batch < self.n_repeat_batch:
                    stud_avg_loss, teacher_logits = self.base_solver.update_student(
                        same_noise=self.same_noise,
                        KT_loss_student_fn=self.base_solver.KT_loss_student,
                        _student=self.server_student,
                        _teachers=self.client_teachers,
                        _generators=self.client_generators,
                        _opt_student=self.optimizer_server_student,
                        teacher_logits=teacher_logits,
                    )
            # after each batch.
            if self.use_server_model_scheduler:
                self.scheduler_server_student.step()
            if self.use_client_generator_scheduler:
                for _scheduler in self.scheduler_client_generators:
                    _scheduler.step()

            # update the tracker after each batch.
            gen_avg_loss = sum(gen_losses) / len(gen_losses)
            server_tracker.update_metrics(
                [stud_avg_loss, gen_avg_loss], n_samples=self.batch_size
            )
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
                    # always put the best model on the first place.
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
        best_server_dict = best_models[0].state_dict()
        self.server_student.load_state_dict(best_server_dict)
        self.server_student = self.server_student.cpu()

        # return the client generator
        if self.scheme_of_next_generator == "optimal_generator_based_on_teacher":
            # return the client generator whose client teacher achieves the highest val perf
            _perfs = []
            for _teacher in self.client_teachers:
                _perfs.append(
                    self.validate(model=_teacher, data_loader=self.val_data_loader)[
                        "top1"
                    ]
                )
            best_client_ind = np.argmax(_perfs)
            client_generator = self.client_generators[best_client_ind].cpu()
        else:
            # or random return one generator.
            client_generator = self.client_generators[
                random.randint(0, self.numb_teachers - 1)
            ].cpu()
        return copy.deepcopy(client_generator)

    def joint_teaching_v2(self):
        # init the tracker.
        server_tracker = RuntimeTracker(
            metrics_to_track=["student_loss", "generator_loss"],
            force_to_replace_metrics=True,
        )
        server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)

        # update the server generator/student
        n_pseudo_batches = 0
        best_models = [None]
        while n_pseudo_batches < self.total_n_server_pseudo_batches:
            # reset the noise.
            [x.reset_z(same_noise=self.same_noise) for x in self.client_generators]

            # update client generators and the server student
            teacher_logits = None
            for idx_pseudo in range(self.n_repeat_batch):
                if idx_pseudo % self.n_repeat_batch < self.n_generator_iter:
                    # update each client generator.
                    gen_losses = []
                    for _client_ind in range(self.numb_teachers):
                        # for local updates, each list contains only one element
                        _student = [self.server_student]
                        _generator = [self.client_generators[_client_ind]]
                        _teacher = [self.client_teachers[_client_ind]]
                        _opt_generator = self.optimizer_client_generators[_client_ind]

                        # adv training the client generator and client student.
                        teacher_logits = None
                        generator_avg_loss = self.base_solver.update_generator(
                            same_noise=self.same_noise,
                            KT_loss_generator_fn=self.base_solver.KT_loss_generator,
                            _generator=_generator[0],
                            _teachers=_teacher,
                            _students=_student,
                            _opt_generator=_opt_generator,
                        )
                        gen_losses.append(generator_avg_loss)

                # update the server student
                elif idx_pseudo % self.n_repeat_batch < self.n_repeat_batch:
                    stud_avg_loss, teacher_logits = self.base_solver.update_student(
                        same_noise=self.same_noise,
                        KT_loss_student_fn=self.base_solver.KT_loss_student,
                        _student=self.server_student,
                        _teachers=self.client_teachers,
                        _generators=self.client_generators,
                        _opt_student=self.optimizer_server_student,
                        teacher_logits=teacher_logits,
                    )
            # after each batch.
            if self.use_server_model_scheduler:
                self.scheduler_server_student.step()
            if self.use_client_generator_scheduler:
                for _scheduler in self.scheduler_client_generators:
                    _scheduler.step()

            # update the tracker after each batch.
            gen_avg_loss = sum(gen_losses) / len(gen_losses)
            server_tracker.update_metrics(
                [stud_avg_loss, gen_avg_loss], n_samples=self.batch_size
            )
            if (n_pseudo_batches + 1) % self.eval_batches_freq == 0:
                validated_perf = self.validate(
                    model=self.server_student, data_loader=self.val_data_loader
                )
                self.log_fn(
                    f"Batch {n_pseudo_batches + 1}/{self.total_n_server_pseudo_batches}: Generator Loss={server_tracker.stat['generator_loss'].avg:02.5f}; Student Loss={server_tracker.stat['student_loss'].avg:02.5f}; Student Validation Acc={validated_perf}."
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

        # return the client generator.
        client_generator = self.client_generators[
            random.randint(0, self.numb_teachers - 1)
        ].cpu()
        self.server_student = self.server_student.cpu()
        return copy.deepcopy(client_generator)

    def ensemble(self):
        # init the tracker.
        server_tracker = RuntimeTracker(
            metrics_to_track=["student_loss", "generator_loss"],
            force_to_replace_metrics=True,
        )
        server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)

        # update the server generator/student
        n_pseudo_batches = 0
        best_models = [None]
        while n_pseudo_batches < self.total_n_server_pseudo_batches:
            # reset the noise.
            self.server_generator.reset_z(same_noise=True)

            # update the server generator
            teacher_logit = None
            teacher_activations = None
            for idx_pseudo in range(self.n_repeat_batch):
                if idx_pseudo % self.n_repeat_batch < self.n_generator_iter:
                    generator_loss = self.base_solver.update_generator_ensemble(
                        KT_loss_generator_fn=self.base_solver.KT_loss_generator,
                        server_generator=self.server_generator,
                        local_teachers=self.client_teachers,
                        server_student=self.server_student,
                        opt_server_generator=self.optimizer_server_generator,
                    )

                # update the server student
                elif idx_pseudo % self.n_repeat_batch < self.n_repeat_batch:
                    (
                        stud_avg_loss,
                        teacher_logit,
                        teacher_activations,
                    ) = self.base_solver.update_student_ensemble(
                        KT_loss_student_fn=self.base_solver.KT_loss_student,
                        server_student=self.server_student,
                        local_teachers=self.client_teachers,
                        server_generator=self.server_generator,
                        opt_server_student=self.optimizer_server_student,
                        teacher_logit=teacher_logit,
                        teacher_activations=teacher_activations,
                    )
            # after each batch.
            if self.use_server_model_scheduler:
                self.scheduler_server_student.step()
            if self.use_server_generator_scheduler:
                self.scheduler_server_generator.step()

            # update the tracker after each batch.
            server_tracker.update_metrics(
                [stud_avg_loss, generator_loss], n_samples=self.batch_size
            )
            if (n_pseudo_batches + 1) % self.eval_batches_freq == 0:
                validated_perf = self.validate(
                    model=self.server_student, data_loader=self.val_data_loader
                )
                self.log_fn(
                    f"Batch {n_pseudo_batches + 1}/{self.total_n_server_pseudo_batches}: Generator Loss={server_tracker.stat['generator_loss'].avg:02.5f}; Student Loss={server_tracker.stat['student_loss'].avg:02.5f}; Student Validation Acc={validated_perf}."
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
        best_server_dict = best_models[0].state_dict()
        self.server_student.load_state_dict(best_server_dict)
        self.server_student = self.server_student.cpu()

        # return the server generator.
        return copy.deepcopy(self.server_generator.cpu())

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
            )
            model = model.to(self.device if device is None else device)
            model.train()
            return val_perf


class BaseKTSolver(object):
    def __init__(self, KL_temperature, AT_beta):
        # general init.
        self.KL_temperature = KL_temperature
        self.AT_beta = AT_beta

    @staticmethod
    def update_generator(
        same_noise,
        KT_loss_generator_fn,
        _generator,
        _opt_generator,
        _teachers,
        _students,
    ):
        """
        Update a single generator, possibly a server generator or a client generator.
        When updating a client generator, _teachers and _students contain only one element.
        When updating a server generator, _teachers and _students contain n elements.

        In the case of joint teaching:
        We update the server generator, where _teachers contain n elements,
        and _students contain only one elements.
        """
        # Note after every batch size of iterations, a new set of noise is generated.
        x_pseudos = _generator.generate(same_noise=same_noise)
        student_logits = [_student(_x) for _x, _student in zip(x_pseudos, _students)]
        teacher_logits = [_teacher(_x) for _x, _teacher in zip(x_pseudos, _teachers)]

        # In the joint teaching case
        if len(student_logits) < len(teacher_logits) and len(student_logits) == 1:
            student_logits = student_logits * len(teacher_logits)

        generator_losses = [
            KT_loss_generator_fn(_student_logits, _teacher_logits)
            for _student_logits, _teacher_logits in zip(student_logits, teacher_logits)
        ]

        _opt_generator.zero_grad()
        generator_avg_loss = sum(generator_losses) / len(generator_losses)
        generator_avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(_generator.parameters(), 5)
        _opt_generator.step()
        return generator_avg_loss.item()

    @staticmethod
    def update_student(
        same_noise,
        KT_loss_student_fn,
        _student,
        _opt_student,
        _teachers,
        _generators,
        teacher_logits=None,
        weights=None,
    ):
        """
        Update a single student, possibly a server student or a client student.
        When updating a client student, _teachers and _generators contain only one element.
        When updating a server student, _teachers and _generators contain n elements.

        In the joint teaching mode:
        We update the server student, where _teachers contain n element, and _generators
        contain either n or one element.
        """
        # data from all generators
        x_pseudos = [_gen.generate(same_noise=same_noise)[0] for _gen in _generators]

        if teacher_logits is None:
            with torch.no_grad():
                teacher_logits = [
                    _teacher(_x) for _x, _teacher in zip(x_pseudos, _teachers)
                ]
            # extend the list in the joint teaching model v1
            if len(_teachers) > len(_generators) and len(_generators) == 1:
                teacher_logits = teacher_logits * len(_teachers)

        student_logits_activations = [
            (_student(_x), _student.activations) for _x in x_pseudos
        ]
        student_losses = [
            KT_loss_student_fn(
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
        _opt_student.zero_grad()
        weights = (
            weights
            if weights is not None
            else [1.0 / len(student_losses)] * len(student_losses)
        )
        student_avg_loss = sum(
            [loss * weight for loss, weight in zip(student_losses, weights)]
        )
        student_avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(_student.parameters(), 5)
        _opt_student.step()
        return student_avg_loss.item(), teacher_logits

    @staticmethod
    def update_generator_ensemble(
        KT_loss_generator_fn,
        server_generator,
        opt_server_generator,
        local_teachers,
        server_student,
    ):
        """
        Update a single server generator in ensemble training case.
        args:
        local_teachers: list, n local client teachers
        server_generator: produces noise of size 1
        """
        # Note after every batch size of iterations, a new set of noise is generated.
        x_pseudo = server_generator.generate(same_noise=True)[0]
        student_logit = server_student(x_pseudo)
        teacher_logit = sum([teacher(x_pseudo) for teacher in local_teachers]) / len(
            local_teachers
        )

        generator_loss = KT_loss_generator_fn(student_logit, teacher_logit)

        opt_server_generator.zero_grad()
        generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(server_generator.parameters(), 5)
        opt_server_generator.step()
        return generator_loss.item()

    @staticmethod
    def update_student_ensemble(
        KT_loss_student_fn,
        server_student,
        opt_server_student,
        local_teachers,
        server_generator,
        teacher_logit=None,
        teacher_activations=None,
    ):
        """
        Update a single server student in ensemble training case.
        args:
        local_teachers: list, n local client teachers
        server_generator: produces noise of size 1
        """
        # data from all generators
        x_pseudo = server_generator.generate(same_noise=True)[0]

        # get logit.
        if teacher_logit is None:
            with torch.no_grad():
                teacher_logit = sum(
                    [teacher(x_pseudo) for teacher in local_teachers]
                ) / len(local_teachers)
        student_logit = server_student(x_pseudo)

        # evaluate the activations.
        if teacher_activations is None:
            teacher_activations = [
                sum([teacher.activations[idx] for teacher in local_teachers])
                / len(local_teachers)
                for idx in range(len(local_teachers[0].activations))
            ]

        student_loss = KT_loss_student_fn(
            student_logit,
            server_student.activations,
            teacher_logit,
            teacher_activations,
        )

        opt_server_student.zero_grad()
        student_loss.backward()
        torch.nn.utils.clip_grad_norm_(server_student.parameters(), 5)
        opt_server_student.step()
        return student_loss.item(), teacher_logit, teacher_activations

    def attention(self, x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def attention_diff(self, x, y):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        :param y = activations
        """
        return (self.attention(x) - self.attention(y)).pow(2).mean()

    def divergence(self, student_logits, teacher, use_teacher_logits=True):
        divergence = F.kl_div(
            F.log_softmax(student_logits / self.KL_temperature, dim=1),
            F.softmax(teacher / self.KL_temperature, dim=1)
            if use_teacher_logits
            else teacher,
            reduction="batchmean",
        )  # forward KL
        return divergence

    def KT_loss_generator(self, student_logits, teacher_logits):
        divergence_loss = self.divergence(student_logits, teacher_logits)
        total_loss = -divergence_loss
        return total_loss

    def KT_loss_student(
        self, student_logits, student_activations, teacher_logits, teacher_activations
    ):
        divergence_loss = self.divergence(student_logits, teacher_logits)
        if self.AT_beta > 0:
            at_loss = 0
            for i in range(len(student_activations)):
                at_loss = at_loss + self.AT_beta * self.attention_diff(
                    student_activations[i], teacher_activations[i]
                )
        else:
            at_loss = 0

        total_loss = divergence_loss + at_loss
        return total_loss

    def check_early_stopping(
        self,
        model,
        model_ind,
        best_tracker,
        validated_perf,
        validated_perfs,
        perf_index,
        early_stopping_batches,
        log_fn=print,
        best_models=None,
    ):
        # update the tracker.
        best_tracker.update(perf=validated_perf["top1"], perf_location=perf_index)
        if validated_perfs is not None:
            validated_perfs[model_ind].append(validated_perf)

        # save the best model.
        if best_tracker.is_best and best_models is not None:
            best_models[model_ind] = copy.deepcopy(model).cpu()

        # check if we need the early stopping or not.
        if perf_index - best_tracker.get_best_perf_loc >= early_stopping_batches:
            log_fn(
                f"\tMeet the early stopping condition (batches={early_stopping_batches}): early stop!! (perf_index={perf_index}, best_perf_loc={best_tracker.get_best_perf_loc})."
            )
            return True
        else:
            return False

    def prepare_model(self, conf, model, device, _is_teacher):
        model = model.to(device)
        model.save_activations = True
        if _is_teacher:
            model = agg_utils.modify_model_trainable_status(
                conf, model, trainable=False
            )
            model.eval()
        else:
            model = copy.deepcopy(model)
            model = agg_utils.modify_model_trainable_status(conf, model, trainable=True)
            model.train()
        return model


"""dummy class for multiprocessing."""


class Dummy(object):
    def __init__(self, run_fn, **kwargs):
        # init.
        self.run_fn = run_fn
        self.kwargs = kwargs

    def run(self):
        return self.run_fn(**self.kwargs)


def run_steps(agent):
    return agent.run()
