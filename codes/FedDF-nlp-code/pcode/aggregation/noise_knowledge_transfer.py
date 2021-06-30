# -*- coding: utf-8 -*-
import copy
import collections

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.stat_tracker import RuntimeTracker, BestPerf
from utils.tensor_buffer import TensorBuffer
import pcode.validation as validation
import pcode.create_dataset as create_dataset


def aggregate(
    conf,
    master_model,
    fedavg_model,
    client_models,
    criterion,
    metrics,
    fa_val_perf,
    distillation_data_loader,
    val_data_loader,
    test_data_loader,
):
    fl_aggregate = conf.fl_aggregate

    for client_model in client_models:
        for param in client_model.parameters():
            param.requires_grad = False

    # evaluate the ensemble of local models on the test_loader
    if "eval_ensemble" in fl_aggregate and fl_aggregate["eval_ensemble"]:
        conf.perfs["ensemble"] = validation.ensembled_validate(
            conf,
            models=client_models,
            data_loader=test_data_loader,
            ensemble_scheme=None
            if "update_student_scheme" not in fl_aggregate
            else fl_aggregate["update_student_scheme"],
        )

    # distillation. "right now the code only supports one arch."
    conf.logger.log(
        f"Master: we have {len(client_models)} local models for noise distillation."
    )
    kt = NoiseKTSolver(
        conf=conf,
        teacher_models=client_models,
        student_model=fedavg_model,
        criterion=criterion,
        metrics=metrics,
        batch_size=64
        if "batch_size" not in fl_aggregate
        else int(fl_aggregate["batch_size"]),
        total_n_server_pseudo_batches=1000 * 10
        if "total_n_server_pseudo_batches" not in fl_aggregate
        else int(fl_aggregate["total_n_server_pseudo_batches"]),
        server_local_steps=1
        if "server_local_steps" not in fl_aggregate
        else int(fl_aggregate["server_local_steps"]),
        val_data_loader=val_data_loader,
        distillation_data_loader=distillation_data_loader,
        use_server_model_scheduler=False
        if "use_server_model_scheduler" not in fl_aggregate
        else fl_aggregate["use_server_model_scheduler"],
        same_noise=True
        if "same_noise" not in fl_aggregate
        else fl_aggregate["same_noise"],
        student_learning_rate=1e-5
        if "student_learning_rate" not in fl_aggregate
        else fl_aggregate["student_learning_rate"],
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
        update_student_scheme="avg_logits"
        if "update_student_scheme" not in fl_aggregate
        else fl_aggregate["update_student_scheme"],
        weighted_server_teaching=False
        if "weighted_server_teaching" not in fl_aggregate
        else fl_aggregate["weighted_server_teaching"],
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

    _master_model = kt.server_student
    if (
        hasattr(fl_aggregate, "server_momentum_factor")
        and 1 > fl_aggregate["server_momentum_factor"] > 0
    ):
        # start the server momentum acceleration.
        previous_model_tb = TensorBuffer(list(master_model.parameters()))
        current_model_tb = TensorBuffer(list(_master_model.parameters()))

        # get the update direction.
        update = previous_model_tb.buffer - current_model_tb.buffer

        # using server momentum for the update.
        if not hasattr(conf, "server_momentum_buffer"):
            conf.server_momentum_buffer = torch.zeros_like(update)
        conf.server_momentum_buffer.mul_(fl_aggregate["server_momentum_factor"]).add_(
            update
        )
        previous_model_tb.buffer.add_(-conf.server_momentum_buffer)

        # update 'master_model' in place.
        _model_param = list(master_model.parameters())
        previous_model_tb.unpack(_model_param)
    else:
        master_model = _master_model

    # free the memory.
    del client_models, kt
    torch.cuda.empty_cache()

    return master_model


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
        distillation_data_loader=None,
        use_server_model_scheduler=False,
        same_noise=True,
        student_learning_rate=1e-5,
        KL_temperature=1,
        log_fn=print,
        eval_batches_freq=100,
        early_stopping_server_batches=1000,
        update_student_scheme="avg_logits",  # either avg_losses or avg_logits
        weighted_server_teaching=False,
        return_best_model_on_val=False,
    ):
        # general init.
        self.conf = conf
        self.device = torch.device("cuda")

        # init the validation criterion and metrics.
        self.criterion = criterion
        self.metrics = metrics

        # init training logics and loaders.
        self.same_noise = same_noise
        self.batch_size = batch_size
        self.total_n_server_pseudo_batches = total_n_server_pseudo_batches
        self.server_local_steps = server_local_steps

        # init the fundamental solver.
        self.base_solver = BaseKTSolver(KL_temperature=KL_temperature)

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
        # self.return_best_model_on_val = return_best_model_on_val
        # self.init_server_student = copy.deepcopy(self.server_student)

        # init the loaders.
        self.val_data_loader = val_data_loader
        self.distillation_data_loader = distillation_data_loader

        # init the optimizers.
        self.optimizer_server_student = optim.Adam(
            self.server_student.parameters(), lr=student_learning_rate
        )

        # init the training scheduler.
        self.weighted_server_teaching = weighted_server_teaching
        self.use_server_model_scheduler = use_server_model_scheduler
        self.scheduler_server_student = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_server_student,
            self.total_n_server_pseudo_batches,
            last_epoch=-1,
        )

        # For distillation.
        self.KL_temperature = KL_temperature

        # Set up & Resume
        self.log_fn = log_fn
        self.eval_batches_freq = eval_batches_freq
        self.early_stopping_server_batches = early_stopping_server_batches
        self.update_student_scheme = update_student_scheme
        print("\tFinished the initialization for NoiseKTSolver.")

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
        data_iter = iter(self.distillation_data_loader)

        # iterate over dataset
        while n_pseudo_batches < self.total_n_server_pseudo_batches:
            # get the inputs.
            if self.distillation_data_loader is not None:
                try:
                    pseudo_data = create_dataset.seqcls_batch_to_device(
                        next(data_iter)
                    )[2]
                except StopIteration:
                    data_iter = iter(self.distillation_data_loader)
                    pseudo_data = create_dataset.seqcls_batch_to_device(
                        next(data_iter)
                    )[2]
            else:
                raise NotImplementedError(
                    "The distillation data loader is not accessible."
                )

            # get the logits.
            with torch.no_grad():
                teacher_logits = [
                    (_teacher(**pseudo_data))[0] for _teacher in self.client_teachers
                ]

            # steps on the same pseudo data
            for _ in range(self.server_local_steps):
                student_logits, *_ = self.server_student(**pseudo_data)
                student_logits = [student_logits] * self.numb_teachers

                stud_avg_loss = self.update_student(
                    student_logits=student_logits,
                    base_solver=self.base_solver,
                    _student=self.server_student,
                    _teachers=self.client_teachers,
                    _opt_student=self.optimizer_server_student,
                    teacher_logits=teacher_logits,
                    update_student_scheme=self.update_student_scheme,
                )

            # after each batch.
            if self.use_server_model_scheduler:
                self.scheduler_server_student.step()

            # update the tracker after each batch.
            server_tracker.update_metrics([stud_avg_loss], n_samples=self.batch_size)

            if (n_pseudo_batches + 1) % self.eval_batches_freq == 0:
                validated_perf, _ = validation.evaluate(
                    self.conf,
                    self.server_student,
                    self.val_data_loader,
                    self.criterion,
                    back_to_cpu=False,
                    label="server_student_on_val_data_loader",
                    save_jason=False,
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
                    perf_index=n_pseudo_batches + 1,
                    early_stopping_batches=self.early_stopping_server_batches,
                    best_models=best_models,
                ):
                    break
            n_pseudo_batches += 1

        # # recover the best server model
        # use_init_server_model = False
        # if self.return_best_model_on_val:
        #     use_init_server_model = (
        #         True
        #         if init_perf_on_val["top1"] > server_best_tracker.best_perf
        #         else False
        #     )

        # # get the server model.
        # if use_init_server_model:
        #     self.log_fn("use init server model instead.")
        #     best_server_dict = self.init_server_student.state_dict()
        # else:
        best_server_dict = best_models[0].state_dict()

        # update the server model.
        self.server_student.load_state_dict(best_server_dict)
        self.server_student = self.server_student.cpu()

    @staticmethod
    def update_student(
        student_logits,
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
                base_solver.KT_loss_student(_student_logits, _teacher_logits)
                for _student_logits, _teacher_logits in zip(
                    student_logits, teacher_logits
                )
            ]
            student_avg_loss = sum(
                [loss * weight for loss, weight in zip(student_losses, weights)]
            )
        elif update_student_scheme == "avg_logits":
            student_logits = student_logits[0]
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
            student_logits = student_logits[0]
            teacher_avg_probs = sum(
                [
                    F.softmax(teacher_logit) * weight
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


class BaseKTSolver(object):
    def __init__(self, KL_temperature):
        # general init.
        self.KL_temperature = KL_temperature

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

    def KT_loss_student(self, student_logits, teacher_logits):
        divergence_loss = self.divergence(student_logits, teacher_logits)

        return divergence_loss

    def check_early_stopping(
        self,
        model,
        model_ind,
        best_tracker,
        validated_perf,
        perf_index,
        early_stopping_batches,
        log_fn=print,
        best_models=None,
    ):
        # update the tracker.
        best_tracker.update(perf=validated_perf, perf_location=perf_index)

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
        if _is_teacher:
            model = modify_model_trainable_status(conf, model, trainable=False)
            model.eval()
        else:
            model = copy.deepcopy(model)
            model = modify_model_trainable_status(conf, model, trainable=True)
            model.train()
        return model


def modify_model_trainable_status(conf, model, trainable):
    for _, _param in model.named_parameters():
        _param.requires_grad = trainable

    model = model.cuda()
    if len(conf.world) > 1:
        model = torch.nn.DataParallel(model, device_ids=conf.world)
    return model
