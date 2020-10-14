# -*- coding: utf-8 -*-
import warnings
import weakref
from functools import wraps
from collections import Counter
from bisect import bisect_right

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pcode.utils.auxiliary as auxiliary


class Scheduler(object):
    def __init__(self, conf, optimizer, display_status=True):
        # init
        self.conf = conf
        self.optimizer = optimizer
        self.display_status = display_status
        self.local_index = 0
        self._update_training_progress()
        self.init_learning_rate()
        self.init_lr_scheduler()

    def init_learning_rate(self):
        # determine the learning_rate_per_samples.
        self.lr_scaleup_init_lr = (
            self.conf.lr_scaleup_init_lr
            if self.conf.lr_scaleup_init_lr is not None
            else self.conf.lr
        )
        self.conf.base_batch_size = (
            self.conf.base_batch_size
            if self.conf.base_batch_size is not None
            else self.conf.batch_size
        )
        self.learning_rate_per_samples = self.conf.lr / self.conf.base_batch_size
        self.learning_rate_ = self.learning_rate_per_samples * self.conf.batch_size

        # if scaleup.
        if self.conf.lr_scaleup:
            if self.conf.lr_scaleup_factor is None:
                self.lr_scaleup_factor = self.conf.graph.n_nodes
            else:
                if auxiliary.is_float(self.conf.lr_scaleup_factor):
                    self.lr_scaleup_factor = float(self.conf.lr_scaleup_factor)
                else:
                    if self.conf.lr_scaleup_factor == "graph":
                        self.lr_scaleup_factor = self.conf.graph.scaling
                    elif self.conf.lr_scaleup_factor == "world":
                        self.lr_scaleup_factor = self.conf.graph.n_nodes
                    else:
                        raise NotImplementedError

            self.learning_rate = self.learning_rate_ * self.lr_scaleup_factor
        else:
            self.learning_rate = self.learning_rate_

        # overwrite lr_scaleup_factor.
        self.lr_scaleup_factor = self.learning_rate / self.lr_scaleup_init_lr
        self.is_scaledup = True if self.lr_scaleup_factor != 1 else False

        # if warmup.
        if self.conf.lr_warmup_epochs is None:
            self.conf.lr_warmup_epochs = min(
                self.conf.lr_scaleup_factor, self.conf.lr_warmup_epochs_upper_bound
            )

        # check the warmup status.
        self.is_warmuped = (
            True if self.conf.lr_scaleup_factor != 1 and self.conf.lr_warmup else False
        )

        # update the lr for the optimizer.
        if self.is_warmuped:
            self.update_lr(self.lr_scaleup_init_lr)
        elif self.is_scaledup:
            self.update_lr(self.learning_rate)

        if self.display_status:
            self.conf.logger.log(
                f"LR initialization (lr={self.conf.lr} for mini-batch size={self.conf.base_batch_size} and scaled to {self.learning_rate_} for local mini-batch size={self.conf.batch_size}): lr scaleup={self.is_scaledup}, lr warmup={self.is_warmuped}, learning_rate={self.learning_rate}."
            )

    def init_lr_scheduler(self):
        if self.conf.lr_scheduler == "MultiStepLR":
            if self.conf.lr_milestones is not None:
                milestones = [int(x) for x in self.conf.lr_milestones.split(",")]
            elif (
                self.conf.lr_milestone_ratios is not None
                and self.conf.lr_milestone_ratios != "None"
            ):
                milestone_ratios = [
                    float(x) for x in self.conf.lr_milestone_ratios.split(",")
                ]
                milestones = [
                    int(milestone_ratio * self.conf.local_n_epochs + 0.5)
                    for milestone_ratio in milestone_ratios
                ]
            else:
                milestones = [self.conf.local_n_epochs + 1]
            lr_scheduler = MultiStepLR(
                self.optimizer, milestones=milestones, gamma=self.conf.lr_decay
            )
            scheduler_info = f"use MultiStepLR scheduler: milestones={milestones}, decay_factor={self.conf.lr_decay}"
        elif self.conf.lr_scheduler == "ExponentialLR":
            lr_scheduler = ExponentialLR(self.optimizer, gamma=self.conf.lr_decay)
            scheduler_info = (
                f"use ExponentialLR scheduler: decay_factor={self.conf.lr_decay}"
            )
        elif self.conf.lr_scheduler == "ReduceLROnPlateau":
            raise NotImplementedError("not support ReduceLROnPlateau yet.")
            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     self.optimizer,
            #     factor=self.conf.lr_decay,
            #     mode="min",
            #     patience=self.conf.lr_patience,
            # )
        else:
            raise NotImplementedError(
                f"we do not support this scheduler={self.conf.lr_scheduler} yet."
            )

        # in case we need to warmup the learning rate scheduler.
        if self.is_warmuped:
            self.lr_scheduler = GradualWarmupScheduler(
                optimizer=self.optimizer,
                multiplier=self.lr_scaleup_factor,
                total_epoch=self.conf.lr_warmup_epochs,
                after_scheduler=lr_scheduler,
            )
            warmup_info = f"first warmup lr={self.lr_scaleup_init_lr} with factor={self.lr_scaleup_factor} from {self.lr_scaleup_init_lr} to {self.learning_rate} for {self.conf.lr_warmup_epochs} epochs, then "
        else:
            self.lr_scheduler = lr_scheduler
            warmup_info = f"first set lr={self.learning_rate}, then "
        if self.display_status:
            self.conf.logger.log(
                f"LR scheduler in a nutshell: {warmup_info}{scheduler_info}."
            )

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, **kargs):
        self.update_training_progress()
        self.lr_scheduler.step(epoch=self.epoch_)

    def update_training_progress(self):
        self.local_index += 1
        self._update_training_progress()

    def _update_training_progress(self):
        self.epoch_ = self.local_index / self.conf.num_batches_per_device_per_epoch
        self.conf.local_index = self.local_index
        self.conf.epoch_ = self.epoch_
        self.epoch = int(self.epoch_)
        self.conf.epoch = self.epoch

    def clean(self):
        self.local_index = 0

    def update_from_checkpoint(self, checkpoint):
        self.conf.local_index = checkpoint["local_index"]
        self.local_index = checkpoint["local_index"]
        self.conf.best_perf = checkpoint["best_perf"]


"""copy a fixed version of pytorch scheduler to avoid the incorrect lr scheduler in the future pytorch version."""


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )
        self._step_count += 1

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class MultiStepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class ExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return self.base_lrs
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch for base_lr in self.base_lrs]


"""GradualWarmupScheduler."""


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning.

        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler is not None:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
