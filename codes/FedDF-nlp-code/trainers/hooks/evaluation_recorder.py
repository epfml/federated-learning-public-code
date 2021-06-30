from .base_hook import Hook
from copy import deepcopy
import torch
import pickle


_valid_metrics = {
    "accuracy",
    "f1",
    "micro_f1",
    "macrof1",
    "recall",
    "precision",
    "mcc",
    "f1_score_ner",
}
# not considering metrics where the lower the better (ppl)


class EvaluationRecorder(Hook):
    """record the best performing state and evaluation results."""

    def __init__(self, init_state_where_, where_, which_metric="accuracy"):
        if which_metric not in _valid_metrics:
            raise ValueError("Invalid evaluation metric!")
        super(EvaluationRecorder, self).__init__()
        self.init_state_where_ = init_state_where_
        self.where_ = where_
        self.which_metric = which_metric
        self.best_score = -1
        self.best_step = -1
        self.best_state = {}

    def on_train_begin(self):
        if hasattr(self.conf, "train_fast") and not self.conf.train_fast:
            torch.save(self.get_state_dict(), self.init_state_where_)
            self.log_fn(f"[INFO] {self.__class__.__name__} saved initial state.")

    def on_batch_end(self):
        pass

    def get_state_dict(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def on_validation_end(self, eval_res):
        if eval_res is None:
            return
        curr_score = eval_res["val_dl"][self.which_metric]
        if curr_score > self.best_score:
            self.best_step = deepcopy(self.batch_step)
            self.best_state = {
                "which_step": self.best_step,
                "eval_res": eval_res,
                "{}".format(self.which_metric): curr_score,
                "state_dict": deepcopy(self.get_state_dict()),
            }
            self.log_fn(
                f"[INFO] Update {self.__class__.__name__}: best {self.which_metric}={self.best_score:.5f} @ {self.best_step} < {curr_score:.5f} (this batch @ {self.batch_step})."
            )
            self.best_score = curr_score
        else:
            self.log_fn(
                f"[INFO] Not update {self.__class__.__name__}: best {self.which_metric}={self.best_score:.5f} @ {self.best_step} > {curr_score:.5f} (this batch @ {self.batch_step})."
            )

    def on_train_end(self):
        if self.where_ is None:
            self.log_fn(f"[INFO] {self.__class__.__name__} is NOT saving anything ... ")
            return
        if hasattr(self.conf, "train_fast") and not self.conf.train_fast:
            torch.save(self.best_state, self.where_)
        self.log_fn(
            f"[INFO] {self.__class__.__name__} saved best state. best score -> {self.best_score:.5f} @ {self.best_step}."
        )
