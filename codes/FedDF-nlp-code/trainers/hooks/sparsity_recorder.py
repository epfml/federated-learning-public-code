# -*- coding: utf-8 -*-
import json
import time
import copy
import functools
import numpy as np
import pickle
from collections import defaultdict, namedtuple
from .base_hook import Hook


StateSparsity = namedtuple("StateSparsity", "step sparsity")
StateFlr = namedtuple("StateFlr", "step flr_wrt_init flr_wrt_prev")
MicroCounts = namedtuple("MicroCounts", "n_tot n_nnz n_intersect n_union n_changed")


class _StatefulParam(object):
    """ given a parameter (weight or bias), record its status during training. """

    def __init__(self, name, init_mask):
        self.name = name
        self.init_mask = init_mask
        self.prev_mask = None
        self.step2sparsity = [StateSparsity(0, self.get_sparsity(init_mask))]
        self.step2flr = [StateFlr(0, 0, 0)]
        self.step2mask = []

    def update(self, step, step_mask):
        flr_wrt_init, micros_wrt_init = self.get_flip_ratio(self.init_mask, step_mask)
        flr_wrt_prev, micros_wrt_prev = self.get_flip_ratio(self.prev_mask, step_mask)
        self.step2sparsity.append(StateSparsity(step, self.get_sparsity(step_mask)))
        self.step2flr.append(StateFlr(step, flr_wrt_init, flr_wrt_prev))
        self.prev_mask = step_mask.data.cpu()
        # should not blowup machine memory for small datasets
        # uncomment this in future analysis
        # self.step2mask.append((step, step_mask.char().detach().cpu().numpy()))
        return micros_wrt_init, micros_wrt_prev

    def __repr__(self):
        mystr = "latest status of {}: \n".format(self.name)
        mystr += "\t step2sparsity: {}\n\t step2flr: {}".format(
            self.step2sparsity[-1], self.step2flr[-1]
        )
        return mystr

    def get_sparsity(self, mask):
        nnz = mask.norm(p=1).item()
        tot = mask.numel()
        return 1.0 * (tot - nnz) / tot

    @staticmethod
    def get_flip_ratio(old_mask, mask):
        if old_mask is None or mask is None:
            return (0, MicroCounts(0, 0, 0, 0, 0))
        assert old_mask.shape == mask.shape
        n_tot = mask.numel()
        n_nnz = mask.norm(p=1).item()

        _old_mask = old_mask.cuda()
        n_intersect = (_old_mask * mask).norm(p=1).item()
        n_union = (_old_mask + mask).norm(p=1).item() - n_intersect
        n_changed = (_old_mask - mask).norm(p=1).item()
        flip_ratio = n_changed / n_tot
        # n_* are returned for computing the micro within the hook
        del _old_mask
        return (flip_ratio, MicroCounts(n_tot, n_nnz, n_intersect, n_union, n_changed))


class RecorderForMicro(object):
    def __init__(self, wrt, param_type):
        self.wrt = wrt
        self.param_type = param_type
        self.tot, self.nnz, self.chan, self.uni, self.inter = 0, 0, 0, 0, 0

    def update(self, stats):
        self.tot += stats.n_tot  # independto `wrt` except first on_validation_end
        self.nnz += stats.n_nnz  # independto `wrt` except first on_validation_end
        self.chan += stats.n_changed
        self.uni += stats.n_union
        self.inter += stats.n_intersect

    def compute(self):
        self.inter_over_uni = self.inter / self.uni if self.uni > 0 else 0
        self.avg_sparsity = 1.0 - self.nnz / self.tot if self.tot > 0 else 0
        self.avg_flr = self.chan / self.tot if self.tot > 0 else 0


class SparsityRecorder(Hook):
    """ record sparsity of a mudule."""

    def __init__(self, where_, init_masks):
        super(SparsityRecorder, self).__init__()
        self.where_ = where_
        self.params = {}
        # not sure, if it's better to split the init mask computation from Masker to here
        for param_name, param_init_mask in init_masks.items():
            self.params[param_name] = _StatefulParam(param_name, param_init_mask)
        self.old_masks = copy.deepcopy(init_masks)

    def on_train_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_validation_end(self):
        if self.conf.train_fast:
            return

        cases = {
            "wrt_init_weight": RecorderForMicro("init", "weight"),
            "wrt_prev_weight": RecorderForMicro("prev", "weight"),
            "wrt_init_bias": RecorderForMicro("init", "bias"),
            "wrt_prev_bias": RecorderForMicro("prev", "bias"),
        }
        for name, param in self.model.named_parameters():
            # discard prefix by dataparallel
            name = name.replace("module.", "")
            # Masker names uses _ rather than .
            name = name.replace(".weight_m", "_weight_m").replace(".bias_m", "_bias_m")
            if name not in self.params:
                continue
            step_mask = self.trainer.masker.eval_binarizer_fn(
                self.conf.name_of_masker, param, self.trainer.masker.threshold
            )
            micros_wrt_init, micros_wrt_prev = self.params[name].update(
                self.batch_step, step_mask
            )
            if "_weight_m" in name:
                cases["wrt_init_weight"].update(micros_wrt_init)
                cases["wrt_prev_weight"].update(micros_wrt_prev)
            elif "_bias_m" in name:
                cases["wrt_init_bias"].update(micros_wrt_init)
                cases["wrt_prev_bias"].update(micros_wrt_prev)
        list(map(lambda x: x.compute(), cases.values()))

        inter_over_uni_wrt_init_mask = cases["wrt_init_weight"].inter_over_uni
        inter_over_uni_wrt_prev_mask = cases["wrt_prev_weight"].inter_over_uni

        avg_flr_init_weight = cases["wrt_init_weight"].avg_flr
        avg_flr_prev_weight = cases["wrt_prev_weight"].avg_flr

        avg_sparsity_weight = cases["wrt_init_weight"].avg_sparsity
        avg_sparsity_bias = cases["wrt_init_bias"].avg_sparsity

        mystr = "\n" + "-" * 25 + "\n"
        mystr += "w.r.t initial mask, inter_over_union: {:.3f}%, flip_ratio: {:.3f}%\n".format(
            100.0 * inter_over_uni_wrt_init_mask, 100.0 * avg_flr_init_weight
        )
        mystr += "w.r.t previous mask, inter_over_union: {:.3f}%, flip_ratio: {:.3f}%\n".format(
            100.0 * inter_over_uni_wrt_prev_mask, 100.0 * avg_flr_prev_weight
        )
        mystr += "weight avg sparsity: {:.3f}%; bias avg sparsity: {:.3f}% \n".format(
            100.0 * avg_sparsity_weight, 100.0 * avg_sparsity_bias
        )
        mystr += "flip_ratio_wrt_init_mask: {:.3f}%; flip_ratio_wrt_previous_mask: {:.3f}%.".format(
            100.0 * avg_flr_init_weight, 100.0 * avg_flr_prev_weight
        )
        mystr += "\n" + "-" * 25 + "\n"
        self.log_fn(mystr)
        for state_param in self.params.values():
            print(state_param)
        self.log_fn_json(
            name="evaluation",
            values={
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "step": self.batch_step,
                "epoch": self.epoch_step,
                "sparsities": [
                    avg_sparsity_weight,
                    avg_sparsity_bias,
                    inter_over_uni_wrt_init_mask,
                    avg_flr_init_weight,
                ],
            },
            tags={"split": "model_sparsity"},
            display=True,
        )

    def on_train_end(self):
        if not self.conf.train_fast:
            with open("{},paramstates.pkl".format(self.where_), "wb") as f:
                pickle.dump(self.params, f)
