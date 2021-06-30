from .base_hook import Hook

import torch
import pickle
import os
from copy import deepcopy
from math import ceil
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class LRScheduler(Hook):
    """ lr scheduler updates optimizer's learning rate and dumps the snapshot """

    def __init__(self, num_batches, num_epochs, num_snapshots, opt, checkpoint_root):
        super(LRScheduler, self).__init__()
        self.num_snapshots = num_snapshots
        self.tot_iterations = num_batches * num_epochs
        self.snapshot_span = ceil(self.tot_iterations / self.num_snapshots)
        self.scheduler = CosineAnnealingWarmRestarts(opt, T_0=self.snapshot_span)

        self.snapshot_idx = 0  # which snapshot we are computing
        self.snapshot_batch_index = 1  # batch index within a snapshot span
        self.lrs = []
        self.losses = []
        self.checkpoint_root = checkpoint_root

    def on_train_begin(self):
        pass

    def on_batch_end(self):
        self.lrs.append(self.scheduler.get_lr()[0])
        self.losses.append(self.trainer.tracker.stat["loss"].val)
        self.log_fn(f"\n current learning rate {self.lrs[-1]}\n")
        self.snapshot_batch_index += 1
        self.scheduler.step()
        if self.snapshot_batch_index % self.snapshot_span == 0:
            self._dump_snapshot()
            self.log_fn(
                f"[INFO] dumped the {self.snapshot_idx+1}th snapshot @ batch step in span "
                f"{self.snapshot_batch_index} / {self.snapshot_span} "
                f"absolute batch step @ {self.batch_step}"
            )
            self.snapshot_batch_index = 1
            self.snapshot_idx += 1

    def _dump_snapshot(self):
        if not self.conf.train_fast:
            torch.save(
                deepcopy(self.get_state_dict()),
                os.path.join(
                    self.checkpoint_root, f"snapshot,{self.snapshot_idx},state_dict.pkl"
                ),
            )

    def on_validation_end(self, eval_res):
        pass

    def on_train_end(self):
        with open(os.path.join(self.checkpoint_root, "lrs.txt"), "w") as f:
            for lr in self.lrs:
                f.write(f"{lr}\n")
        self.plot_curve(self.lrs, "lrs")
        self.plot_curve(self.losses, "losses")

    def plot_curve(self, vals, label):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        plt.plot(range(len(vals)), vals)
        plt.grid()
        plt.xlabel("batch index", fontsize=20)
        plt.ylabel(f"{label}", fontsize=20)
        fig.savefig(
            os.path.join(self.checkpoint_root, f"{label}.pdf"), format="pdf", quality=50
        )

    def get_state_dict(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()
