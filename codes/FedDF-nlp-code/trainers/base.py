# -*- coding: utf-8 -*-
import torch
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import pcode.create_dataset as create_dataset


class BaseTrainer(object):
    """a basic trainer"""

    def __init__(
        self, conf, logger, data_partitioner, criterion=torch.nn.CrossEntropyLoss()
    ):
        self.conf = conf
        self.logger = logger
        self.log_fn_json = logger.log_metric
        self.log_fn = logger.log
        self.data_partitioner = data_partitioner
        self.criterion = criterion

        # counter.
        self._batch_step = 0
        self._epoch_step = 0

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def _parallel_to_device(self, model):
        model = model.cuda()
        if len(self.conf.world) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.conf.world)
        return model

    @property
    def batch_step(self):
        return self._batch_step

    @property
    def epoch_step(self):
        return self._epoch_step

    def _wrap_datasplits(self, data_iter, client_id):
        # The current client picks up its partition of data.
        loader, *_ = create_dataset.define_data_loader(
            self.conf,
            dataset=data_iter.trn_dl,
            # localdata_id start from 0 to the # of clients - 1.
            # client_id starts from 1 to the # of clients.
            localdata_id=client_id - 1,
            is_train=True,
            data_partitioner=self.data_partitioner,
        )

        setattr(self, "trn_dl", loader)


def parallel_to_device(conf, model):
    model = model.cuda()
    if len(conf.world) > 1:
        model = torch.nn.DataParallel(model, device_ids=conf.world)
    return model
