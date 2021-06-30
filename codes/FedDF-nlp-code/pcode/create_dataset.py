# -*- coding: utf-8 -*-
import math
import functools

import numpy as np
import torch
import torch.distributed as dist

import data_loader.data_iters as data_iters


def seqcls_batch_to_device(batched):
    uids = batched[0]
    input_ids, golds, attention_mask, token_type_ids = map(
        lambda x: x.cuda(), batched[1:]
    )
    return (
        uids,
        golds,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "token_type_ids": token_type_ids,
        },
        None,
    )


def split_train_dataset(conf, train_dataset, agg_data_ratio=0, return_val=False):
    assert conf.train_data_ratio >= 0
    assert agg_data_ratio >= 0

    partition_sizes = (
        [
            conf.train_data_ratio,
            (1 - conf.train_data_ratio) * agg_data_ratio,
            (1 - conf.train_data_ratio) * (1 - agg_data_ratio),
        ]
        if agg_data_ratio > 0 and conf.train_data_ratio < 1
        else [conf.train_data_ratio, (1 - conf.train_data_ratio)]
    )

    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="origin",
        consistent_indices=False,
    )

    if agg_data_ratio == 0 or conf.train_data_ratio == 1:
        return data_partitioner.use(0), None, None
    elif return_val:
        return data_partitioner.use(0), data_partitioner.use(1), data_partitioner.use(2)
    else:
        return data_partitioner.use(0), data_partitioner.use(1), None


def define_data_loader(
    conf,
    dataset,
    localdata_id=None,
    is_train=True,
    shuffle=True,
    data_partitioner=None,
    agg_data_ratio=0,
    return_val=False,
):
    # determine the data to load,
    # either the whole dataset, or a subset specified by partition_type.
    if is_train:
        train_data, agg_data, val_data = split_train_dataset(
            conf, dataset, agg_data_ratio=agg_data_ratio, return_val=return_val
        )

        world_size = conf.n_clients
        partition_sizes = [1.0 / world_size for _ in range(world_size)]
        assert localdata_id is not None

        # (general) partitioned by "labels".
        # in case we have a global dataset and want to manually partition them.
        if data_partitioner is None:
            # update the data_partitioner.
            data_partitioner = DataPartitioner(
                conf, train_data, partition_sizes, partition_type=conf.partition_data
            )
        # note that the master node will not consume the training dataset.
        data_to_load = data_partitioner.use(localdata_id)
        conf.logger.log(
            f"Data partition for train (client_id={localdata_id + 1}): partitioned data and use subdata."
        )
    else:
        data_to_load = dataset
        conf.logger.log("Data partition for validation/test.")
        shuffle = False
        agg_data = None
        val_data = None

    # use Dataloader.
    data_loader = torch.utils.data.DataLoader(
        data_to_load,
        batch_size=conf.batch_size,
        shuffle=shuffle,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
    )

    # Some simple statistics.
    conf.logger.log(
        "\tData stat for {}: # of samples={} for {}. # of batches={}. The batch size={}".format(
            "train" if is_train else "validation/test",
            len(data_to_load),
            f"client_id={localdata_id + 1}" if localdata_id is not None else "Master",
            len(data_loader),
            conf.batch_size,
        )
    )

    if is_train:
        return data_loader, data_partitioner, agg_data, val_data
    else:
        return data_loader


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices
        self.replaced_targets = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        if self.replaced_targets is None:
            return self.data[data_idx]
        else:
            return (self.data[data_idx][0], self.replaced_targets[index])

    def update_replaced_targets(self, replaced_targets):
        self.replaced_targets = replaced_targets

        # evaluate the the difference between original labels and the simulated labels.
        count = 0
        for index in range(len(replaced_targets)):
            data_idx = self.indices[index]

            if self.replaced_targets[index] == self.data[data_idx][1]:
                count += 1
        return count / len(replaced_targets)

    def clean_replaced_targets(self):
        self.replaced_targets = None


class DataSampler(object):
    def __init__(self, conf, data, data_scheme, data_percentage):
        # init.
        self.conf = conf
        self.data = data
        self.data_size = len(self.data)
        self.data_scheme = data_scheme
        self.data_percentage = data_percentage

        # get unshuffled indices.
        self.indices = np.array([x for x in range(0, self.data_size)])
        self.sampled_indices = None

    def sample_indices(self):
        if self.data_scheme == "random_sampling":
            self.sampled_indices = self.conf.random_state.choice(
                self.indices,
                size=int(self.data_size * self.data_percentage),
                replace=False,
            )
        else:
            raise NotImplementedError(
                "this sampling scheme has not been supported yet."
            )

    def use_indices(self, sampled_indices=None):
        assert sampled_indices is not None or self.sampled_indices is not None
        return Partition(
            self.data,
            indices=sampled_indices
            if sampled_indices is not None
            else self.sampled_indices,
        )


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(
        self, conf, data, partition_sizes, partition_type, consistent_indices=True
    ):
        # prepare info.
        self.conf = conf
        self.partition_sizes = partition_sizes
        self.partition_type = partition_type
        self.consistent_indices = consistent_indices
        self.partitions = []

        # get data, data_size, indices of the data.
        self.data_size = len(data)
        if type(data) is not Partition:
            self.data = data
            indices = np.array([x for x in range(0, self.data_size)])
        else:
            self.data = data.data
            indices = data.indices

        # apply partition function.
        self.partition_indices(indices)

    def partition_indices(self, indices):
        indices = self._create_indices(indices)
        if self.consistent_indices:
            indices = self._get_consistent_indices(indices)

        # partition indices.
        from_index = 0
        for partition_size in self.partition_sizes:
            to_index = from_index + int(partition_size * self.data_size)
            self.partitions.append(indices[from_index:to_index])
            from_index = to_index

        # display the class distribution over the partitions.
        record_class_distribution(
            self.partitions, self.data.golds, print_fn=self.conf.logger.log
        )

    def _create_indices(self, indices):
        if self.partition_type == "origin":
            pass
        elif self.partition_type == "random":
            # it will randomly shuffle the indices.
            self.conf.random_state.shuffle(indices)
        elif self.partition_type == "sorted":
            # it will sort the indices based on the data label.
            indices = [
                i[0]
                for i in sorted(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.golds)
                        if idx in indices
                    ],
                    key=lambda x: x[1],
                )
            ]
        elif self.partition_type == "non_iid_dirichlet":
            num_classes = len(np.unique(self.data.golds))
            num_indices = len(indices)
            n_workers = len(self.partition_sizes)

            list_of_indices = build_non_iid_by_dirichlet(
                random_state=self.conf.random_state,
                indices2targets=np.array(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.golds)
                        if idx in indices
                    ]
                ),
                non_iid_alpha=self.conf.non_iid_alpha,
                num_classes=num_classes,
                num_indices=num_indices,
                n_workers=n_workers,
            )
            indices = functools.reduce(lambda a, b: a + b, list_of_indices)
        else:
            raise NotImplementedError(
                f"The partition scheme={self.partition_type} is not implemented yet"
            )
        return indices

    def _get_consistent_indices(self, indices):
        if dist.is_initialized():
            # sync the indices over clients.
            indices = torch.IntTensor(indices)
            dist.broadcast(indices, src=0)
            return list(indices)
        else:
            return indices

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])


def build_non_iid_by_dirichlet(
    random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    n_auxi_workers = 10
    # assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index : (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


def record_class_distribution(partitions, targets, print_fn):
    targets_of_partitions = {}
    targets_np = np.array(targets)
    for idx, partition in enumerate(partitions):
        unique_elements, counts_elements = np.unique(
            targets_np[partition], return_counts=True
        )
        targets_of_partitions[idx] = list(zip(unique_elements, counts_elements))
    print_fn(
        f"the histogram of the targets in the partitions: {targets_of_partitions.items()}"
    )
    return targets_of_partitions
