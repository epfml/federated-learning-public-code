# -*- coding: utf-8 -*-
"""some utilities for mixup."""
import numpy as np
import torch


def mixup_criterion(criterion, pred, y_a, y_b, _lambda):
    return _lambda * criterion(pred, y_a) + (1 - _lambda) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0, assist_non_iid=False, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        _lambda = np.random.beta(alpha, alpha)
    else:
        _lambda = 1

    batch_size = x.size()[0]
    if not assist_non_iid:
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = _lambda * x + (1 - _lambda) * x[index, :]
        y_a, y_b = y, y[index]
    else:
        # build the sampling probability for target.
        unique_y, counts = torch.unique(y, sorted=True, return_counts=True)
        unique_y, counts = unique_y.unsqueeze(1), counts.unsqueeze(1)
        replaced_counts = y.clone()
        for _unique_y, _count in zip(unique_y, counts):
            replaced_counts = torch.where(
                replaced_counts == _unique_y, _count, replaced_counts
            )
        prob_y = 1.0 - 1.0 * replaced_counts / batch_size

        # get index.
        index = torch.multinomial(
            input=prob_y, num_samples=batch_size, replacement=True
        )
        if use_cuda:
            index = index.cuda()

        # mixup.
        mixed_x = _lambda * x + (1 - _lambda) * x[index, :]
        y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, _lambda
