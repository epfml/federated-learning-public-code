# -*- coding: utf-8 -*-
import copy

import numpy as np
import torch
import torch.utils.data as data

import pcode.aggregation.utils as agg_utils
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.utils.misc import to_one_hot


def aggregate(
    conf,
    fedavg_model,
    client_models,
    criterion,
    metrics,
    flatten_local_models,
    fa_val_perf,
):
    if (
        "top1_starting_threshold" in conf.fl_aggregate
        and fa_val_perf["top1"] > conf.fl_aggregate["top1_starting_threshold"]
    ):
        # recover the models on the computation device.
        _, local_models = agg_utils.recover_models(
            conf, client_models, flatten_local_models
        )

        # generate the data for each local models.
        generated_data = {}
        for idx, local_model in local_models.items():
            conf.logger.log(f"distill the knowledge for model_idx={idx}.")
            kt_data_generator = DataGenerator(conf, model=local_model, model_idx=idx)
            generated_data[idx] = kt_data_generator.construct_data()

        #
        for out_iter in range(int(conf.fl_aggregate["outer_iters"])):
            conf.logger.log(f"starting the {out_iter}-th knowledge transfer.")
            for idx, dataset in generated_data.items():
                master_model = distill_knowledge(
                    conf,
                    fedavg_model,
                    dataset=dataset,
                    num_epochs=int(conf.fl_aggregate["inner_epochs"]),
                    batch_size=int(conf.fl_aggregate["kt_g_batch_size_per_class"]),
                    teacher_model=local_models[idx]
                    if "softmax_temperature" in conf.fl_aggregate
                    else None,
                    softmax_temperature=1
                    if "softmax_temperature" not in conf.fl_aggregate
                    else conf.fl_aggregate["softmax_temperature"],
                )

        # free the memory.
        del local_models
    else:
        conf.logger.log(f"skip and directly return the model.")

    # a temp hack (only for debug reason).
    client_models = dict(
        (used_client_arch, master_model.cpu())
        for used_client_arch in conf.used_client_archs
    )
    return master_model, client_models


class DataGenerator(object):
    def __init__(self, conf, model, model_idx):
        self.conf = conf
        self.model = copy.deepcopy(model)
        self.model_idx = model_idx

        # init to construct the data.
        assert "construct_scheme" in conf.fl_aggregate
        self.init_data_construction()
        self.construction_fn = self.get_construction_fn()

    def init_data_construction(self):
        self.pre_construction_info = {}

        if self.conf.fl_aggregate["construct_scheme"] == "class_with_smoothing":
            pass
        elif self.conf.fl_aggregate["construct_scheme"] == "class_similarity_dirichlet":
            normalized_c_weights = self.model.classifier.weight.data / torch.norm(
                self.model.classifier.weight.data, p=2
            )
            similarity_matrix = torch.matmul(
                normalized_c_weights, normalized_c_weights.t()
            )
            normalized_classes_similarity = (
                similarity_matrix - torch.min(similarity_matrix)
            ) / (torch.max(similarity_matrix) - torch.min(similarity_matrix))
            self.pre_construction_info[
                "normalized_classes_similarity"
            ] = normalized_classes_similarity.cpu()
        else:
            raise NotImplementedError(
                "this data construction scheme has not been implemented yet."
            )

    def _fn_class_with_smoothing(self):
        def _generate_for_given_class(
            batch_size_per_class, class_idx, num_classes, smooth_eps=1e-1
        ):
            simulated_probs_for_given_class = []
            while True:
                eps_nll = smooth_eps / num_classes
                simulated_prob = to_one_hot(torch.tensor(class_idx), n_dims=num_classes)
                simulated_prob.add_(eps_nll)
                simulated_prob[class_idx] - eps_nll * (num_classes + 1)

                # check the generated dirichlet distribution.
                if np.argmax(simulated_prob) == class_idx:
                    simulated_probs_for_given_class.append(
                        torch.FloatTensor(simulated_prob).unsqueeze(0)
                    )
                if len(simulated_probs_for_given_class) >= batch_size_per_class:
                    break

            # concat the generated probs to tensors.
            if len(simulated_probs_for_given_class) != 0:
                simulated_probs_for_given_class = torch.cat(
                    simulated_probs_for_given_class, dim=0
                )
            return simulated_probs_for_given_class

        def _generate(batch_size_per_class, smooth_eps=1e-1):
            simulated_probs = []
            for class_idx in range(self.model.num_classes):
                _tmp = _generate_for_given_class(
                    batch_size_per_class, class_idx, self.model.num_classes, smooth_eps
                )
                if len(_tmp) != 0:
                    simulated_probs.append(_tmp)
            return simulated_probs

        # construct different distributions by class.
        constructed_probs_and_labels = _generate(
            self.conf.fl_aggregate["kt_batch_size_per_class"],
            smooth_eps=self.conf.fl_aggregate["kt_smooth_eps"]
            if "kt_smooth_eps" in self.conf.fl_aggregate
            else 1e-1,
        )

        # generated the input based on the given distributions.
        return self._construct_input_via_expected_output_space(
            constructed_probs_and_labels
        )

    def _fn_class_similarity_dirichlet(self):
        # this function here corresponds to the paper https://arxiv.org/abs/1905.08114
        def _generate_dirc_dist_for_given_class(
            batch_size_per_class,
            class_idx,
            scaling_factor,
            normalized_classes_similarity,
            epsilon=1e-4,
        ):
            # generate the probs.
            simulated_probs_for_given_class = []
            while True:
                _sim = normalized_classes_similarity[class_idx, :].numpy()
                sim = (_sim - np.min(_sim)) / (
                    np.max(_sim) - np.min(_sim)
                ) * scaling_factor + epsilon
                simulated_prob = self.conf.random_state.dirichlet(sim)
                simulated_prob = simulated_prob / simulated_prob.sum()

                # check the generated dirichlet distribution.
                if np.argmax(simulated_prob) == class_idx:
                    simulated_probs_for_given_class.append(
                        torch.FloatTensor(simulated_prob).unsqueeze(0)
                    )
                if len(simulated_probs_for_given_class) >= batch_size_per_class:
                    break

            # concat the generated probs to tensors.
            if len(simulated_probs_for_given_class) != 0:
                simulated_probs_for_given_class = torch.cat(
                    simulated_probs_for_given_class, dim=0
                )
            return simulated_probs_for_given_class

        def _generate_dirc_dist(
            batch_size_per_class,
            scaling_factor,
            normalized_classes_similarity,
            epsilon=1e-4,
        ):
            simulated_probs = []
            for class_idx in range(self.model.num_classes):
                _tmp = _generate_dirc_dist_for_given_class(
                    batch_size_per_class,
                    class_idx,
                    scaling_factor,
                    normalized_classes_similarity,
                    epsilon,
                )
                if len(_tmp) != 0:
                    simulated_probs.append(_tmp)
            return simulated_probs

        # construct different dirichlet distributions.
        constructed_probs_and_labels = []
        assert "kt_scaling_factors" in self.conf.fl_aggregate
        self.conf.fl_aggregate["kt_scaling_factors"] = (
            [float(x) for x in self.conf.fl_aggregate["kt_scaling_factors"].split(":")]
            if type(self.conf.fl_aggregate["kt_scaling_factors"]) is str
            else self.conf.fl_aggregate["kt_scaling_factors"]
        )
        for scaling_factor in self.conf.fl_aggregate["kt_scaling_factors"]:
            _tmp = _generate_dirc_dist(
                self.conf.fl_aggregate["kt_batch_size_per_class"],
                scaling_factor,
                normalized_classes_similarity=self.pre_construction_info[
                    "normalized_classes_similarity"
                ],
                epsilon=1e-4,
            )
            if len(_tmp) != 0:
                constructed_probs_and_labels += _tmp
        constructed_probs_and_labels = torch.cat(constructed_probs_and_labels, dim=0)

        # generated the input based on these dirichlet distributions.
        return self._construct_input_via_expected_output_space(
            constructed_probs_and_labels
        )

    def _construct_input_via_expected_output_space(self, constructed_probs_and_labels):
        # generated the input based on these dirichlet distributions.
        generated_inputs, generated_probs = [], []
        model = agg_utils.modify_model_trainable_status(
            self.conf, self.model, trainable=False
        )
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        tracker = RuntimeTracker(metrics_to_track=[])

        # init the dataset for the training
        dataset = CustomDataset(constructed_probs_and_labels)
        data_loader = create_data_loader(
            dataset, batch_size=int(self.conf.fl_aggregate["kt_g_batch_size_per_class"])
        )
        num_update_per_batch = int(self.conf.fl_aggregate["kt_data_generate_iters"])
        self.conf.logger.log(
            f"# of mini-batches={len(data_loader)}, size of mini-batch={self.conf.fl_aggregate['kt_g_batch_size_per_class']}, # of update per-mini-batch={num_update_per_batch}"
        )

        # training the dataset.
        for batch_idx, probs in enumerate(data_loader):
            _generated_input = torch.rand(
                (len(probs), 3, 32, 32),
                requires_grad=True,
                device="cuda" if self.conf.graph.on_cuda else "cpu",
            )
            optimizer = torch.optim.Adam(
                [_generated_input],
                lr=self.conf.fl_aggregate["step_size"],
                betas=(self.conf.adam_beta_1, self.conf.adam_beta_2),
                eps=self.conf.adam_eps,
            )

            # improve the input_data to minic the output space of the network.
            for _ in range(num_update_per_batch):
                loss = update_input_data(
                    self.conf,
                    model,
                    criterion,
                    optimizer,
                    _generated_input,
                    expected_probs=probs.cuda() if self.conf.graph.on_cuda else probs,
                )
                tracker.update_metrics(
                    [loss.item()], n_samples=_generated_input.size(0)
                )
            self.conf.logger.log(
                f"\t the data generation loss (model index={self.model_idx}, batch index={batch_idx}) = {tracker()}."
            )
            tracker.reset()
            generated_inputs.append(copy.deepcopy(_generated_input.data))
            generated_probs.append(probs)
        generated_inputs = torch.cat(generated_inputs, dim=0).data.cpu()
        generated_probs = torch.cat(generated_probs, dim=0).data.cpu()
        return generated_inputs, generated_probs

    def get_construction_fn(self):
        if self.conf.fl_aggregate["construct_scheme"] == "class_similarity_dirichlet":
            return self._fn_class_similarity_dirichlet
        elif self.conf.fl_aggregate["construct_scheme"] == "class_with_smoothing":
            return self._fn_class_with_smoothing
        else:
            raise NotImplementedError

    def construct_data(self):
        generated_samples_and_probs = self.construction_fn()
        return CustomDataset(generated_samples_and_probs)


def distill_knowledge(
    conf,
    student_model,
    dataset,
    num_epochs,
    batch_size,
    teacher_model=None,
    softmax_temperature=1,
):
    # init.
    data_loader = create_data_loader(dataset, batch_size=batch_size)
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    tracker = RuntimeTracker(metrics_to_track=[])

    # check model status.
    untrainable_teacher_model = (
        agg_utils.modify_model_trainable_status(conf, teacher_model, trainable=False)
        if teacher_model is not None
        else None
    )
    trainable_student_model = agg_utils.check_trainable(conf, student_model)
    optimizer = create_optimizer(conf, trainable_student_model)

    # start the formal training.
    for epoch_idx in range(num_epochs):
        for _input, _target in data_loader:
            # init the _input, _target.
            if conf.graph.on_cuda:
                _input = _input.cuda()
            if untrainable_teacher_model is None and conf.graph.on_cuda:
                _target_prob = _target.cuda()

            # perform fp/bp on the student model.
            optimizer.zero_grad()
            _output = trainable_student_model(_input)

            # evaluate the loss.
            if untrainable_teacher_model is not None:
                loss = (softmax_temperature ** 2) * criterion(
                    torch.nn.functional.log_softmax(
                        _output / softmax_temperature, dim=1
                    ),
                    torch.nn.functional.softmax(
                        untrainable_teacher_model(_input).detach()
                        / softmax_temperature,
                        dim=1,
                    ),
                )
            else:
                loss = criterion(
                    torch.nn.functional.log_softmax(_output, dim=1), _target_prob
                )
            loss.backward()
            optimizer.step()
            tracker.update_metrics([loss.item()], n_samples=_input.size(0))
    conf.logger.log(f"# of epochs={epoch_idx + 1}: {tracker()}")
    return trainable_student_model.cpu()


def update_input_data(conf, model, criterion, optimizer, _inputs, expected_probs):
    model.train()
    optimizer.zero_grad()
    loss = criterion(
        torch.nn.functional.log_softmax(model(_inputs), dim=1), expected_probs
    )
    loss.backward()
    optimizer.step()
    _inputs.data = (_inputs.data - torch.min(_inputs.data)) / (
        torch.max(_inputs.data) - torch.min(_inputs.data)
    )
    return loss


"""Custom the dataset."""


def create_data_loader(
    dataset, batch_size, shuffle=True, num_workers=2, pin_memory=False
):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def concat_dataset(dataset1: data.Dataset, dataset2: data.Dataset):
    samples = dataset1.samples + dataset2.samples
    if dataset1.probs is not None and dataset2.probs is not None:
        probs = dataset1.probs + dataset2.probs
    elif dataset1.probs is None and dataset2.probs is not None:
        probs = dataset2.probs
    elif dataset1.probs is not None and dataset2.probs is None:
        probs = dataset1.probs
    else:
        probs = None
    return CustomDataset(data=(samples, probs))


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        if type(data) is tuple:
            self.samples, self.probs = data
        else:
            self.samples = data
            self.probs = None
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.probs is not None:
            return sample, self.probs[idx]
        else:
            return sample


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
