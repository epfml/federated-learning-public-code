# -*- coding: utf-8 -*-
import ot
import torch
import numpy as np

import pcode.aggregation.utils as agg_utils


def aggregate(conf, client_models, flatten_local_models):
    # init the local models.
    wasserstein_conf = {
        "exact": True,
        "correction": True,
        "proper_marginals": True,
        "skip_last_layer": False,
        "ensemble_step": 0.5,
        "reg": 1e-2,
        "past_correction": True,
        "unbalanced": False,
        "ground_metric": "euclidean",
        "ground_metric_eff": True,
        "ground_metric_normalize": "none",
        "clip_gm": False,
        "clip_min": 0.0,
        "clip_max": 5,
        "activation_histograms": False,
        "dist_normalize": True,
        "act_num_samples": 100,
        "softmax_temperature": 1,
        "geom_ensemble_type": "wts",
        "normalize_wts": True,
        "importance": None,
    }

    num_models, local_models = agg_utils.recover_models(
        conf, client_models, flatten_local_models
    )

    local_models = list(local_models.values())
    _model = local_models[0]

    for idx in range(1, num_models):
        avg_aligned_layers = get_wassersteinized_layers_modularized(
            conf, wasserstein_conf, [_model, local_models[idx]]
        )
        _model = get_network_from_param_list(avg_aligned_layers, _model)
    return _model


def get_network_from_param_list(param_list, new_network):
    assert len(list(new_network.parameters())) == len(param_list)

    layer_idx = 0
    model_state_dict = new_network.state_dict()

    for key, _ in model_state_dict.items():
        model_state_dict[key] = param_list[layer_idx]
        layer_idx += 1

    new_network.load_state_dict(model_state_dict)
    return new_network


def get_wassersteinized_layers_modularized(conf, wasserstein_conf, networks, eps=1e-7):
    """
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*
    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    """
    avg_aligned_layers = []
    T_var = None
    previous_layer_shape = None
    ground_metric_object = GroundMetric(wasserstein_conf)
    device = torch.device("cuda") if conf.graph.on_cuda else torch.device("cpu")
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))

    for (
        idx,
        ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)),
    ) in enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        previous_layer_shape = fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        # mu = np.ones(fc_layer0_weight.shape[0])/fc_layer0_weight.shape[0]
        # nu = np.ones(fc_layer1_weight.shape[0])/fc_layer1_weight.shape[0]
        layer_shape = fc_layer0_weight.shape
        if len(layer_shape) > 2:
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(
                fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1
            )
            fc_layer1_weight_data = fc_layer1_weight.data.view(
                fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1
            )
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            if is_conv:
                M = ground_metric_object.process(
                    fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1),
                )
                # M = cost_matrix(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                #                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                # print("layer data is ", fc_layer0_weight_data, fc_layer1_weight_data)
                M = ground_metric_object.process(
                    fc_layer0_weight_data, fc_layer1_weight_data
                )
                # M = cost_matrix(fc_layer0_weight, fc_layer1_weight)

            aligned_wt = fc_layer0_weight_data
        else:
            # print("shape of layer: model 0", fc_layer0_weight_data.shape)
            # print("shape of layer: model 1", fc_layer1_weight_data.shape)
            # print("shape of previous transport map", T_var.shape)

            # aligned_wt = None, this caches the tensor and causes OOM
            if is_conv:
                T_var_conv = T_var.unsqueeze(0).repeat(
                    fc_layer0_weight_data.shape[2], 1, 1
                )
                aligned_wt = torch.bmm(
                    fc_layer0_weight_data.permute(2, 0, 1), T_var_conv
                ).permute(1, 2, 0)

                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1),
                )
            else:
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    fc_layer0_unflattened = fc_layer0_weight.data.view(
                        fc_layer0_weight.shape[0], T_var.shape[0], -1
                    ).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1),
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    # print("layer data (aligned) is ", aligned_wt, fc_layer1_weight_data)
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                # M = cost_matrix(aligned_wt, fc_layer1_weight)
                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
                # print("ground metric is ", M)
            if wasserstein_conf["skip_last_layer"] and idx == (num_layers - 1):
                print(
                    "Simple averaging of last layer weights. NO transport map needs to be computed"
                )
                if wasserstein_conf["ensemble_step"] != 0.5:
                    avg_aligned_layers.append(
                        (1 - wasserstein_conf["ensemble_step"]) * aligned_wt
                        + wasserstein_conf["ensemble_step"] * fc_layer1_weight
                    )
                else:
                    avg_aligned_layers.append((aligned_wt + fc_layer1_weight) / 2)
                return avg_aligned_layers

        if wasserstein_conf["importance"] is None or (idx == num_layers - 1):
            mu = get_histogram(wasserstein_conf, 0, mu_cardinality, layer0_name)
            nu = get_histogram(wasserstein_conf, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(
                wasserstein_conf, fc_layer0_weight_data, is_conv
            )
            nu = _get_neuron_importance_histogram(
                wasserstein_conf, fc_layer1_weight_data, is_conv
            )
            # print(mu, nu)
            assert wasserstein_conf["proper_marginals"]

        cpuM = M.data.cpu().numpy()
        if wasserstein_conf["exact"]:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=wasserstein_conf["reg"])
        # T = ot.emd(mu, nu, log_cpuM)

        T_var = torch.from_numpy(T).float()
        if conf.graph.on_cuda:
            T_var = T_var.cuda()

        # torch.set_printoptions(profile="full")
        # print("the transport map is ", T_var)
        # torch.set_printoptions(profile="default")

        if wasserstein_conf["correction"]:
            if not wasserstein_conf["proper_marginals"]:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                if conf.graph.on_cuda:
                    marginals = marginals.cuda()
                marginals = torch.diag(1.0 / (marginals + eps))  # take inverse
                T_var = torch.matmul(T_var, marginals)
            else:
                # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
                marginals_beta = T_var.t() @ torch.ones(
                    T_var.shape[0], dtype=T_var.dtype
                ).to(device)

                marginals = 1 / (marginals_beta + eps)
                # print("shape of inverse marginals beta is ", marginals_beta.shape)
                # print("inverse marginals beta is ", marginals_beta)

                T_var = T_var * marginals
                # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                # this should all be ones, and number equal to number of neurons in 2nd model
                # print(T_var.sum(dim=0))
                # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        if wasserstein_conf["past_correction"]:
            # print("this is past correction for weight mode")
            # print("Shape of aligned wt is ", aligned_wt.shape)
            # print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)
            t_fc0_model = torch.matmul(
                T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
            )
        else:
            t_fc0_model = torch.matmul(
                T_var.t(),
                fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
            )

        # Average the weights of aligned first layers
        if wasserstein_conf["ensemble_step"] != 0.5:
            geometric_fc = (
                1 - wasserstein_conf["ensemble_step"]
            ) * t_fc0_model + wasserstein_conf[
                "ensemble_step"
            ] * fc_layer1_weight_data.view(
                fc_layer1_weight_data.shape[0], -1
            )
        else:
            geometric_fc = (
                t_fc0_model
                + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            ) / 2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)
    return avg_aligned_layers


"""the utility function."""


def get_histogram(
    wasserstein_conf,
    idx,
    cardinality,
    layer_name,
    activations=None,
    return_numpy=True,
    float64=False,
):
    if activations is None:
        # returns a uniform measure
        if not wasserstein_conf["unbalanced"]:
            # print("returns a uniform measure of cardinality: ", cardinality)
            return np.ones(cardinality) / cardinality
        else:
            return np.ones(cardinality)
    else:
        # return softmax over the activations raised to a temperature
        # layer_name is like 'fc1.weight', while activations only contains 'fc1'
        # print(activations[idx].keys())
        unnormalized_weights = activations[idx][layer_name.split(".")[0]]
        # print(
        #     "For layer {},  shape of unnormalized weights is ".format(layer_name),
        #     unnormalized_weights.shape,
        # )
        unnormalized_weights = unnormalized_weights.squeeze()
        assert unnormalized_weights.shape[0] == cardinality

        if return_numpy:
            if float64:
                return (
                    torch.softmax(
                        unnormalized_weights / wasserstein_conf["softmax_temperature"],
                        dim=0,
                    )
                    .data.cpu()
                    .numpy()
                    .astype(np.float64)
                )
            else:
                return (
                    torch.softmax(
                        unnormalized_weights / wasserstein_conf["softmax_temperature"],
                        dim=0,
                    )
                    .data.cpu()
                    .numpy()
                )
        else:
            return torch.softmax(
                unnormalized_weights / wasserstein_conf["softmax_temperature"], dim=0
            )


def _get_neuron_importance_histogram(wasserstein_conf, layer_weight, is_conv, eps=1e-9):
    # print("shape of layer_weight is ", layer_weight.shape)
    if is_conv:
        layer = layer_weight.contiguous().view(layer_weight.shape[0], -1).cpu().numpy()
    else:
        layer = layer_weight.cpu().numpy()

    if wasserstein_conf["importance"] == "l1":
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(np.float64) + eps
    elif wasserstein_conf["importance"] == "l2":
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(np.float64) + eps
    else:
        raise NotImplementedError

    if not wasserstein_conf["unbalanced"]:
        importance_hist = importance_hist / importance_hist.sum()
        # print("sum of importance hist is ", importance_hist.sum())
    # assert importance_hist.sum() == 1.0
    return importance_hist


def isnan(x):
    return x != x


class GroundMetric:
    """Ground Metric object for Wasserstein computations."""

    def __init__(self, params, not_squared=False):
        self.params = params
        self.ground_metric_type = params["ground_metric"]
        self.ground_metric_normalize = params["ground_metric_normalize"]
        self.reg = params["reg"]
        if "not_squared" in params:
            self.squared = not params["not_squared"]
        else:
            # so by default squared will be on!
            self.squared = not not_squared
        self.mem_eff = params["ground_metric_eff"]

    def _clip(self, ground_metric_matrix):
        # if self.params.debug:
        #     print("before clipping", ground_metric_matrix.data)

        percent_clipped = (
            float(
                (ground_metric_matrix >= self.reg * self.params["clip_max"])
                .long()
                .sum()
                .data
            )
            / ground_metric_matrix.numel()
        ) * 100
        # print("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        setattr(self.params, "percent_clipped", percent_clipped)
        # will keep the M' = M/reg in range clip_min and clip_max
        ground_metric_matrix.clamp_(
            min=self.reg * self.params["clip_min"],
            max=self.reg * self.params["clip_max"],
        )
        # if self.params.debug:
        #     print("after clipping", ground_metric_matrix.data)
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):

        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            # print(
            #     "Normalizing by max of ground metric and which is ",
            #     ground_metric_matrix.max(),
            # )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.ground_metric_normalize == "median":
            # print(
            #     "Normalizing by median of ground metric and which is ",
            #     ground_metric_matrix.median(),
            # )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.ground_metric_normalize == "mean":
            # print(
            #     "Normalizing by mean of ground metric and which is ",
            #     ground_metric_matrix.mean(),
            # )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (isnan(ground_metric_matrix).any())

    def _cost_matrix_xy(self, x, y, p=2, squared=True):
        # TODO: Use this to guarantee reproducibility of previous results and then move onto better way
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        if not squared:
            # print("dont leave off the squaring of the ground metric")
            c = c ** (1 / 2)
        # print(c.size())
        if self.params["dist_normalize"]:
            assert NotImplementedError
        return c

    def _pairwise_distances(self, x, y=None, squared=True):
        """
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        dist = torch.clamp(dist, min=0.0)

        if self.params["activation_histograms"] and self.params["dist_normalize"]:
            dist = dist / self.params["act_num_samples"]
            # print("Divide squared distances by the num samples")

        if not squared:
            # print("dont leave off the squaring of the ground metric")
            dist = dist ** (1 / 2)

        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        # TODO: Replace by torch.pdist (which is said to be much more memory efficient)
        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1])
                - coordinates,
                p=2,
                dim=2,
            )
        else:
            if self.mem_eff:
                matrix = self._pairwise_distances(
                    coordinates, other_coordinates, squared=self.squared
                )
            else:
                matrix = self._cost_matrix_xy(
                    coordinates, other_coordinates, squared=self.squared
                )

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        # print(
        #     "stats of vecs are: mean {}, min {}, max {}, std {}".format(
        #         norms.mean(), norms.min(), norms.max(), norms.std()
        #     )
        # )
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1)
                @ torch.norm(other_coordinates, dim=1).view(1, -1),
            )
        return matrix.clamp_(min=0)

    def _get_angular(self, coordinates, other_coordinates=None):
        pass

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            "euclidean": self._get_euclidean,
            "cosine": self._get_cosine,
            "angular": self._get_angular,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        # print("Processing the coordinates to form ground_metric")
        if self.params["geom_ensemble_type"] == "wts" and self.params["normalize_wts"]:
            # print("In weight mode: normalizing weights to unit norm")
            coordinates = self._normed_vecs(coordinates)
            if other_coordinates is not None:
                other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)

        # if self.params.debug:
        #     print("coordinates is ", coordinates)
        #     if other_coordinates is not None:
        #         print("other_coordinates is ", other_coordinates)
        #     print("ground_metric_matrix is ", ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        ground_metric_matrix = self._normalize(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params["clip_gm"]:
            ground_metric_matrix = self._clip(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        # if self.params.debug:
        #     print("ground_metric_matrix at the end is ", ground_metric_matrix)

        return ground_metric_matrix
