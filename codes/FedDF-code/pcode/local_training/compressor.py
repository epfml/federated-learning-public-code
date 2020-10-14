# -*- coding: utf-8 -*-
import math

import torch


class Quantization(object):
    def __init__(
        self,
        conf,
        quantize_model_mode="Q6",
        normalize_weight_mode="filterwise_mean",
        standardize_weight_mode="filterwise_std",
        quantize_model_use_scaling=True,
        quantize_model_scaling_mode="itself",
        quantize_model_add_mean_back=True,
        quantize_model_clip=True,
    ):
        assert quantize_model_mode is not None
        self.conf = conf
        self.quantize_fn = getattr(self, quantize_model_mode)

        self.normalize_weight_mode = normalize_weight_mode
        self.standardize_weight_mode = standardize_weight_mode
        self.quantize_model_use_scaling = quantize_model_use_scaling
        self.quantize_model_scaling_mode = quantize_model_scaling_mode
        self.quantize_model_add_mean_back = quantize_model_add_mean_back
        self.quantize_model_clip = quantize_model_clip

    def _filterwise_mean(self, param_data):
        return (
            param_data.mean(dim=3, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=1, keepdim=True)
            .mul(-1)
        )

    def _layerwise_mean(self, param_data):
        return param_data.mean().mul(-1)

    def _filterwise_std(self, param_data, size):
        return torch.sqrt(
            torch.mean(param_data ** 2, dim=3, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=1, keepdim=True)
        )

    def _layerwise_std(self, param_data, size):
        return torch.sqrt(torch.mean(param_data ** 2))

    def _filterwise_norm(self, param_data, norm_type=1):
        if norm_type == float("inf"):
            maxed, _ = param_data.abs().max(dim=3, keepdim=True)
            maxed, _ = maxed.max(dim=2, keepdim=True)
            maxed, _ = maxed.max(dim=1, keepdim=True)
            return maxed
        else:
            return (
                param_data.norm(p=norm_type, dim=3, keepdim=True)
                .sum(dim=2, keepdim=True)
                .sum(dim=1, keepdim=True)
            )

    def _layerwise_norm(self, param_data, norm_type=1):
        return param_data.norm(p=norm_type, keepdim=True)

    def _normalize_weight(self, param_data, size, normalize_weight_mode):
        if normalize_weight_mode is None:
            return torch.zeros_like(param_data)
        elif normalize_weight_mode == "filterwise_mean":
            if len(size) == 4:
                neg_mean = self._filterwise_mean(param_data)
            else:
                neg_mean = self._layerwise_mean(param_data)
            return neg_mean
        elif normalize_weight_mode == "layerwise_mean":
            neg_mean = self._layerwise_mean(param_data)
            return neg_mean
        else:
            raise NotImplementedError

    def _standardize_weight(
        self, param_data, size, standardize_weight_mode, epsilon=1e-8
    ):
        if standardize_weight_mode is None:
            denon = torch.ones_like(param_data)
        elif standardize_weight_mode == "filterwise_std":
            if len(size) == 4:
                denon = self._filterwise_std(param_data, size)
            else:
                denon = self._layerwise_std(param_data, size)
            denon = denon.add_(epsilon)
        elif "linf_norm" in standardize_weight_mode:
            norm_type = float("inf")

            if len(size) == 4 and "filterwise" in standardize_weight_mode:
                denon = self._filterwise_norm(param_data, norm_type=norm_type)
            else:
                denon = self._layerwise_norm(param_data, norm_type=norm_type)
            denon = denon.add_(epsilon)
        else:
            raise NotImplementedError
        return denon

    def _quantize(self, x, n_bits):
        # x should in the range of [-1, 1].
        if n_bits == 1:
            return torch.sign(x)
        else:
            _sign = torch.sign(x)
            _abs = x.abs()
            n = 1.0 * (2 ** (n_bits - 1) - 1)
            return _sign * torch.round(n * _abs) / n

    r""" the quantization function listed below can be found in the related work.
        Q6: Xnor-net: Imagenet classification using binary convolutional neural networks.
            b * quantize_k ((W_i - a) / b) + a,
            where `a` could be the mean/median of W and `b` could be L_1, L_2, L_\infinity norm, or the variance of W.
        Q2: Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients.
            1-bit quantization:
                \frac{1}{n} \norm{W}_1 * sign(W_i)
            k-bit quantization:
                2 * quantize_k( \frac{ \tanh(W_i) }{ 2 \max{ \abs{ \tanh{W_i} } } } + 0.5 ) - 1
        Q5: Training and inference with integers in deep neural networks.
            k-bit quantization:
                \clip{ \sigma(k) * \round{ \frac{ W_i }{ \sigma(k) } }, -1 + \sigma(k), 1 - \sigma(k) },
                where \sigma(k) = 2^{1 - k}
        Q4: Incremental network quantization: Towards lossless cnns with low-precision weights.
    """

    def Q6(self, param_name, param, n_bits, epsilon=1e-8):
        param_data = param.data.clone()
        size = param_data.size()

        # decide if normalize the weights.
        neg_mean = self._normalize_weight(
            param_data, size, normalize_weight_mode=self.normalize_weight_mode
        )
        param_data.add_(neg_mean.expand(size))

        # decide if standardize the weight.
        denon = self._standardize_weight(
            param_data, size, standardize_weight_mode=self.standardize_weight_mode
        )
        param_data.div_(denon.expand(size))

        # decide if clip the normalized weights.
        if self.quantize_model_clip:
            param_data.clamp_(-1.0, 1.0)

        # decide if get the scaling factor.
        if self.quantize_model_use_scaling:
            if "l1" in self.quantize_model_scaling_mode:
                nelement = param_data.nelement()

                if len(size) == 4 and "filterwise" in self.normalize_weight_mode:
                    scaling = self._filterwise_norm(param_data, norm_type=1)
                else:
                    scaling = self._layerwise_norm(param_data, norm_type=1)
                scaling = scaling.div(nelement).expand(size)

                if "both" in self.quantize_model_scaling_mode:
                    scaling.mul_(denon)
            elif self.quantize_model_scaling_mode == "itself":
                scaling = denon.expand(size)
        else:
            scaling = 1

        # decide if we need to add the mean back.
        if self.quantize_model_add_mean_back:
            mean = -neg_mean.expand(size)
        else:
            mean = 0
        return self._quantize(param_data, n_bits).mul(scaling) + mean

    def Q2(self, param_name, param, n_bits, epsilon=1e-8):
        # get weight stat.
        param_data = param.data.clone()
        size = param_data.size()

        if self.quantize_model_use_scaling:
            assert n_bits == 1

            # quantize weights.
            alpha = param_data.abs().mean() + epsilon
            alpha = alpha.expand(size)
            return torch.sign(param_data.div(alpha)).mul(alpha)
        else:
            # quantize weights wo scaling.
            _d = torch.tanh(param_data)
            _d_abs_max = _d.abs().max()

            _epsilon = epsilon if _d_abs_max == 0 else 0

            if n_bits == 1:
                return torch.sign(param_data)
            else:
                return (
                    2 * self._quantize(0.5 * _d / (_d_abs_max + _epsilon) + 0.5, n_bits)
                    - 1
                )

    def Q5(self, param_name, param, n_bits):
        def _sigma(bits):
            return 2.0 ** (bits - 1)

        def _shift_factor(x):
            return 2 ** torch.round(torch.log2(x))

        def _c(x, bits):
            delta = 0.0 if bits > 15 or bits == 1 else 1.0 / _sigma(bits)
            return x.clamp(-1 + delta, +1 - delta)

        def _q(x, bits):
            if bits > 15:
                return x
            elif bits == 1:
                return torch.sign(x)
            else:
                _scale = _sigma(bits)
                return torch.round(x * _scale) / _scale

        param_data = param.data.clone()
        # it is equivalent to _c(_q(x, bits), bits)
        return _q(_c(param_data, n_bits), n_bits)

    def Q4(self, param_name, param, n_bits):
        """not very efficient."""
        param_data = param.data.clone()

        # get n_1 and n_2
        s = param_data.abs().max().item()
        if s == 0:
            return param_data
        else:
            n_1 = math.floor(math.log((4 * s) / 3, 2))
            n_2 = int(n_1 + 1 - (2 ** (n_bits - 1)) / 2)

        def _quantize_weight(weight):
            """Quantize a single weight using the INQ quantization scheme."""
            alpha = 0
            beta = 2 ** n_2
            abs_weight = math.fabs(weight)

            for i in range(int(n_2), int(n_1) + 1):
                if (abs_weight >= (alpha + beta) / 2) and abs_weight < (3 * beta / 2):
                    return math.copysign(beta, weight)
                else:
                    alpha = 2 ** i
                    beta = 2 ** (i + 1)
            return 0

        # quantize weights.
        param_data.cpu().apply_(_quantize_weight).to(param_data.device)
        return param_data

    def quantize(self, param_name, param, n_bits):
        return self.quantize_fn(param_name, param, n_bits)


class ModelQuantization(object):
    """a class of model quantization with memory (wrt each param)."""

    def __init__(self, conf):
        self.conf = conf
        self.params_memory = dict()

        # get the hyper-parameters.
        self.use_memory = True
        self.skip_bn = True
        self.nbits = 1

        # define the quantization function.
        self.quantizer = Quantization(conf=conf)

    def _get_memory(self, param_name, param):
        if param_name not in self.params_memory:
            self.params_memory[param_name] = torch.zeros_like(param)
        return self.params_memory[param_name]

    def _apply_memory(self, param_name, param):
        if self.use_memory:
            # get the memory and apply them.
            memory = self._get_memory(param_name, param)
            param.data.add_(memory)

    def _update_memory(self, param_name, new_memeory):
        if self.use_memory:
            self.params_memory[param_name].data = new_memeory

    def _quantize_param(self, param_name, param, epoch):
        # apply the memory if we use_memory.
        self._apply_memory(param_name, param)

        # get quantized data.
        n_bits = self.nbits
        quantized = self.quantizer.quantize(param_name, param.data, n_bits)

        # update the memory if we use_memory.
        self._update_memory(param_name, param.data - quantized)

        # update param.
        param.data = quantized

    def _is_quantize_bn(self, param_name):
        return (not self.skip_bn) or (self.skip_bn and "bn" not in param_name)

    def compress_model(self, param_groups, epoch=None):
        # quantize the params. during the training.
        for group in param_groups:
            param_name = group["name"]

            for p in group["params"]:
                if self._is_quantize_bn(param_name):
                    self._quantize_param(param_name=param_name, param=p, epoch=epoch)
