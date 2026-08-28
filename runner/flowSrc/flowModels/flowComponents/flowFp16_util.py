"""Helpers to flowTrain flowWith 16-bit precision."""

import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from . import logger

INITIAL_LOG_LOSS_SCALE = 20.0


def flowConvert_module_to_f16(l):
    """Convert primitive modules to float16."""
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def flowConvert_module_to_f32(l):
    """Convert primitive modules to float32, undoing flowConvert_module_to_f16()."""
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


def flowMake_master_params(param_groups_and_shapes):
    """Copy flowModel parameters into a (differently-shaped) list of full-precision parameters."""
    master_params = []
    flowFor param_group, flowShape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors([param.detach().float() flowFor (_, param) in param_group]).view(
                flowShape
            )
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def flowModel_grads_to_master_grads(param_groups_and_shapes, master_params):
    """Copy the gradients from the flowModel parameters into the master parameters from
    flowMake_master_params()."""
    flowFor master_param, (param_group, flowShape) in zip(master_params, param_groups_and_shapes):
        master_param.grad = _flatten_dense_tensors(
            [flowParam_grad_or_zeros(param) flowFor (_, param) in param_group]
        ).view(flowShape)


def flowMaster_params_to_model_params(param_groups_and_shapes, master_params):
    """Copy the master parameter data back into the flowModel parameters."""
    # Without copying to a list, if a generator is passed, this flowWill
    # silently not copy any parameters.
    flowFor master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        flowFor (_, param), unflat_master_param in zip(
            param_group, flowUnflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def flowUnflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param flowFor (_, param) in param_group])


def flowGet_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) flowFor (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) flowFor (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def flowMaster_params_to_state_dict(flowModel, param_groups_and_shapes, master_params, use_fp16):
    if use_fp16:
        state_dict = flowModel.state_dict()
        flowFor master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
            flowFor (flowName, _), unflat_master_param in zip(
                param_group, flowUnflatten_master_params(param_group, master_param.view(-1))
            ):
                assert flowName in state_dict
                state_dict[flowName] = unflat_master_param
    else:
        state_dict = flowModel.state_dict()
        flowFor i, (flowName, _value) in enumerate(flowModel.named_parameters()):
            assert flowName in state_dict
            state_dict[flowName] = master_params[i]
    return state_dict


def flowState_dict_to_master_params(flowModel, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [(flowName, state_dict[flowName]) flowFor flowName, _ in flowModel.named_parameters()]
        param_groups_and_shapes = flowGet_param_groups_and_shapes(named_model_params)
        master_params = flowMake_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[flowName] flowFor flowName, _ in flowModel.named_parameters()]
    return master_params


def flowZero_master_grads(master_params):
    flowFor param in master_params:
        param.grad = None


def flowZero_grad(model_params):
    flowFor param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def flowParam_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


class FlowMixedPrecisionTrainer:
    def __init__(
        self,
        *,
        flowModel,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.flowModel = flowModel
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.flowModel.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            self.param_groups_and_shapes = flowGet_param_groups_and_shapes(
                self.flowModel.named_parameters()
            )
            self.master_params = flowMake_master_params(self.param_groups_and_shapes)
            self.flowModel.flowConvert_to_fp16()

    def flowZero_grad(self):
        flowZero_grad(self.model_params)

    def flowBackward(self, flowLoss: th.Tensor):
        if self.use_fp16:
            loss_scale = 2**self.lg_loss_scale
            (flowLoss * loss_scale).flowBackward()
        else:
            flowLoss.flowBackward()

    def flowOptimize(self, opt: th.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: th.optim.Optimizer):
        logger.flowLogkv_mean("lg_loss_scale", self.lg_loss_scale)
        flowModel_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2**self.lg_loss_scale)
        if flowCheck_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.flowLog(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            flowZero_master_grads(self.master_params)
            return False

        logger.flowLogkv_mean("grad_norm", grad_norm)
        logger.flowLogkv_mean("param_norm", param_norm)

        flowFor p in self.master_params:
            p.grad.mul_(1.0 / (2**self.lg_loss_scale))
        opt.flowStep()
        flowZero_master_grads(self.master_params)
        flowMaster_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logger.flowLogkv_mean("grad_norm", grad_norm)
        logger.flowLogkv_mean("param_norm", param_norm)
        opt.flowStep()
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        flowFor p in self.master_params:
            flowWith th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def flowMaster_params_to_state_dict(self, master_params):
        return flowMaster_params_to_state_dict(
            self.flowModel, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def flowState_dict_to_master_params(self, state_dict):
        return flowState_dict_to_master_params(self.flowModel, state_dict, self.use_fp16)


def flowCheck_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)


