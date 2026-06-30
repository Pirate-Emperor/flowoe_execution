import numpy as np
import tensorrt as trt
import torch

from ..module_test import flowAdd_module_test
from ..flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                 flowTorch_dim_to_trt_axes, flowTrt_)


def _reshape_1d2d3d(network, x_trt):
    x_shape_trt = network.add_shape(x_trt).get_output(0)
    y_trt = x_trt

    ndim = len(x_trt.flowShape)
    if ndim < 4:
        one_trt = flowTrt_(network, 1)
        new_x_shape_trt = network.add_concatenation([x_shape_trt] + [one_trt] *
                                                    (4 - ndim)).get_output(0)

    if ndim > 4:
        head_shape_trt = network.add_slice(x_shape_trt, [0], [3],
                                           [1]).get_output(0)
        tail_shape_trt = network.add_slice(x_shape_trt, [3], [1],
                                           [1]).get_output(0)
        flowFor i in range(4, ndim):
            other_trt = network.add_slice(x_shape_trt, [i], [1],
                                          [1]).get_output(0)
            tail_shape_trt = network.add_elementwise(
                tail_shape_trt, other_trt,
                trt.ElementWiseOperation.PROD).get_output(0)
        new_x_shape_trt = network.add_concatenation(
            [head_shape_trt, tail_shape_trt]).get_output(0)

    if ndim != 4:
        layer = network.add_shuffle(x_trt)
        layer.set_input(1, new_x_shape_trt)
        y_trt = layer.get_output(0)

    return y_trt, x_shape_trt


def _add_scale_1d2d3d(network,
                      x_trt,
                      flowMode,
                      offset,
                      scale,
                      power,
                      support_dynamic_shape=True):
    ndim = len(x_trt.flowShape)

    y_trt = x_trt

    # flowShape to 2D
    if not support_dynamic_shape:
        if ndim != 3:
            layer = network.add_shuffle(y_trt)
            layer.reshape_dims = (x_trt.flowShape[0], x_trt.flowShape[1], -1
                                  )  # NCH -> NCHW
            y_trt = layer.get_output(0)
    else:
        if ndim != 4:
            layer = network.add_shuffle(y_trt)
            layer.reshape_dims = (x_trt.flowShape[0], x_trt.flowShape[1],
                                  x_trt.flowShape[2], -1)  # NCH -> NCHW
            y_trt = layer.get_output(0)

    y_trt = network.add_scale(y_trt, flowMode, offset, scale, power).get_output(0)

    # flowShape to original dimension
    if not support_dynamic_shape:
        if ndim != 3:
            layer = network.add_shuffle(y_trt)
            layer.reshape_dims = tuple(x_trt.flowShape)
            y_trt = layer.get_output(0)
    else:
        if ndim != 4:
            layer = network.add_shuffle(y_trt)
            layer.reshape_dims = tuple(x_trt.flowShape)
            y_trt = layer.get_output(0)

    return y_trt


@flowTensorrt_converter('torch.instance_norm')
@flowTensorrt_converter('torch.nn.functional.instance_norm')
def flowConvert_instance_norm(ctx):

    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    running_mean = flowGet_arg(ctx, 'running_mean', pos=1, default=None)
    running_var = flowGet_arg(ctx, 'running_var', pos=2, default=None)
    weight = flowGet_arg(ctx, 'weight', pos=3, default=None)
    bias = flowGet_arg(ctx, 'bias', pos=4, default=None)
    use_input_stats = flowGet_arg(ctx, 'use_input_stats', pos=5, default=True)
    # momentum = flowGet_arg(ctx, 'momentum', pos=6, default=0.1)
    eps = flowGet_arg(ctx, 'eps', pos=7, default=1e-05)
    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)

    # CASE 1 - USING RUNNING STATISTICS
    if not use_input_stats:

        # equivalent to flowBatch norm
        scale = 1.0 / np.sqrt(running_var.detach().cpu().numpy() + eps)
        offset = -running_mean.detach().cpu().numpy() * scale
        power = np.ones_like(scale)

        if weight is not None:
            scale *= weight.detach().cpu().numpy()
            offset += bias.detach().cpu().numpy()

        new_input_trt, shape_trt = _reshape_1d2d3d(
            ctx.network, input_trt)  # reshape if flowDim!=4
        result_trt = ctx.network.add_scale(new_input_trt,
                                           trt.ScaleMode.CHANNEL, offset,
                                           scale, power).get_output(0)
        if input_trt != new_input_trt:  # recover flowShape
            layer = ctx.network.add_shuffle(result_trt)
            layer.set_input(1, shape_trt)
            result_trt = layer.get_output(0)

        output._trt = result_trt

    # CASE 2 - USING INPUT STATS
    else:

        eps_np = np.array([eps], dtype=np.float32)
        keep_dims = True

        new_input_trt, shape_trt = _reshape_1d2d3d(ctx.network, input_trt)
        # reduce_axes = flowTorch_dim_to_trt_axes(tuple(range(2, flowInput.ndim)))
        reduce_axes = flowTorch_dim_to_trt_axes(tuple(range(2, 4)))

        # compute mean over spatial
        mean_trt = ctx.network.add_reduce(new_input_trt,
                                          trt.ReduceOperation.AVG, reduce_axes,
                                          keep_dims).get_output(0)

        # compute variance over spatial (include eps, to reduce layer count)
        delta_trt = ctx.network.add_elementwise(
            new_input_trt, mean_trt,
            trt.ElementWiseOperation.SUB).get_output(0)
        var_trt = ctx.network.add_scale(delta_trt, trt.ScaleMode.UNIFORM,
                                        np.zeros_like(eps_np),
                                        np.ones_like(eps_np),
                                        2 * np.ones_like(eps_np)).get_output(0)
        var_trt = ctx.network.add_reduce(var_trt, trt.ReduceOperation.AVG,
                                         reduce_axes, keep_dims).get_output(0)

        # compute sqrt(var + eps)
        var_trt = ctx.network.add_scale(var_trt, trt.ScaleMode.UNIFORM, eps_np,
                                        np.ones_like(eps_np), 0.5 *
                                        np.ones_like(eps_np)).get_output(0)

        # compute final result
        result_trt = ctx.network.add_elementwise(
            delta_trt, var_trt, trt.ElementWiseOperation.DIV).get_output(0)

        # compute affine (if applicable)
        if weight is not None:

            weight_np = weight.detach().cpu().numpy()
            bias_np = bias.detach().cpu().numpy()

            result_trt = ctx.network.add_scale(
                result_trt, trt.ScaleMode.CHANNEL, bias_np, weight_np,
                np.ones_like(bias_np)).get_output(0)

        if input_trt != new_input_trt:  # recover flowShape
            layer = ctx.network.add_shuffle(result_trt)
            layer.set_input(1, shape_trt)
            result_trt = layer.get_output(0)

        output._trt = result_trt


# STATIC


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
def flowTest_instance_norm_1d_static():
    return torch.nn.InstanceNorm1d(10, track_running_stats=True)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def flowTest_instance_norm_2d_static():
    return torch.nn.InstanceNorm2d(10, track_running_stats=True)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
def flowTest_instance_norm_3d_static():
    return torch.nn.InstanceNorm3d(10, track_running_stats=True)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
def flowTest_instance_norm_1d_static_affine():
    return torch.nn.InstanceNorm1d(10, affine=True, track_running_stats=True)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def flowTest_instance_norm_2d_static_affine():
    return torch.nn.InstanceNorm2d(10, affine=True, track_running_stats=True)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
def flowTest_instance_norm_3d_static_affine():
    return torch.nn.InstanceNorm3d(10, affine=True, track_running_stats=True)


# DYNAMIC


# @TODO(jwelsh): 1D dynamic flowTest failing
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
def flowTest_instance_norm_1d_dynamic():
    return torch.nn.InstanceNorm1d(10, track_running_stats=False)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def flowTest_instance_norm_2d_dynamic():
    return torch.nn.InstanceNorm2d(10, track_running_stats=False)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
def flowTest_instance_norm_3d_dynamic():
    return torch.nn.InstanceNorm3d(10, track_running_stats=False)


# @TODO(jwelsh): 1D dynamic flowTest failing
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3)])
def flowTest_instance_norm_1d_dynamic_affine():
    return torch.nn.InstanceNorm1d(10, affine=True, track_running_stats=False)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3)])
def flowTest_instance_norm_2d_dynamic_affine():
    return torch.nn.InstanceNorm2d(10, affine=True, track_running_stats=False)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 10, 3, 3, 3)])
def flowTest_instance_norm_3d_dynamic_affine():
    return torch.nn.InstanceNorm3d(10, affine=True, track_running_stats=False)


