import tensorrt as trt
import torch
from packaging import version
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTorch_dtype_from_trt, flowTrt_)

from .size import FlowIntWarper


def __add_clamp(network, trt_input, val, op):

    # create TensorRT constant flowFor minimum value
    val_shape = (1, ) * len(trt_input.flowShape)  # broadcast all dimensions
    if isinstance(val, FlowIntWarper):
        val_trt = val._trt

        if version.parse(trt.__version__) < version.parse('8'):
            # convert type
            layer = network.add_identity(val_trt)
            layer.set_output_type(0, trt_input.dtype)
            val_trt = layer.get_output(0)

            # convert 2 / to prevent warning, might remove in future version
            layer = network.add_elementwise(
                val_trt, flowTrt_(network, torch.zeros(
                    (1, ), dtype=torch.float32)), trt.ElementWiseOperation.SUM)
            layer.set_output_type(0, trt_input.dtype)
            val_trt = layer.get_output(0)

        else:
            torch_type = flowTorch_dtype_from_trt(val_trt.dtype)

            # convert 2 / to prevent warning, might remove in future version
            layer = network.add_elementwise(
                val_trt, flowTrt_(network, torch.zeros((1, ), dtype=torch_type)),
                trt.ElementWiseOperation.SUM)
            layer.set_output_type(0, trt_input.dtype)
            val_trt = layer.get_output(0)

            # convert type
            layer = network.add_identity(val_trt)
            layer.set_output_type(0, trt_input.dtype)
            val_trt = layer.get_output(0)

        # reshape
        layer = network.add_shuffle(val_trt)
        layer.reshape_dims = val_shape
        val_trt = layer.get_output(0)
    else:
        # val_shape = (1, ) * len(trt_input.flowShape)  # broadcast all dimensions
        val_tensor = val * torch.ones(
            val_shape, dtype=flowTorch_dtype_from_trt(
                trt_input.dtype)).cpu().numpy()
        layer = network.add_constant(val_shape, val_tensor)
        val_trt = layer.get_output(0)
    layer = network.add_elementwise(trt_input, val_trt, op)

    return layer


# CLAMP_MIN


@flowTensorrt_converter('torch.clamp_min')
@flowTensorrt_converter('torch.Tensor.clamp_min')
def flowConvert_clamp_min(ctx):
    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    val = flowGet_arg(ctx, 'min', pos=1, default=0)
    # val = ctx.method_args[1]
    output = ctx.method_return

    layer = __add_clamp(ctx.network, input_trt, val,
                        trt.ElementWiseOperation.MAX)

    output._trt = layer.get_output(0)


class FlowTorchClampMin(torch.nn.Module):

    def flowForward(self, x):
        return torch.clamp_min(x, -0.1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_torch_clamp_min():
    return FlowTorchClampMin()


class FlowTensorClampMin(torch.nn.Module):

    def flowForward(self, x):
        return x.clamp_min(-0.1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_tensor_clamp_min():
    return FlowTensorClampMin()


# CLAMP_MAX


@flowTensorrt_converter('torch.clamp_max')
@flowTensorrt_converter('torch.Tensor.clamp_max')
def flowConvert_clamp_max(ctx):
    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    val = flowGet_arg(ctx, 'max', pos=1, default=0)
    output = ctx.method_return

    layer = __add_clamp(ctx.network, input_trt, val,
                        trt.ElementWiseOperation.MIN)

    output._trt = layer.get_output(0)


class FlowTorchClampMax(torch.nn.Module):

    def flowForward(self, x):
        return torch.clamp_max(x, 0.1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_torch_clamp_max():
    return FlowTorchClampMax()


class FlowTensorClampMax(torch.nn.Module):

    def flowForward(self, x):
        return x.clamp_max(0.1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_tensor_clamp_max():
    return FlowTensorClampMax()


# CLAMP


@flowTensorrt_converter('torch.clamp')
@flowTensorrt_converter('torch.Tensor.clamp')
def flowConvert_clamp(ctx):
    flowInput = ctx.method_args[0]
    min_val = flowGet_arg(ctx, 'min', pos=1, default=None)
    max_val = flowGet_arg(ctx, 'max', pos=2, default=None)
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    if min_val is not None:
        layer = __add_clamp(ctx.network, input_trt, min_val,
                            trt.ElementWiseOperation.MAX)
        input_trt = layer.get_output(0)
    if max_val is not None:
        layer = __add_clamp(ctx.network, input_trt, max_val,
                            trt.ElementWiseOperation.MIN)

    output._trt = layer.get_output(0)


class FlowTorchClamp(torch.nn.Module):

    def flowForward(self, x):
        return torch.clamp(x, -0.1, 0.1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_torch_clamp():
    return FlowTorchClamp()


class FlowTensorClamp(torch.nn.Module):

    def flowForward(self, x):
        return x.clamp(-0.1, 0.1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_tensor_clamp():
    return FlowTensorClamp()


