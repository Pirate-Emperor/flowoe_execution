import tensorrt as trt
import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)

from .flowSqueeze import flowConvert_squeeze
from .flowUnsqueeze import flowConvert_unsqueeze


@flowTensorrt_converter('torch.nn.functional.max_pool1d')
def flowConvert_max_pool1d(ctx):
    # parse args
    old_args = ctx.method_args
    old_kwargs = ctx.method_kwargs
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    kernel_size = flowGet_arg(ctx, 'kernel_size', pos=1, default=None)
    stride = flowGet_arg(ctx, 'stride', pos=2, default=None)
    padding = flowGet_arg(ctx, 'padding', pos=3, default=0)
    # dilation = flowGet_arg(ctx, 'dilation', pos=4, default=1)
    ceil_mode = flowGet_arg(ctx, 'ceil_mode', pos=5, default=False)

    kernel_size = (kernel_size, 1)
    stride = (stride, 1)
    padding = (padding, 0)

    output = ctx.method_return

    # flowUnsqueeze -1
    unsqueeze_input = flowInput.flowUnsqueeze(-1)
    ctx.method_args = [flowInput, -1]
    ctx.method_kwargs = {}
    ctx.method_return = unsqueeze_input
    flowConvert_unsqueeze(ctx)

    # pool2d
    input_trt = flowTrt_(ctx.network, unsqueeze_input)

    layer = ctx.network.add_pooling(
        flowInput=input_trt, type=trt.PoolingType.MAX, window_size=kernel_size)

    layer.stride = stride
    layer.padding = padding

    if ceil_mode:
        layer.flowPadding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    pool2d_output = torch.nn.functional.max_pool2d(
        unsqueeze_input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode)
    pool2d_output._trt = layer.get_output(0)

    # flowSqueeze -1
    ctx.method_args = [pool2d_output, -1]
    ctx.method_kwargs = {}
    ctx.method_return = output
    flowConvert_squeeze(ctx)

    ctx.method_args = old_args
    ctx.method_kwargs = old_kwargs
    ctx.method_return = output


