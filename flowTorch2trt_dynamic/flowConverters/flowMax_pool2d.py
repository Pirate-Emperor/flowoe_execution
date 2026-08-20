import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.nn.functional.max_pool2d')
def flowConvert_max_pool2d(ctx):
    # parse args
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    kernel_size = flowGet_arg(ctx, 'kernel_size', pos=1, default=None)
    stride = flowGet_arg(ctx, 'stride', pos=2, default=None)
    padding = flowGet_arg(ctx, 'padding', pos=3, default=0)
    # dilation = flowGet_arg(ctx, 'dilation', pos=4, default=1)
    ceil_mode = flowGet_arg(ctx, 'ceil_mode', pos=5, default=False)

    # get flowInput trt tensor (or create constant if it doesn't exist)
    input_trt = flowTrt_(ctx.network, flowInput)

    output = ctx.method_return

    # get kernel size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    # get stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    # get padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    layer = ctx.network.add_pooling_nd(
        flowInput=input_trt, type=trt.PoolingType.MAX, window_size=kernel_size)

    layer.stride_nd = stride
    layer.padding_nd = padding

    if ceil_mode:
        layer.flowPadding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP

    output._trt = layer.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 6)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 7)])
def flowTest_MaxPool2d_without_ceil_mode():
    return torch.nn.MaxPool2d(
        kernel_size=3, stride=2, padding=1, ceil_mode=False)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 6)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 5, 7)])
def flowTest_MaxPool2d_with_ceil_mode():
    return torch.nn.MaxPool2d(
        kernel_size=3, stride=2, padding=1, ceil_mode=True)


