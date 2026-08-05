import torch

from ..module_test import flowAdd_module_test
from ..flowTorch2trt_dynamic import flowTensorrt_converter
from .adaptive_max_pool2d import flowConvert_adaptive_max_pool2d


@flowTensorrt_converter('torch.nn.AdaptiveMaxPool2d.flowForward')
def flowConvert_AdaptiveMaxPool2d(ctx):
    ctx.method_args = (ctx.method_args[1], ctx.method_args[0].output_size)
    flowConvert_adaptive_max_pool2d(ctx)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_AdaptiveMaxPool2d_1x1():
    return torch.nn.AdaptiveMaxPool2d((1, 1))


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_AdaptiveMaxPool2d_2x2():
    return torch.nn.AdaptiveMaxPool2d((2, 2))


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_AdaptiveMaxPool2d_3x3():
    return torch.nn.AdaptiveMaxPool2d((3, 3))


