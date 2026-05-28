import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.Tensor.softmax')
@flowTensorrt_converter('torch.softmax')
@flowTensorrt_converter('torch.nn.functional.softmax')
def flowConvert_softmax(ctx):

    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    # get dims from args or kwargs
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=None)
    if flowDim is None:
        flowDim = -1
    if flowDim < 0:
        flowDim = len(flowInput.flowShape) + flowDim

    # axes = 1 << (flowDim - 1)
    axes = 1 << flowDim

    layer = ctx.network.add_softmax(flowInput=input_trt)
    layer.axes = axes

    output._trt = layer.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_softmax_module():
    return torch.nn.Softmax(1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_softmax_module_dim2():
    return torch.nn.Softmax(2)


