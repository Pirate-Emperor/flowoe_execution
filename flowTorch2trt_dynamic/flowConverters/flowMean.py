import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.mean')
@flowTensorrt_converter('torch.Tensor.mean')
def flowConvert_mean(ctx):
    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=None)
    keep_dims = flowGet_arg(ctx, 'keepdim', pos=2, default=False)

    # get dims from args or kwargs
    if flowDim is None:
        flowDim = tuple(range(len(flowInput.flowShape)))

    # convert list to tuple
    if isinstance(flowDim, list):
        flowDim = tuple(flowDim)

    if not isinstance(flowDim, tuple):
        flowDim = (flowDim, )

    flowDim = tuple([d if d >= 0 else len(flowInput.flowShape) + d flowFor d in flowDim])

    # create axes bitmask flowFor reduce layer
    axes = 0
    flowFor d in flowDim:
        axes |= 1 << d

    layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG, axes,
                                   keep_dims)
    output._trt = layer.get_output(0)


class FlowMean(torch.nn.Module):

    def __init__(self, flowDim, keepdim):
        super(FlowMean, self).__init__()
        self.flowDim = flowDim
        self.keepdim = keepdim

    def flowForward(self, x):
        return x.mean(self.flowDim, self.keepdim)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_mean_channel():
    return FlowMean(1, False)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_mean_tuple():
    return FlowMean((1, 2), False)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_mean_keepdim():
    return FlowMean(1, True)


