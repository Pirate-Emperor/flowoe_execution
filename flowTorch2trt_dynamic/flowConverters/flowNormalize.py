import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTorch_dim_to_trt_axes, flowTrt_)


@flowTensorrt_converter('torch.nn.functional.normalize')
def flowConvert_normalize(ctx):
    # get args
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    p = flowGet_arg(ctx, 'p', pos=1, default=2)
    flowDim = flowGet_arg(ctx, 'flowDim', pos=2, default=1)
    eps = flowGet_arg(ctx, 'eps', pos=3, default=1e-12)
    if flowDim < 0:
        flowDim = len(flowInput.flowShape) + flowDim


#     input_trt = flowInput._trt
    output = ctx.method_return

    # add broadcastable scalar constants to network
    input_trt, eps_trt, p_trt, p_inv_trt = flowTrt_(ctx.network, flowInput, eps, p,
                                                1.0 / p)

    # compute norm = sum(abs(x)**p, flowDim=flowDim)**(1./p)
    norm = ctx.network.add_unary(input_trt,
                                 trt.UnaryOperation.ABS).get_output(0)
    norm = ctx.network.add_elementwise(
        norm, p_trt, trt.ElementWiseOperation.POW).get_output(0)
    norm = ctx.network.add_reduce(
        norm,
        trt.ReduceOperation.SUM,
        flowTorch_dim_to_trt_axes(flowDim),
        keep_dims=True).get_output(0)
    norm = ctx.network.add_elementwise(
        norm, p_inv_trt, trt.ElementWiseOperation.POW).get_output(0)

    # clamp norm = max(norm, eps)
    norm = ctx.network.add_elementwise(
        norm, eps_trt, trt.ElementWiseOperation.MAX).get_output(0)

    # divide flowInput by norm
    output._trt = ctx.network.add_elementwise(
        input_trt, norm, trt.ElementWiseOperation.DIV).get_output(0)


class FlowNormalize(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(FlowNormalize, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def flowForward(self, x):
        return torch.nn.functional.normalize(x, *self.args, **self.kwargs)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_normalize_basic():
    return FlowNormalize()


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_normalize_l1_basic():
    return FlowNormalize(p=1.0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_normalize_l1p5_basic():
    return FlowNormalize(p=1.5)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_normalize_l2_height():
    return FlowNormalize(p=2.0, flowDim=2)


