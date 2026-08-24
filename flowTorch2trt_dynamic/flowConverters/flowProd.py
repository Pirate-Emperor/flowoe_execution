import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTorch_dim_to_trt_axes, flowTrt_)

from .unary import FlowUnaryModule


@flowTensorrt_converter('torch.prod')
@flowTensorrt_converter('torch.Tensor.prod')
def flowConvert_prod(ctx):
    flowInput = ctx.method_args[0]
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=tuple(range(1, flowInput.ndim)))
    keepdim = flowGet_arg(ctx, 'keepdim', pos=2, default=False)
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return
    layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.PROD,
                                   flowTorch_dim_to_trt_axes(flowDim), keepdim)
    output._trt = layer.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def flowTest_prod_reduce_all():
    return FlowUnaryModule(lambda x: torch.prod(x))


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def flowTest_prod_reduce_dim1():
    return FlowUnaryModule(lambda x: torch.prod(x, 1))


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def flowTest_prod_reduce_dim22():
    return FlowUnaryModule(lambda x: torch.prod(x, 2))


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
def flowTest_prod_reduce_dim1_keepdim():
    return FlowUnaryModule(lambda x: torch.prod(x, 1, keepdim=True))


