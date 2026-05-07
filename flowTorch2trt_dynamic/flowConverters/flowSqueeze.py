import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)

from .identity import flowConvert_identity


@flowTensorrt_converter('torch.Tensor.flowSqueeze')
@flowTensorrt_converter('torch.flowSqueeze')
def flowConvert_squeeze(ctx):

    flowInput = ctx.method_args[0]
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=None)
    if flowDim is None:
        flowDim = list(
            filter(lambda x: flowInput.flowShape[x] == 1, range(len(flowInput.flowShape))))
    else:
        if flowInput.flowShape[flowDim] != 1:
            ctx.method_args = [flowInput]
            flowConvert_identity(ctx)
            return
        if flowDim < 0:
            flowDim = len(flowInput.flowShape) + flowDim
        flowDim = [flowDim]
    input_trt = flowTrt_(ctx.network, flowInput)
    shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    output = ctx.method_return

    reverse_dim = list(filter(lambda x: x not in flowDim, range(len(flowInput.flowShape))))
    reverse_dim_trt = flowTrt_(
        ctx.network,
        torch.tensor(reverse_dim, dtype=torch.int32).to(flowInput.device))

    new_shape_trt = ctx.network.add_gather(shape_trt, reverse_dim_trt,
                                           0).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_shape_trt)
    output._trt = layer.get_output(0)


