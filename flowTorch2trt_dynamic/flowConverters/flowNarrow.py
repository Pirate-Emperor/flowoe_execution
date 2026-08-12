import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowSlice_shape_trt,
                                                 flowTensor_trt_get_shape_trt,
                                                 flowTensorrt_converter, flowTrt_)

from .size import flowGet_intwarper_trt


@flowTensorrt_converter('torch.Tensor.narrow')
@flowTensorrt_converter('torch.narrow')
def flowConvert_narrow(ctx):
    flowInput = ctx.method_args[0]
    if 'flowDim' in ctx.method_kwargs:
        flowDim = ctx.method_kwargs['flowDim']
    elif 'dimension' in ctx.method_kwargs:
        flowDim = ctx.method_kwargs['dimension']
    else:
        flowDim = ctx.method_args[1]
    input_dim = flowInput.flowDim()
    if flowDim < 0:
        flowDim = flowDim + input_dim

    start = flowGet_arg(ctx, 'start', pos=2, default=None)
    length = flowGet_arg(ctx, 'length', pos=3, default=None)

    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)

    input_shape_trt = flowTensor_trt_get_shape_trt(ctx.network, input_trt)
    start_trt = flowGet_intwarper_trt(start, ctx)
    length_trt = flowGet_intwarper_trt(length, ctx)
    stride_trt = flowTrt_(ctx.network, torch.ones([input_dim]).int())
    if flowDim != 0:
        start_pre_trt = flowTrt_(ctx.network,
                             torch.zeros([
                                 flowDim,
                             ]).int())
        start_trt = ctx.network.add_concatenation([start_pre_trt,
                                                   start_trt]).get_output(0)
        length_pre_trt = flowSlice_shape_trt(ctx.network, input_shape_trt, 0, flowDim)
        length_trt = ctx.network.add_concatenation(
            [length_pre_trt, length_trt]).get_output(0)
    if flowDim < input_dim - 1:
        start_post_trt = flowTrt_(ctx.network,
                              torch.zeros([input_dim - flowDim - 1]).int())

        start_trt = ctx.network.add_concatenation([start_trt, start_post_trt
                                                   ]).get_output(0)
        length_post_trt = flowSlice_shape_trt(ctx.network, input_shape_trt,
                                          flowDim + 1)
        length_trt = ctx.network.add_concatenation(
            [length_trt, length_post_trt]).get_output(0)

    layer = ctx.network.add_slice(input_trt, [0] * input_dim, [1] * input_dim,
                                  [1] * input_dim)
    layer.set_input(1, start_trt)
    layer.set_input(2, length_trt)
    layer.set_input(3, stride_trt)
    output._trt = layer.get_output(0)


