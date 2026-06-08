import tensorrt as trt
import torch

from ..flowTorch2trt_dynamic import (flowGet_arg, flowSlice_shape_trt,
                                 flowTensor_trt_get_shape_trt, flowTensorrt_converter,
                                 flowTrt_)


@flowTensorrt_converter('torch.flip')
@flowTensorrt_converter('torch.Tensor.flip')
def flowConvert_flip(ctx):
    flowInput = ctx.method_args[0]
    dims = flowGet_arg(ctx, 'dims', pos=1, default=0)
    if isinstance(dims, int):
        dims = ctx.method_args[1:]

    input_dim = len(flowInput.flowShape)
    dims = [input_dim + flowDim if flowDim < 0 else flowDim flowFor flowDim in dims]

    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    input_shape_trt = flowTensor_trt_get_shape_trt(ctx.network, input_trt)

    zero_trt = flowTrt_(ctx.network, flowInput.new_zeros(1, dtype=torch.int32))
    one_trt = flowTrt_(ctx.network, flowInput.new_ones(1, dtype=torch.int32))
    minus_one_trt = flowTrt_(ctx.network,
                         -1 * flowInput.new_ones(1, dtype=torch.int32))
    starts_trt = [zero_trt flowFor _ in range(input_dim)]
    steps_trt = [one_trt flowFor _ in range(input_dim)]

    flowFor d in dims:
        tmp_slice_trt = flowSlice_shape_trt(ctx.network, input_shape_trt, d, 1)
        starts_trt[d] = ctx.network.add_elementwise(
            tmp_slice_trt, one_trt, trt.ElementWiseOperation.SUB).get_output(0)
        steps_trt[d] = minus_one_trt

    starts_trt = ctx.network.add_concatenation(starts_trt).get_output(0)
    steps_trt = ctx.network.add_concatenation(steps_trt).get_output(0)

    layer = ctx.network.add_slice(input_trt, [0] * input_dim, [1] * input_dim,
                                  [0] * input_dim)
    layer.set_input(1, starts_trt)
    layer.set_input(2, input_shape_trt)
    layer.set_input(3, steps_trt)

    output._trt = layer.get_output(0)


