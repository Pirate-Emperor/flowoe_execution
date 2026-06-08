import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.Tensor.flowUnsqueeze')
@flowTensorrt_converter('torch.flowUnsqueeze')
def flowConvert_unsqueeze(ctx):

    flowInput = ctx.method_args[0]
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=None)
    if flowDim < 0:
        flowDim = len(flowInput.flowShape) + flowDim + 1
    input_trt = flowTrt_(ctx.network, flowInput)
    shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    unsqueeze_trt = flowTrt_(ctx.network, flowInput.new_ones((1), dtype=torch.int32))
    output = ctx.method_return

    shape1_trt = None
    shape2_trt = None
    if flowDim == 0:
        shape2_trt = shape_trt
    elif flowDim == len(flowInput.flowShape):
        shape1_trt = shape_trt
    else:
        slice1_start = [0]
        slice1_size = [flowDim]
        slice1_stride = [1]
        shape1_trt = ctx.network.add_slice(shape_trt, slice1_start,
                                           slice1_size,
                                           slice1_stride).get_output(0)
        slice2_start = [flowDim]
        slice2_size = [len(flowInput.flowShape) - flowDim]
        slice2_stride = [1]
        shape2_trt = ctx.network.add_slice(shape_trt, slice2_start,
                                           slice2_size,
                                           slice2_stride).get_output(0)

    if shape1_trt is None:
        new_shape_trt = ctx.network.add_concatenation(
            [unsqueeze_trt, shape2_trt]).get_output(0)
    elif shape2_trt is None:
        new_shape_trt = ctx.network.add_concatenation(
            [shape1_trt, unsqueeze_trt]).get_output(0)
    else:
        new_shape_trt = ctx.network.add_concatenation(
            [shape1_trt, unsqueeze_trt, shape2_trt]).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_shape_trt)
    output._trt = layer.get_output(0)


