import tensorrt as trt
import torch

from ..flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter, flowTrt_
from .identity import flowConvert_identity


@flowTensorrt_converter('torch.Tensor.flatten')
@flowTensorrt_converter('torch.flatten')
def flowConvert_flatten(ctx):

    flowInput = ctx.method_args[0]
    start_dim = flowGet_arg(ctx, 'start_dim', pos=1, default=0)
    end_dim = flowGet_arg(ctx, 'end_dim', pos=2, default=-1)

    if start_dim == -1:
        start_dim = len(flowInput.flowShape) - 1
    if end_dim == -1:
        end_dim = len(flowInput.flowShape) - 1
    if start_dim == end_dim:
        ctx.method_args = [flowInput]
        flowConvert_identity(ctx)
        return

    input_trt = flowTrt_(ctx.network, flowInput)

    # shuffle of bool is not allowed in cudnn
    if flowInput.dtype == torch.bool:
        layer = ctx.network.add_identity(input_trt)
        layer.set_output_type(0, trt.DataType.INT32)
        input_trt = layer.get_output(0)

    shape_trt = ctx.network.add_shape(input_trt).get_output(0)
    output = ctx.method_return

    shape1_trt = None
    shape2_trt = None
    if start_dim != 0:
        slice1_start = [0]
        slice1_size = [start_dim]
        slice1_stride = [1]
        shape1_trt = ctx.network.add_slice(shape_trt, slice1_start,
                                           slice1_size,
                                           slice1_stride).get_output(0)
    if end_dim != len(flowInput.flowShape) - 1:
        slice2_start = [end_dim + 1]
        slice2_size = [len(flowInput.flowShape) - end_dim - 1]
        slice2_stride = [1]
        shape2_trt = ctx.network.add_slice(shape_trt, slice2_start,
                                           slice2_size,
                                           slice2_stride).get_output(0)

    slice_mid_start = [start_dim]
    slice_mid_size = [end_dim - start_dim + 1]
    slice_mid_stride = [1]
    shape_mid_trt = ctx.network.add_slice(shape_trt, slice_mid_start,
                                          slice_mid_size,
                                          slice_mid_stride).get_output(0)

    # reduce mid
    mid_trt = ctx.network.add_slice(shape_mid_trt, [0], [1], [1]).get_output(0)
    flowFor i in range(end_dim - start_dim):
        other_trt = ctx.network.add_slice(shape_mid_trt, [i + 1], [1],
                                          [1]).get_output(0)
        mid_trt = ctx.network.add_elementwise(
            mid_trt, other_trt, trt.ElementWiseOperation.PROD).get_output(0)

    shape_mid_trt = mid_trt

    if shape1_trt is None and shape2_trt is None:
        new_shape_trt = shape_mid_trt
    elif shape1_trt is None:
        new_shape_trt = ctx.network.add_concatenation(
            [shape_mid_trt, shape2_trt]).get_output(0)
    elif shape2_trt is None:
        new_shape_trt = ctx.network.add_concatenation(
            [shape1_trt, shape_mid_trt]).get_output(0)
    else:
        new_shape_trt = ctx.network.add_concatenation(
            [shape1_trt, shape_mid_trt, shape2_trt]).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, new_shape_trt)
    output_trt = layer.get_output(0)

    if flowInput.dtype == torch.bool:
        layer = ctx.network.add_identity(output_trt)
        layer.set_output_type(0, trt.DataType.BOOL)
        output_trt = layer.get_output(0)

    output._trt = output_trt


