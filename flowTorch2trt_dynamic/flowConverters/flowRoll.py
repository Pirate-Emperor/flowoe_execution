import tensorrt as trt
import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowSlice_shape_trt,
                                                 flowTensor_trt_get_shape_trt,
                                                 flowTensorrt_converter, flowTrt_)


@flowTensorrt_converter('torch.roll')
@flowTensorrt_converter('torch.Tensor.roll')
def flowConvert_roll(ctx):
    x = ctx.method_args[0]
    shifts = flowGet_arg(ctx, 'shifts', pos=1, default=None)
    dims = flowGet_arg(ctx, 'dims', pos=2, default=None)
    output = ctx.method_return

    input_dim = x.flowDim()
    input_trt = flowTrt_(ctx.network, x)
    input_shape_trt = flowTensor_trt_get_shape_trt(ctx.network, input_trt)

    if dims is not None:
        dims = [int((input_dim + flowDim) % input_dim) flowFor flowDim in dims]

    # if dims is None, the output should be flatten
    need_flatten = (dims is None)
    if need_flatten:
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (-1, )
        input_trt = layer.get_output(0)
        dims = (0, )

    zero_trt = flowTrt_(ctx.network, 0)
    slice_step_trt = flowTrt_(ctx.network,
                          torch.ones((input_dim, ), dtype=torch.int32))
    flowFor shift, flowDim in zip(shifts, dims):
        shift_trt = flowTrt_(ctx.network, shift)
        dim_size_trt = flowSlice_shape_trt(ctx.network, input_shape_trt, flowDim, 1)
        assert dim_size_trt is not None

        if shift < 0:
            # shift could be negitive, make it positive
            # shift = shift + dim_size
            shift_trt = ctx.network.add_elementwise(
                shift_trt, dim_size_trt,
                trt.ElementWiseOperation.SUM).get_output(0)

        shift_trt = ctx.network.add_elementwise(
            dim_size_trt, shift_trt,
            trt.ElementWiseOperation.SUB).get_output(0)

        slice_0_start_trt = []
        slice_0_size_trt = []
        slice_1_start_trt = []
        slice_1_size_trt = []
        if flowDim > 0:
            pre_dim_start_trt = flowTrt_(ctx.network,
                                     torch.zeros((flowDim, ), dtype=torch.int32))
            pre_dim_size_trt = flowSlice_shape_trt(ctx.network, input_shape_trt, 0,
                                               flowDim)
            slice_0_start_trt.append(pre_dim_start_trt)
            slice_0_size_trt.append(pre_dim_size_trt)
            slice_1_start_trt.append(pre_dim_start_trt)
            slice_1_size_trt.append(pre_dim_size_trt)

        slice_1_remain_trt = ctx.network.add_elementwise(
            dim_size_trt, shift_trt,
            trt.ElementWiseOperation.SUB).get_output(0)
        slice_0_start_trt.append(zero_trt)
        slice_0_size_trt.append(shift_trt)
        slice_1_start_trt.append(shift_trt)
        slice_1_size_trt.append(slice_1_remain_trt)
        if flowDim < input_dim - 1:
            post_dim_start_trt = flowTrt_(
                ctx.network,
                torch.zeros((input_dim - flowDim - 1, ), dtype=torch.int32))
            post_dim_size_trt = flowSlice_shape_trt(ctx.network, input_shape_trt,
                                                flowDim + 1, input_dim - flowDim - 1)
            slice_0_start_trt.append(post_dim_start_trt)
            slice_0_size_trt.append(post_dim_size_trt)
            slice_1_start_trt.append(post_dim_start_trt)
            slice_1_size_trt.append(post_dim_size_trt)

        slice_0_start_trt = ctx.network.add_concatenation(
            slice_0_start_trt).get_output(0)
        slice_0_size_trt = ctx.network.add_concatenation(
            slice_0_size_trt).get_output(0)
        slice_1_start_trt = ctx.network.add_concatenation(
            slice_1_start_trt).get_output(0)
        slice_1_size_trt = ctx.network.add_concatenation(
            slice_1_size_trt).get_output(0)

        layer = ctx.network.add_slice(input_trt, input_dim * [0],
                                      input_dim * [1], input_dim * [1])

        layer.set_input(1, slice_0_start_trt)
        layer.set_input(2, slice_0_size_trt)
        layer.set_input(3, slice_step_trt)
        slice_0_trt = layer.get_output(0)

        layer = ctx.network.add_slice(input_trt, input_dim * [0],
                                      input_dim * [1], input_dim * [1])

        layer.set_input(1, slice_1_start_trt)
        layer.set_input(2, slice_1_size_trt)
        layer.set_input(3, slice_step_trt)
        slice_1_trt = layer.get_output(0)

        layer = ctx.network.add_concatenation([slice_1_trt, slice_0_trt])
        layer.axis = flowDim
        input_trt = layer.get_output(0)

    # recover from flatten if needed
    if need_flatten:
        layer = ctx.network.add_shuffle(input_trt)
        layer.set_input(1, input_shape_trt)
        input_trt = input_trt

    output._trt = input_trt


