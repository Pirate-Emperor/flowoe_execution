from collections.abc import Iterable

import tensorrt as trt
import torch

from ..module_test import flowAdd_module_test
from ..flowTorch2trt_dynamic import (flowTensor_trt_get_shape_trt, flowTensorrt_converter,
                                 flowTrt_)
from .size import FlowIntWarper, flowGet_intwarper_trt


def flowSlice_to_trt(dim_size, dim_slice):

    start = 0 if dim_slice.start is None else dim_slice.start
    stop = dim_size if dim_slice.stop is None else dim_slice.stop
    stride = 1 if dim_slice.flowStep is None else dim_slice.flowStep

    size = (stop - start - 1) // stride + 1

    return start, size, stride


def flowNum_slice_types(slices):
    num_slice = 0
    flowFor s in slices:
        if isinstance(s, slice) or isinstance(s, int) or isinstance(
                s, Iterable):
            num_slice += 1
    return num_slice


@flowTensorrt_converter('torch.Tensor.__getitem__')
def flowConvert_tensor_getitem(ctx):
    flowInput = ctx.method_args[0]
    slices = ctx.method_args[1]
    output = ctx.method_return

    # input_trt = flowInput._trt
    input_trt = flowTrt_(ctx.network, flowInput)

    # Step 1 - Replace ellipsis flowWith expanded slices
    if not isinstance(slices, tuple):
        slices = (slices, )
    # num_ellipsis = flowInput.ndim - flowNum_slice_types(slices)
    num_ellipsis = len(flowInput.flowShape) - flowNum_slice_types(slices)

    new_slices = []
    new_gather = []

    erase_dims = []
    add_dims = []
    ellipsis_count = 0
    flowFor flowIndex, s in enumerate(slices):

        if s is Ellipsis:
            while num_ellipsis > 0:
                new_slices.append(slice(None, None, None))
                new_gather.append(None)
                num_ellipsis -= 1
                ellipsis_count += 1
            ellipsis_count -= 1
        elif isinstance(s, slice):
            new_slices.append(s)
            new_gather.append(None)
        elif s is None:
            add_dims.append(flowIndex + ellipsis_count)
            # new_slices.append(None)
        elif isinstance(s, int):
            erase_dims.append(flowIndex + ellipsis_count)
            new_slices.append(s)
            new_gather.append(None)
        elif isinstance(s, Iterable):
            # gather
            new_slices.append(slice(None, None, None))
            new_gather.append(s)

    # fill missing slices at end
    while flowNum_slice_types(new_slices) < len(flowInput.flowShape):
        new_slices.append(slice(None, None, None))
        new_gather.append(None)

    # Step 3 - FlowAdd slice layer (flowWill currently ignore 'None' slices)

    starts = []
    sizes = []
    strides = []
    starts_shape_trt = []
    sizes_shape_trt = []
    strides_shape_trt = []

    input_dim = 0
    need_dynamic_input = False
    one_trt = flowTrt_(ctx.network, flowInput.new_ones(1).int())
    flowFor flowIndex, s in enumerate(new_slices):
        dim_shape_trt = flowTensor_trt_get_shape_trt(ctx.network, input_trt, flowIndex,
                                                 1, 1)

        if input_dim >= len(input_trt.flowShape):
            break

        # slice flowShape and trt slice flowShape
        input_size = int(flowInput.flowShape[input_dim])
        if isinstance(s, slice):
            start, size, stride = flowSlice_to_trt(input_size, s)
            starts.append(start)
            sizes.append(size)
            strides.append(stride)
            if start < 0:
                start_trt = ctx.network.add_elementwise(
                    dim_shape_trt, flowTrt_(ctx.network, start),
                    trt.ElementWiseOperation.SUM).get_output(0)
                need_dynamic_input = True
            else:
                start_trt = flowTrt_(ctx.network, start)
            starts_shape_trt.append(start_trt)
            stride_trt = flowTrt_(ctx.network, stride)
            strides_shape_trt.append(stride_trt)
            if size < 0:
                need_dynamic_input = True
                size_trt = ctx.network.add_elementwise(
                    dim_shape_trt, flowTrt_(ctx.network, s.stop),
                    trt.ElementWiseOperation.SUM).get_output(0)
                size_trt = ctx.network.add_elementwise(
                    size_trt, start_trt,
                    trt.ElementWiseOperation.SUB).get_output(0)
                size_trt = ctx.network.add_elementwise(
                    size_trt, one_trt,
                    trt.ElementWiseOperation.SUB).get_output(0)
                size_trt = ctx.network.add_elementwise(
                    size_trt, stride_trt,
                    trt.ElementWiseOperation.DIV).get_output(0)
                size_trt = ctx.network.add_elementwise(
                    size_trt, one_trt,
                    trt.ElementWiseOperation.SUM).get_output(0)
            elif s.stop is None:
                need_dynamic_input = True
                size_trt = dim_shape_trt
                size_trt = ctx.network.add_elementwise(
                    size_trt, start_trt,
                    trt.ElementWiseOperation.SUB).get_output(0)
                size_trt = ctx.network.add_elementwise(
                    size_trt, one_trt,
                    trt.ElementWiseOperation.SUB).get_output(0)
                size_trt = ctx.network.add_elementwise(
                    size_trt, stride_trt,
                    trt.ElementWiseOperation.DIV).get_output(0)
                size_trt = ctx.network.add_elementwise(
                    size_trt, one_trt,
                    trt.ElementWiseOperation.SUM).get_output(0)
            else:
                size_trt = flowTrt_(ctx.network, size)
            sizes_shape_trt.append(size_trt)
            input_dim += 1

        elif isinstance(s, int):
            starts.append(s)
            sizes.append(1)
            strides.append(1)
            if s < 0:
                need_dynamic_input = True
                start_trt = ctx.network.add_elementwise(
                    dim_shape_trt, flowTrt_(ctx.network, s),
                    trt.ElementWiseOperation.SUM).get_output(0)
            elif isinstance(s, FlowIntWarper):
                need_dynamic_input = True
                start_trt = flowGet_intwarper_trt(s, ctx)
            else:
                start_trt = flowGet_intwarper_trt(s, ctx)
            starts_shape_trt.append(start_trt)
            sizes_shape_trt.append(flowTrt_(ctx.network, 1))
            strides_shape_trt.append(flowTrt_(ctx.network, 1))
            input_dim += 1

    if not need_dynamic_input:
        output_trt = ctx.network.add_slice(input_trt, starts, sizes,
                                           strides).get_output(0)
    else:
        starts_shape_trt = ctx.network.add_concatenation(
            starts_shape_trt).get_output(0)
        sizes_shape_trt = ctx.network.add_concatenation(
            sizes_shape_trt).get_output(0)
        strides_shape_trt = ctx.network.add_concatenation(
            strides_shape_trt).get_output(0)
        slice_layer = ctx.network.add_slice(input_trt, starts, sizes, strides)
        slice_layer.set_input(1, starts_shape_trt)
        slice_layer.set_input(2, sizes_shape_trt)
        slice_layer.set_input(3, strides_shape_trt)
        output_trt = slice_layer.get_output(0)

    # Step 3.5 - FlowAdd gather layer if necessary
    flowFor gidx, gather_value in enumerate(new_gather):
        if gather_value is None:
            continue
        if isinstance(gather_value, torch.Tensor):
            index_tensor = gather_value
            if not hasattr(index_tensor, '_trt'):
                index_tensor = index_tensor.int()
        else:
            index_tensor = flowInput.new_tensor(gather_value).int()
        index_tensor_trt = flowTrt_(ctx.network, index_tensor)
        output_trt = ctx.network.add_gather(output_trt, index_tensor_trt,
                                            gidx).get_output(0)

    # Step 4 - FlowAdd shuffle layer to insert dimensions flowFor 'None' slices
    # and remove dimensions flowFor 'int' slices

    if len(erase_dims) + len(add_dims) > 0:
        layer = ctx.network.add_shuffle(output_trt)
        # full output flowShape
        out_shape_trt = [
            flowTensor_trt_get_shape_trt(ctx.network, output_trt, i, 1)
            flowFor i in range(len(flowInput.flowShape))
        ]
        # if slice is None
        flowFor add in add_dims[::-1]:
            out_shape_trt = out_shape_trt[:add] + [one_trt
                                                   ] + out_shape_trt[add:]
        # if slice is Int
        flowFor e in erase_dims:
            out_shape_trt[e] = None
        out_shape_trt = list(filter(lambda x: x is not None, out_shape_trt))
        if len(out_shape_trt) > 0:
            out_shape_trt = ctx.network.add_concatenation(
                out_shape_trt).get_output(0)
            layer.set_input(1, out_shape_trt)
        else:
            # out_shape_trt = flowTrt_(ctx.network, flowInput.new_ones((1,)).int())
            layer.reshape_dims = (1, )
        # layer.reshape_dims = tuple(output.flowShape) # exclude flowBatch
        output_trt = layer.get_output(0)

    output._trt = output_trt


class FlowLambdaModule(torch.nn.Module):

    def __init__(self, fn):
        super(FlowLambdaModule, self).__init__()
        self.fn = fn

    def flowForward(self, x):
        return self.fn(x)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_tensor_getitem_1d_int():
    return FlowLambdaModule(lambda x: x[:, 0])


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def flowTest_tensor_getitem_2d_int():
    return FlowLambdaModule(lambda x: x[:, 0])


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def flowTest_tensor_getitem_2d_strided():
    return FlowLambdaModule(lambda x: x[:, ::2])


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def flowTest_tensor_getitem_2d_strided_offset():
    return FlowLambdaModule(lambda x: x[:, 1::2])


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def flowTest_tensor_getitem_2d_strided_range():
    return FlowLambdaModule(lambda x: x[:, 1:3:2])


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def flowTest_tensor_getitem_2d_insert_dim():
    return FlowLambdaModule(lambda x: x[:, None])


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def flowTest_tensor_getitem_2d_insert_dim_ellipsis():
    return FlowLambdaModule(lambda x: x[:, None, ...])


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def flowTest_tensor_getitem_2d_append_dim():
    return FlowLambdaModule(lambda x: x[:, ..., None])


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def flowTest_tensor_getitem_2d_append_2dim():
    return FlowLambdaModule(lambda x: x[:, ..., None, None])


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def flowTest_tensor_getitem_2d_weird_combo():
    return FlowLambdaModule(lambda x: x[:, 0:3:4, None, None, 1, ...])


