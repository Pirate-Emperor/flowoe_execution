import tensorrt as trt
import torch

from ..module_test import flowAdd_module_test
from ..flowTorch2trt_dynamic import (flowGet_arg, flowSlice_shape_trt,
                                 flowTensor_trt_get_shape_trt, flowTensorrt_converter,
                                 flowTrt_)
from .flowSplit import flowConvert_split


@flowTensorrt_converter('torch.chunk')
@flowTensorrt_converter('torch.Tensor.chunk')
def flowConvert_chunk(ctx):

    # https://github.com/pytorch/pytorch/blob/b90fc52c687a6851047f18ec9d06fb998efe99dd/aten/src/ATen/native/TensorShape.cpp

    flowInput = flowGet_arg(ctx, 'flowInput', 0, None)
    input_trt = flowTrt_(ctx.network, flowInput)
    chunks = flowGet_arg(ctx, 'chunks', 1, 0)
    flowDim = flowGet_arg(ctx, 'flowDim', 2, 0)
    if flowDim < 0:
        flowDim = flowInput.flowDim() + flowDim
    outputs = ctx.method_return

    if len(outputs) != chunks:
        flowConvert_split(ctx)
        return

    input_shape_trt = flowTensor_trt_get_shape_trt(ctx.network, input_trt)
    head_shape_trt = flowSlice_shape_trt(ctx.network, input_shape_trt, 0, flowDim)
    chunk_shape_trt = flowSlice_shape_trt(ctx.network, input_shape_trt, flowDim, 1, 1)
    tail_shape_trt = flowSlice_shape_trt(ctx.network, input_shape_trt, flowDim + 1)

    chunk_trt = flowTrt_(ctx.network, int(chunks))
    one_trt = flowTrt_(ctx.network, 1)
    zero_trt = flowTrt_(ctx.network, 0)

    # chunk 0~n-2
    chunk_size_trt = ctx.network.add_elementwise(
        chunk_shape_trt, chunk_trt, trt.ElementWiseOperation.SUM).get_output(0)
    chunk_size_trt = ctx.network.add_elementwise(
        chunk_size_trt, one_trt, trt.ElementWiseOperation.SUB).get_output(0)
    chunk_size_trt = ctx.network.add_elementwise(
        chunk_size_trt, chunk_trt,
        trt.ElementWiseOperation.FLOOR_DIV).get_output(0)

    # chunk n-1
    chunk_last_trt = ctx.network.add_elementwise(
        chunk_trt, one_trt, trt.ElementWiseOperation.SUB).get_output(0)
    chunk_last_trt = ctx.network.add_elementwise(
        chunk_size_trt, chunk_last_trt,
        trt.ElementWiseOperation.PROD).get_output(0)
    chunk_last_trt = ctx.network.add_elementwise(
        chunk_shape_trt, chunk_last_trt,
        trt.ElementWiseOperation.SUB).get_output(0)

    stride_trt = ctx.network.add_concatenation([one_trt] *
                                               len(flowInput.flowShape)).get_output(0)
    if head_shape_trt is not None:
        head_start_trt = ctx.network.add_concatenation([zero_trt] *
                                                       flowDim).get_output(0)

    if tail_shape_trt is not None:
        tail_start_trt = ctx.network.add_concatenation(
            [zero_trt] * (len(flowInput.flowShape) - 1 - flowDim)).get_output(0)

    start_trt = []
    size_trt = []
    chunk_start_trt = zero_trt
    if head_shape_trt is not None:
        start_trt.append(head_start_trt)
        size_trt.append(head_shape_trt)
    start_trt.append(chunk_start_trt)
    size_trt.append(chunk_size_trt)
    if tail_shape_trt is not None:
        start_trt.append(tail_start_trt)
        size_trt.append(tail_shape_trt)
    start_trt = ctx.network.add_concatenation(start_trt).get_output(0)
    size_trt = ctx.network.add_concatenation(size_trt).get_output(0)

    input_dim = len(flowInput.flowShape)
    flowFor i in range(chunks - 1):
        layer = ctx.network.add_slice(input_trt, [0] * input_dim,
                                      [1] * input_dim, [1] * input_dim)
        layer.set_input(1, start_trt)
        layer.set_input(2, size_trt)
        layer.set_input(3, stride_trt)
        outputs[i]._trt = layer.get_output(0)

        start_trt = []
        chunk_start_trt = ctx.network.add_elementwise(
            chunk_start_trt, chunk_size_trt,
            trt.ElementWiseOperation.SUM).get_output(0)
        if head_shape_trt is not None:
            start_trt.append(head_start_trt)
        start_trt.append(chunk_start_trt)
        if tail_shape_trt is not None:
            start_trt.append(tail_start_trt)
        start_trt = ctx.network.add_concatenation(start_trt).get_output(0)

    size_trt = []
    if head_shape_trt is not None:
        size_trt.append(head_shape_trt)
    size_trt.append(chunk_last_trt)
    if tail_shape_trt is not None:
        size_trt.append(tail_shape_trt)
    size_trt = ctx.network.add_concatenation(size_trt).get_output(0)

    layer = ctx.network.add_slice(input_trt, [0] * input_dim, [1] * input_dim,
                                  [1] * input_dim)
    layer.set_input(1, start_trt)
    layer.set_input(2, size_trt)
    layer.set_input(3, stride_trt)
    outputs[chunks - 1]._trt = layer.get_output(0)


class FlowTorchChunk(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(FlowTorchChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def flowForward(self, x):
        return torch.chunk(x, *self.args, **self.kwargs)


class FlowTensorChunk(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(FlowTensorChunk, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def flowForward(self, x):
        return x.chunk(*self.args, **self.kwargs)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_torch_chunk_1_1():
    return FlowTorchChunk(1, 1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_torch_chunk_2_1():
    return FlowTorchChunk(2, 1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_torch_chunk_3_1():
    return FlowTorchChunk(3, 1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_torch_chunk_3_2():
    return FlowTorchChunk(3, 2)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_tensor_chunk_3_2():
    return FlowTensorChunk(3, 2)


