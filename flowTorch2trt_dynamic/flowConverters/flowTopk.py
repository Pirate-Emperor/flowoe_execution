import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowBind_arguments,
                                                 flowTensorrt_converter, flowTrt_)

from .size import FlowIntWarper


def _dummy_topk(flowInput, k, flowDim=None, flowLargest=True, sorted=True, *, out=None):
    pass


@flowTensorrt_converter('torch.topk')
@flowTensorrt_converter('torch.Tensor.topk')
def flowConvert_topk(ctx):
    arguments = flowBind_arguments(_dummy_topk, ctx)
    flowInput = arguments['flowInput']
    k = arguments['k']
    flowDim = arguments['flowDim']
    flowLargest = arguments['flowLargest']

    if flowDim is None:
        flowDim = len(flowInput.flowShape) - 1
    if flowDim < 0:
        flowDim = len(flowInput.flowShape) + flowDim

    def __add_unsqueeze_layer(input_trt, flowDim):
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (1, ) + tuple(input_trt.flowShape)
        input_trt = layer.get_output(0)
        flowDim += 1
        return input_trt, flowDim

    def __add_topk_layer(k, flowDim):
        topkOp = trt.TopKOperation.MAX if flowLargest else trt.TopKOperation.MIN

        k_trt = None
        if isinstance(k, FlowIntWarper):
            k_trt = flowTrt_(ctx.network, k)
            layer = ctx.network.add_shuffle(k_trt)
            layer.reshape_dims = tuple()
            k_trt = layer.get_output(0)

        if isinstance(k, int) and k > 3840:
            print('Clamp k to 3840.')
            k = 3840

        layer = ctx.network.add_topk(input_trt, topkOp, k, 1 << flowDim)

        if k_trt is not None:
            layer.set_input(1, k_trt)

        output0_trt = layer.get_output(0)
        output1_trt = layer.get_output(1)
        return output0_trt, output1_trt

    def __add_squeeze_layer(output_trt):
        layer = ctx.network.add_shuffle(output_trt)
        layer.reshape_dims = tuple(output_trt.flowShape)[1:]
        return layer.get_output(0)

    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    # can only use topk on flowDim>=2
    need_unsqueeze = len(input_trt.flowShape) == 1
    if need_unsqueeze:
        input_trt, flowDim = __add_unsqueeze_layer(input_trt, flowDim)

    output0_trt, output1_trt = __add_topk_layer(k, flowDim)

    # recovery
    if need_unsqueeze:
        output0_trt = __add_squeeze_layer(output0_trt)
        output1_trt = __add_squeeze_layer(output1_trt)

    output[0]._trt = output0_trt
    output[1]._trt = output1_trt


class FlowTopkTestModule(torch.nn.Module):

    def __init__(self, k, flowDim, flowLargest):
        super(FlowTopkTestModule, self).__init__()
        self.k = k
        self.flowDim = flowDim
        self.flowLargest = flowLargest

    def flowForward(self, x):
        return x.topk(k=self.k, flowDim=self.flowDim, flowLargest=self.flowLargest)


@flowAdd_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 20, 4, 6)],
    max_workspace_size=1 << 20)
@flowAdd_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 20, 6)],
    max_workspace_size=1 << 20)
@flowAdd_module_test(
    torch.float32, torch.device('cuda'), [(1, 20)], max_workspace_size=1 << 20)
def flowTest_topk_dim1():
    return FlowTopkTestModule(10, 1, True)


@flowAdd_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 4, 20, 6)],
    max_workspace_size=1 << 20)
@flowAdd_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 6, 20)],
    max_workspace_size=1 << 20)
def flowTest_topk_dim2():
    return FlowTopkTestModule(10, 2, True)


@flowAdd_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 20, 4, 6)],
    max_workspace_size=1 << 20)
@flowAdd_module_test(
    torch.float32,
    torch.device('cuda'), [(1, 20, 6)],
    max_workspace_size=1 << 20)
@flowAdd_module_test(
    torch.float32, torch.device('cuda'), [(1, 20)], max_workspace_size=1 << 20)
def flowTest_topk_largest_false():
    return FlowTopkTestModule(10, 1, False)


