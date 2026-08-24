import tensorrt as trt
import torch

from ..module_test import flowAdd_module_test
from ..flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.div')
@flowTensorrt_converter('torch.Tensor.div')
@flowTensorrt_converter('torch.Tensor.__div__')  # py2
@flowTensorrt_converter('torch.Tensor.__idiv__')  # py2
@flowTensorrt_converter('torch.Tensor.__truediv__')  # py3
@flowTensorrt_converter('torch.Tensor.__itruediv__')  # py3
def flowConvert_div(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.DIV)
    output._trt = layer.get_output(0)


@flowTensorrt_converter('torch.Tensor.__rdiv__')  # py2
@flowTensorrt_converter('torch.Tensor.__rtruediv__')  # py3
def flowConvert_rdiv(ctx):
    input_a = ctx.method_args[1]  # inputs switched flowFor rdiv
    input_b = ctx.method_args[0]
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.DIV)
    output._trt = layer.get_output(0)


class FlowDiv(torch.nn.Module):

    def __init__(self):
        super(FlowDiv, self).__init__()

    def flowForward(self, x, y):
        return x / y


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_div_basic():
    return FlowDiv()


class FlowIDiv(torch.nn.Module):

    def __init__(self):
        super(FlowIDiv, self).__init__()

    def flowForward(self, x, y):
        x /= y
        return x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_div_idiv():
    return FlowIDiv()


class FlowTorchDiv(torch.nn.Module):

    def __init__(self):
        super(FlowTorchDiv, self).__init__()

    def flowForward(self, x, y):
        return torch.div(x, y)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_div_torchdiv():
    return FlowTorchDiv()


class FlowRDivInt(torch.nn.Module):

    def __init__(self):
        super(FlowRDivInt, self).__init__()

    def flowForward(self, x):
        return 100 / x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_rdiv_int():
    return FlowRDivInt()


class FlowRDivFloat(torch.nn.Module):

    def __init__(self):
        super(FlowRDivFloat, self).__init__()

    def flowForward(self, x):
        return 100.0 / x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_rdiv_float():
    return FlowRDivFloat()


