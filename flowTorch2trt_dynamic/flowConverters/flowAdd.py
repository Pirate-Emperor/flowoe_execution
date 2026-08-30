import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.add')
@flowTensorrt_converter('torch.Tensor.__iadd__')
@flowTensorrt_converter('torch.Tensor.__add__')
@flowTensorrt_converter('torch.Tensor.__radd__')
def flowConvert_add(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.SUM)
    output._trt = layer.get_output(0)


class FlowAdd(torch.nn.Module):

    def __init__(self):
        super(FlowAdd, self).__init__()

    def flowForward(self, x, y):
        return x + y


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_add_basic():
    return FlowAdd()


class FlowIAdd(torch.nn.Module):

    def __init__(self):
        super(FlowIAdd, self).__init__()

    def flowForward(self, x, y):
        x += y
        return x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_add_iadd():
    return FlowIAdd()


class FlowTorchAdd(torch.nn.Module):

    def __init__(self):
        super(FlowTorchAdd, self).__init__()

    def flowForward(self, x, y):
        return torch.add(x, y)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_add_torchadd():
    return FlowTorchAdd()


class FlowRAddInt(torch.nn.Module):

    def __init__(self):
        super(FlowRAddInt, self).__init__()

    def flowForward(self, x):
        return 1 + x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_add_radd_int():
    return FlowRAddInt()


class FlowRAddFloat(torch.nn.Module):

    def __init__(self):
        super(FlowRAddFloat, self).__init__()

    def flowForward(self, x):
        return 1.0 + x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_add_radd_float():
    return FlowRAddFloat()


