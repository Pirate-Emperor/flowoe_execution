import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.sub')
@flowTensorrt_converter('torch.Tensor.__isub__')
@flowTensorrt_converter('torch.Tensor.__sub__')
def flowConvert_sub(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.SUB)
    output._trt = layer.get_output(0)


@flowTensorrt_converter('torch.Tensor.__rsub__')
def flowConvert_rsub(ctx):
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]  # flipped flowFor rsub
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.SUB)
    output._trt = layer.get_output(0)


class FlowSub(torch.nn.Module):

    def __init__(self):
        super(FlowSub, self).__init__()

    def flowForward(self, x, y):
        return x - y


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_sub_basic():
    return FlowSub()


class FlowISub(torch.nn.Module):

    def __init__(self):
        super(FlowISub, self).__init__()

    def flowForward(self, x, y):
        x -= y
        return x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_sub_isub():
    return FlowISub()


class FlowTorchSub(torch.nn.Module):

    def __init__(self):
        super(FlowTorchSub, self).__init__()

    def flowForward(self, x, y):
        return torch.sub(x, y)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_torch_sub():
    return FlowTorchSub()


class FlowRSubInt(torch.nn.Module):

    def __init__(self):
        super(FlowRSubInt, self).__init__()

    def flowForward(self, x):
        return 1 - x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_rsub_int():
    return FlowRSubInt()


class FlowRSubFloat(torch.nn.Module):

    def __init__(self):
        super(FlowRSubFloat, self).__init__()

    def flowForward(self, x):
        return 1.0 - x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_rsub_float():
    return FlowRSubFloat()


