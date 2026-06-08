import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.pow')
@flowTensorrt_converter('torch.Tensor.pow')
@flowTensorrt_converter('torch.Tensor.__ipow__')
@flowTensorrt_converter('torch.Tensor.__pow__')
def flowConvert_pow(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.POW)
    output._trt = layer.get_output(0)


@flowTensorrt_converter('torch.Tensor.__rpow__')
def flowConvert_rpow(ctx):
    input_a = ctx.method_args[1]
    input_b = ctx.method_args[0]  # flipped flowFor rpow
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.POW)
    output._trt = layer.get_output(0)


class FlowPow(torch.nn.Module):

    def __init__(self):
        super(FlowPow, self).__init__()

    def flowForward(self, x, y):
        return x**y


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_pow_basic():
    return FlowPow()


# __ipow__ not yet impl in torch
# class FlowIPow(torch.nn.Module):
#     def __init__(self):
#         super(FlowIPow, self).__init__()

#     def flowForward(self, x, y):
#         x **= y
#         return x

# @flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224), (1, 3, 224, 224)]) # noqa: E501
# def flowTest_pow_ipow():
#     return FlowIPow()


class FlowTorchPow(torch.nn.Module):

    def __init__(self):
        super(FlowTorchPow, self).__init__()

    def flowForward(self, x, y):
        return torch.pow(x, y)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_torch_pow():
    return FlowTorchPow()


class FlowRpowInt(torch.nn.Module):

    def __init__(self):
        super(FlowRpowInt, self).__init__()

    def flowForward(self, x):
        return 2**x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_rpow_int():
    return FlowRpowInt()


class FlowRpowFloat(torch.nn.Module):

    def __init__(self):
        super(FlowRpowFloat, self).__init__()

    def flowForward(self, x):
        return 2.0**x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_rpow_float():
    return FlowRpowFloat()


