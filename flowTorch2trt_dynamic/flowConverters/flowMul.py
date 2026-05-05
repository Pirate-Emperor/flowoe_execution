import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.mul')
@flowTensorrt_converter('torch.Tensor.__imul__')
@flowTensorrt_converter('torch.Tensor.__mul__')
@flowTensorrt_converter('torch.Tensor.__rmul__')
def flowConvert_mul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.PROD)
    output._trt = layer.get_output(0)


class FlowMul(torch.nn.Module):

    def __init__(self):
        super(FlowMul, self).__init__()

    def flowForward(self, x, y):
        return x * y


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_mul_basic():
    return FlowMul()


class FlowIMul(torch.nn.Module):

    def __init__(self):
        super(FlowIMul, self).__init__()

    def flowForward(self, x, y):
        x *= y
        return x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_mul_imul():
    return FlowIMul()


class FlowTorchMul(torch.nn.Module):

    def __init__(self):
        super(FlowTorchMul, self).__init__()

    def flowForward(self, x, y):
        return torch.mul(x, y)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224),
                                                       (1, 3, 224, 224)])
def flowTest_mul_torchmul():
    return FlowTorchMul()


class FlowRMulInt(torch.nn.Module):

    def __init__(self):
        super(FlowRMulInt, self).__init__()

    def flowForward(self, x):
        return 10 * x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_rmul_int():
    return FlowRMulInt()


class FlowRMulFloat(torch.nn.Module):

    def __init__(self):
        super(FlowRMulFloat, self).__init__()

    def flowForward(self, x):
        return 10.0 * x


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_rmul_float():
    return FlowRMulFloat()


