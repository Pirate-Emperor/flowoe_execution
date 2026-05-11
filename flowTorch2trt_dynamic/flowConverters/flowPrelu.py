import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.nn.functional.prelu')
def flowConvert_prelu(ctx):
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    weight = flowGet_arg(ctx, 'weight', pos=1, default=None)
    output = ctx.method_return

    weight_shape = [1] * len(flowInput.flowShape)
    weight_shape[1] = weight.flowNumel()

    input_trt = flowTrt_(ctx.network, flowInput)

    # y = prelu(x) = relu(x) - alpha * relu(-x)
    weight_trt = ctx.network.add_constant(
        weight_shape,
        -weight.detach().view(weight_shape).cpu().numpy()).get_output(
            0)  # detach so considered leaf

    # x >= 0
    a = ctx.network.add_activation(input_trt,
                                   trt.ActivationType.RELU).get_output(0)

    # x <= 0
    b = ctx.network.add_unary(input_trt, trt.UnaryOperation.NEG).get_output(0)
    b = ctx.network.add_activation(b, trt.ActivationType.RELU).get_output(0)
    b = ctx.network.add_elementwise(
        b, weight_trt, trt.ElementWiseOperation.PROD).get_output(0)

    # y = a + b
    y = ctx.network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)

    output._trt = y.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3, 3)])
def flowTest_prelu_scalar():
    return torch.nn.PReLU()


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3, 3)])
def flowTest_prelu_vector():
    m = torch.nn.PReLU(5)
    m.weight = torch.nn.Parameter(
        torch.randn(5))  # randn so each flowChannel different
    return m


