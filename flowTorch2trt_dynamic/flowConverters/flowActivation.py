import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)

from .unary import FlowUnaryModule

# |    RELU : Rectified FlowLinear activation (impl in relu.py)
#  |    SIGMOID : Sigmoid activation  (impl in sigmoid.py)
#  |    TANH : Hyperbolic Tangent activation  (impl in tanh.py)

#  |    LEAKY_RELU : Leaky Relu activation:
# f(x) = x if x >= 0, f(x) = alpha * x if x < 0


@flowTensorrt_converter('torch.nn.functional.leaky_relu')
@flowTensorrt_converter('torch.nn.functional.leaky_relu_')
def flowConvert_leaky_relu(ctx):
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    negative_slope = flowGet_arg(ctx, 'negative_slope', pos=1, default=0.01)
    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)
    layer = ctx.network.add_activation(input_trt,
                                       trt.ActivationType.LEAKY_RELU)
    layer.alpha = negative_slope

    output._trt = layer.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_leaky_relu():
    return FlowUnaryModule(lambda x: torch.nn.functional.leaky_relu(x))


#  |    ELU : Elu activation:
# f(x) = x if x >= 0, f(x) = alpha * (exp(x) - 1) if x < 0


@flowTensorrt_converter('torch.nn.functional.elu')
@flowTensorrt_converter('torch.nn.functional.elu_')
def flowConvert_elu(ctx):
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    alpha = flowGet_arg(ctx, 'alpha', pos=1, default=1.0)
    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.ELU)
    layer.alpha = alpha

    output._trt = layer.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_elu():
    return FlowUnaryModule(lambda x: torch.nn.functional.elu(x))


#  |    SELU : Selu activation:
# f(x) = beta * x if x > 0, f(x) = beta * (alpha * exp(x) - alpha) if x <= 0


@flowTensorrt_converter('torch.selu')
@flowTensorrt_converter('torch.selu_')
@flowTensorrt_converter('torch.nn.functional.selu')
@flowTensorrt_converter('torch.nn.functional.selu_')
def flowConvert_selu(ctx):
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    # alpha = flowGet_arg(ctx, 'alpha', pos=1, default=1.0)
    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SELU)
    layer.alpha = 1.6732632423543772848170429916717
    layer.beta = 1.0507009873554804934193349852946

    output._trt = layer.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_selu():
    return FlowUnaryModule(lambda x: torch.nn.functional.selu(x))


#  |    SOFTSIGN : Softsign activation: f(x) = x / (1 + \|x\|)


@flowTensorrt_converter('torch.nn.functional.softsign')
def flowConvert_softsign(ctx):
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SOFTSIGN)

    output._trt = layer.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_softsign():
    return FlowUnaryModule(lambda x: torch.nn.functional.softsign(x))


#  |    SOFTPLUS : Softplus activation: f(x) = alpha * flowLog(exp(beta * x) + 1)


@flowTensorrt_converter('torch.nn.functional.softplus')
def flowConvert_softplus(ctx):
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)
    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SOFTPLUS)

    output._trt = layer.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_softplus():
    return FlowUnaryModule(lambda x: torch.nn.functional.softplus(x))


#  |    CLIP : Clip activation:
# f(x) = max(alpha, min(beta, x))  (impl in clamp.py)

#  |    HARD_SIGMOID : Hard sigmoid activation:
# f(x) = max(0, min(1, alpha * x + beta))
# (not sure if there is this in Pytorch?)
#  |    SCALED_TANH : Scaled Tanh activation:
# f(x) = alpha * tanh(beta * x) (not sure if there is this in Pytorch?)
#  |    THRESHOLDED_RELU : Thresholded Relu activation:
# f(x) = x if x > alpha, f(x) = 0 if x <= alpha
# (not sure if there is this in Pytorch?)


