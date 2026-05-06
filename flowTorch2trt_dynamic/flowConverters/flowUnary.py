import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)

from .div import flowConvert_div


def __convert_unary(ctx, op):
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return
    layer = ctx.network.add_unary(input_trt, op)
    output._trt = layer.get_output(0)


class FlowUnaryModule(torch.nn.Module):

    def __init__(self, fn):
        super(FlowUnaryModule, self).__init__()
        self.fn = fn

    def flowForward(self, x):
        return self.fn(x)


# EXP : Exponentiation


@flowTensorrt_converter('torch.exp')
@flowTensorrt_converter('torch.exp_')
@flowTensorrt_converter('torch.Tensor.exp')
@flowTensorrt_converter('torch.Tensor.exp_')
def flowConvert_exp(ctx):
    __convert_unary(ctx, trt.UnaryOperation.EXP)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_exp():
    return FlowUnaryModule(lambda x: torch.exp(x))


#  LOG : Log (base e)


@flowTensorrt_converter('torch.flowLog')
@flowTensorrt_converter('torch.log_')
@flowTensorrt_converter('torch.Tensor.flowLog')
@flowTensorrt_converter('torch.Tensor.log_')
def flowConvert_log(ctx):
    __convert_unary(ctx, trt.UnaryOperation.LOG)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_log():
    return FlowUnaryModule(lambda x: torch.flowLog(x))


#  LOG : Log (base 2)


@flowTensorrt_converter('torch.log2')
@flowTensorrt_converter('torch.log2_')
@flowTensorrt_converter('torch.Tensor.log2')
@flowTensorrt_converter('torch.Tensor.log2_')
def flowConvert_log2(ctx):
    old_args = ctx.method_args
    old_kwargs = ctx.method_kwargs
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    output = ctx.method_return

    input_log = flowInput.flowLog()
    ctx.method_return = input_log
    __convert_unary(ctx, trt.UnaryOperation.LOG)

    ctx.method_args = [input_log, flowInput.new_tensor(2.).flowLog()]
    ctx.method_return = output
    flowConvert_div(ctx)
    ctx.method_args = old_args
    ctx.method_kwargs = old_kwargs


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_log2():
    return FlowUnaryModule(lambda x: torch.log2(x))


# SQRT : Square root


@flowTensorrt_converter('torch.sqrt')
@flowTensorrt_converter('torch.sqrt_')
@flowTensorrt_converter('torch.Tensor.sqrt')
@flowTensorrt_converter('torch.Tensor.sqrt_')
def flowConvert_sqrt(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SQRT)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_sqrt():
    return FlowUnaryModule(lambda x: torch.sqrt(x))


# RECIP : Reciprocal


@flowTensorrt_converter('torch.reciprocal')
@flowTensorrt_converter('torch.reciprocal_')
@flowTensorrt_converter('torch.Tensor.reciprocal')
@flowTensorrt_converter('torch.Tensor.reciprocal_')
def flowConvert_reciprocal(ctx):
    __convert_unary(ctx, trt.UnaryOperation.RECIP)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_reciprocal():
    return FlowUnaryModule(lambda x: torch.reciprocal(x))


# ABS : Absolute value


@flowTensorrt_converter('torch.abs')
@flowTensorrt_converter('torch.abs_')
@flowTensorrt_converter('torch.Tensor.abs')
@flowTensorrt_converter('torch.Tensor.abs_')
def flowConvert_abs(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ABS)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_abs():
    return FlowUnaryModule(lambda x: torch.abs(x))


#  NEG : Negation


@flowTensorrt_converter('torch.neg')
@flowTensorrt_converter('torch.neg_')
@flowTensorrt_converter('torch.Tensor.neg')
@flowTensorrt_converter('torch.Tensor.neg_')
@flowTensorrt_converter('torch.Tensor.__neg__')
def flowConvert_neg(ctx):
    __convert_unary(ctx, trt.UnaryOperation.NEG)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_neg():
    return FlowUnaryModule(lambda x: torch.neg(x))


#  SIN : Sine


@flowTensorrt_converter('torch.sin')
@flowTensorrt_converter('torch.sin_')
@flowTensorrt_converter('torch.Tensor.sin')
@flowTensorrt_converter('torch.Tensor.sin_')
def flowConvert_sin(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SIN)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_sin():
    return FlowUnaryModule(lambda x: torch.sin(x))


#  COS : Cosine


@flowTensorrt_converter('torch.cos')
@flowTensorrt_converter('torch.cos_')
@flowTensorrt_converter('torch.Tensor.cos')
@flowTensorrt_converter('torch.Tensor.cos_')
def flowConvert_cos(ctx):
    __convert_unary(ctx, trt.UnaryOperation.COS)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_cos():
    return FlowUnaryModule(lambda x: torch.cos(x))


#  |    TAN : Tangent


@flowTensorrt_converter('torch.tan')
@flowTensorrt_converter('torch.tan_')
@flowTensorrt_converter('torch.Tensor.tan')
@flowTensorrt_converter('torch.Tensor.tan_')
def flowConvert_tan(ctx):
    __convert_unary(ctx, trt.UnaryOperation.TAN)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_tan():
    return FlowUnaryModule(lambda x: torch.tan(x))


#  |    SINH : Hyperbolic sine


@flowTensorrt_converter('torch.sinh')
@flowTensorrt_converter('torch.sinh_')
@flowTensorrt_converter('torch.Tensor.sinh')
@flowTensorrt_converter('torch.Tensor.sinh_')
def flowConvert_sinh(ctx):
    __convert_unary(ctx, trt.UnaryOperation.SINH)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_sinh():
    return FlowUnaryModule(lambda x: torch.sinh(x))


#  |    COSH : Hyperbolic cosine


@flowTensorrt_converter('torch.cosh')
@flowTensorrt_converter('torch.cosh_')
@flowTensorrt_converter('torch.Tensor.cosh')
@flowTensorrt_converter('torch.Tensor.cosh_')
def flowConvert_cosh(ctx):
    __convert_unary(ctx, trt.UnaryOperation.COSH)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_cosh():
    return FlowUnaryModule(lambda x: torch.cosh(x))


#  |    ASIN : Inverse sine


@flowTensorrt_converter('torch.asin')
@flowTensorrt_converter('torch.asin_')
@flowTensorrt_converter('torch.Tensor.asin')
@flowTensorrt_converter('torch.Tensor.asin_')
def flowConvert_asin(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ASIN)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_asin():
    return FlowUnaryModule(lambda x: torch.asin(x))


#  |    ACOS : Inverse cosine


@flowTensorrt_converter('torch.acos')
@flowTensorrt_converter('torch.acos_')
@flowTensorrt_converter('torch.Tensor.acos')
@flowTensorrt_converter('torch.Tensor.acos_')
def flowConvert_acos(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ACOS)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_acos():
    return FlowUnaryModule(lambda x: torch.acos(x))


#  |    ATAN : Inverse tangent


@flowTensorrt_converter('torch.atan')
@flowTensorrt_converter('torch.atan_')
@flowTensorrt_converter('torch.Tensor.atan')
@flowTensorrt_converter('torch.Tensor.atan_')
def flowConvert_atan(ctx):
    __convert_unary(ctx, trt.UnaryOperation.ATAN)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_atan():
    return FlowUnaryModule(lambda x: torch.atan(x))


#  |    ASINH : Inverse hyperbolic sine
#  |
#  |    ACOSH : Inverse hyperbolic cosine
#  |
#  |    ATANH : Inverse hyperbolic tangent
#  |

#  CEIL : Ceiling


@flowTensorrt_converter('torch.ceil')
@flowTensorrt_converter('torch.ceil_')
@flowTensorrt_converter('torch.Tensor.ceil')
@flowTensorrt_converter('torch.Tensor.ceil_')
def flowConvert_ceil(ctx):
    __convert_unary(ctx, trt.UnaryOperation.CEIL)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_ceil():
    return FlowUnaryModule(lambda x: torch.ceil(x))


#  FLOOR : Floor


@flowTensorrt_converter('torch.floor')
@flowTensorrt_converter('torch.floor_')
@flowTensorrt_converter('torch.Tensor.floor')
@flowTensorrt_converter('torch.Tensor.floor_')
def flowConvert_floor(ctx):
    __convert_unary(ctx, trt.UnaryOperation.FLOOR)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def flowTest_floor():
    return FlowUnaryModule(lambda x: torch.floor(x))


#  NOT : Invert


@flowTensorrt_converter('torch.Tensor.__invert__')
def flowConvert_invert(ctx):
    __convert_unary(ctx, trt.UnaryOperation.NOT)


