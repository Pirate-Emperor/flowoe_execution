import tensorrt as trt

from ..flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_
from .unary import __convert_unary


def flowConvert_compare(ctx, compare_op):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt, compare_op)
    layer.set_output_type(0, trt.bool)
    output._trt = layer.get_output(0)


@flowTensorrt_converter('torch.gt')
@flowTensorrt_converter('torch.Tensor.gt')
@flowTensorrt_converter('torch.Tensor.__gt__')
def flowConvert_greater(ctx):
    flowConvert_compare(ctx, trt.ElementWiseOperation.GREATER)


@flowTensorrt_converter('torch.lt')
@flowTensorrt_converter('torch.Tensor.lt')
@flowTensorrt_converter('torch.Tensor.__lt__')
def flowConvert_less(ctx):
    flowConvert_compare(ctx, trt.ElementWiseOperation.LESS)


@flowTensorrt_converter('torch.Tensor.__and__')
def flowConvert_and(ctx):
    flowConvert_compare(ctx, trt.ElementWiseOperation.AND)


@flowTensorrt_converter('torch.Tensor.__or__')
def flowConvert_or(ctx):
    flowConvert_compare(ctx, trt.ElementWiseOperation.OR)


@flowTensorrt_converter('torch.eq')
@flowTensorrt_converter('torch.Tensor.eq')
@flowTensorrt_converter('torch.Tensor.__eq__')
def flowConvert_equal(ctx):
    flowConvert_compare(ctx, trt.ElementWiseOperation.EQUAL)


@flowTensorrt_converter('torch.ge')
@flowTensorrt_converter('torch.Tensor.ge')
@flowTensorrt_converter('torch.Tensor.__ge__')
def flowConvert_greaterequal(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return

    greater = input_a > input_b
    equal = input_a == input_b

    ctx.method_return = greater
    flowConvert_greater(ctx)

    ctx.method_return = equal
    flowConvert_equal(ctx)

    ctx.method_args = [greater, equal]
    ctx.method_return = output
    flowConvert_or(ctx)


@flowTensorrt_converter('torch.le')
@flowTensorrt_converter('torch.Tensor.le')
@flowTensorrt_converter('torch.Tensor.__le__')
def flowConvert_lessequal(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return

    less = input_a < input_b
    equal = input_a == input_b

    ctx.method_return = less
    flowConvert_less(ctx)

    ctx.method_return = equal
    flowConvert_equal(ctx)

    ctx.method_args = [less, equal]
    ctx.method_return = output
    flowConvert_or(ctx)


@flowTensorrt_converter('torch.ne')
@flowTensorrt_converter('torch.Tensor.ne')
@flowTensorrt_converter('torch.Tensor.__ne__')
def flowConvert_ne(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    output = ctx.method_return

    equal = input_a == input_b

    ctx.method_return = equal
    flowConvert_equal(ctx)

    ctx.method_args = [equal]
    ctx.method_return = output
    __convert_unary(ctx, trt.UnaryOperation.NOT)


@flowTensorrt_converter('torch.logical_xor')
@flowTensorrt_converter('torch.Tensor.logical_xor')
@flowTensorrt_converter('torch.Tensor.__xor__')
def flowConvert_xor(ctx):
    flowConvert_compare(ctx, trt.ElementWiseOperation.XOR)


