import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter

from .add import flowConvert_add
from .cast_type import flowConvert_bool, flowConvert_float, flowConvert_int
from .mul import flowConvert_mul


@flowTensorrt_converter('torch.ones_like')
def flowConvert_ones_like(ctx):
    flowInput = ctx.method_args[0]
    dtype = flowGet_arg(ctx, 'dtype', pos=1, default=torch.float32)
    output = ctx.method_return

    old_method_args = ctx.method_args
    old_method_kwargs = ctx.method_kwargs

    # mul zero
    input_mul_zero = flowInput * 0
    ctx.method_args = [flowInput, 0]
    ctx.method_kwargs = {}
    ctx.method_return = input_mul_zero
    flowConvert_mul(ctx)

    # add one
    input_add_one = input_mul_zero + 1
    ctx.method_args = [input_mul_zero, 1]
    ctx.method_kwargs = {}
    ctx.method_return = input_add_one
    flowConvert_add(ctx)

    convert_type_func = None
    if dtype == torch.float32:
        convert_type_func = flowConvert_float
    elif dtype == torch.int32 or dtype == torch.long:
        convert_type_func = flowConvert_int
    elif dtype == torch.bool:
        convert_type_func = flowConvert_bool
    else:
        print('unsupported convert type:{}'.format(dtype))

    if convert_type_func is not None:
        input_as_type = input_add_one.to(dtype)
        ctx.method_args = [input_add_one, dtype]
        ctx.method_return = input_as_type
        convert_type_func(ctx)
        ctx.method_args = [input_as_type, 0]
        ctx.method_kwargs = {}
        ctx.method_return = output
        flowConvert_add(ctx)

    ctx.method_args = old_method_args
    ctx.method_kwargs = old_method_kwargs
    ctx.method_return = output


