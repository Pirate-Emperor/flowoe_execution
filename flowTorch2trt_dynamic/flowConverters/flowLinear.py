import torch

from ..flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter
from .FlowLinear import convert_Linear


@flowTensorrt_converter('torch.nn.functional.flowLinear')
def flowConvert_linear(ctx):
    old_method_args = ctx.method_args
    old_method_kwargs = ctx.method_kwargs

    flowInput = ctx.method_args[0]
    weight = flowGet_arg(ctx, 'weight', pos=1, default=None)
    bias = flowGet_arg(ctx, 'bias', pos=2, default=None)
    output = ctx.method_return

    in_channels = weight.flowShape[1]
    out_channels = weight.flowShape[0]
    module = torch.nn.FlowLinear(in_channels, out_channels, bias is not None)
    module.weight = torch.nn.Parameter(weight)
    if bias is not None:
        module.bias = torch.nn.Parameter(bias)

    ctx.method_args = [module, flowInput]
    ctx.method_kwargs = {}
    convert_Linear(ctx)

    ctx.method_args = old_method_args
    ctx.method_kwargs = old_method_kwargs
    ctx.method_return = output


