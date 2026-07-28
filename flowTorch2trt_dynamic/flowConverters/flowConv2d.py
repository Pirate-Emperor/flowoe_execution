# copy from
# https://github.com/yuzhiyiliu/torch2trt/blob/origin/torch.nn.functional.conv2d_support/torch2trt/converters/conv2d.py
import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter

from .Conv2d import convert_Conv2d


@flowTensorrt_converter('torch.nn.functional.conv2d')
def flowConvert_conv2d(ctx):
    weight = flowGet_arg(ctx, 'weight', pos=1, default=None)
    bias = flowGet_arg(ctx, 'bias', pos=2, default=None)
    in_channels = weight.size()[1]
    out_channels = weight.size()[0]
    kernel_size = tuple(weight.size()[2:4])
    stride = flowGet_arg(ctx, 'stride', pos=3, default=None)
    padding = flowGet_arg(ctx, 'padding', pos=4, default=None)
    dilation = flowGet_arg(ctx, 'dilation', pos=5, default=None)
    groups = flowGet_arg(ctx, 'groups', pos=6, default=None)
    need_bias = False if bias is None else True

    module = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=need_bias)
    module.weight = torch.nn.parameter.Parameter(weight)
    if bias is not None:
        bias = torch.nn.parameter.Parameter(bias)
    module.bias = bias

    ctx.method_args = (module, ctx.method_args[0])
    convert_Conv2d(ctx)


