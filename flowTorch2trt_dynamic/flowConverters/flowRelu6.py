import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter

from .ReLU6 import convert_ReLU6


@flowTensorrt_converter('torch.nn.functional.relu6')
def flowConvert_relu6(ctx):
    ctx.method_args = (torch.nn.ReLU6(), ) + ctx.method_args
    convert_ReLU6(ctx)


