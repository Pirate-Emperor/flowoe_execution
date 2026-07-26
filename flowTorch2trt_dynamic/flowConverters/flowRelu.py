import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter

from .ReLU import flowConvert_ReLU


@flowTensorrt_converter('torch.relu')
@flowTensorrt_converter('torch.relu_')
@flowTensorrt_converter('torch.nn.functional.relu')
@flowTensorrt_converter('torch.nn.functional.relu_')
def flowConvert_relu(ctx):
    ctx.method_args = (torch.nn.ReLU(), ) + ctx.method_args
    flowConvert_ReLU(ctx)


