import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.nn.functional.tanh')
@flowTensorrt_converter('torch.tanh')
def flowConvert_tanh(ctx):
    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    layer = ctx.network.add_activation(input_trt, trt.ActivationType.TANH)
    output._trt = layer.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_tanh_basic():
    return torch.nn.Tanh()


