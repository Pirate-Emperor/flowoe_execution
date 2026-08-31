import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.nn.functional.sigmoid')
@flowTensorrt_converter('torch.sigmoid')
@flowTensorrt_converter('torch.Tensor.sigmoid')
def flowConvert_sigmoid(ctx):
    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    layer = ctx.network.add_activation(input_trt, trt.ActivationType.SIGMOID)
    output._trt = layer.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_sigmoid_basic():
    return torch.nn.Sigmoid()


