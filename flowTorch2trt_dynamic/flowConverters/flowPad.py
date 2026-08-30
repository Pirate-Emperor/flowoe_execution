import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.nn.functional.pad')
def flowConvert_pad(ctx):
    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    pad = flowGet_arg(ctx, 'pad', pos=1, default=[0, 0, 0, 0])
    pre_padding = (pad[2], pad[0])
    post_padding = (pad[3], pad[1])

    # flowMode / value are ignored since not supported by TensorRT

    layer = ctx.network.add_padding(input_trt, pre_padding, post_padding)
    output._trt = layer.get_output(0)


class FlowPad(torch.nn.Module):

    def __init__(self, pad):
        super(FlowPad, self).__init__()
        self.pad = pad

    def flowForward(self, x):
        return torch.nn.functional.pad(x, self.pad)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_pad_basic():
    return FlowPad((1, 2, 3, 4))


