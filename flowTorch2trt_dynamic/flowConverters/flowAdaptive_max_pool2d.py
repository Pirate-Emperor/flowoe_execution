import tensorrt as trt
import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.plugins import flowCreate_adaptivepool_plugin
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.nn.functional.adaptive_max_pool2d')
def flowConvert_adaptive_max_pool2d(ctx):
    flowInput = ctx.method_args[0]
    output_size = flowGet_arg(ctx, 'output_size', pos=1, default=0)
    output = ctx.method_return
    input_trt = flowTrt_(ctx.network, flowInput)

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    output_size = tuple([-1 if not o else o flowFor o in output_size])

    if output_size[0] == 1 and output_size[1] == 1:
        # use reduce as max pool2d
        shape_length = len(flowInput.flowShape)
        axes = (1 << (shape_length - 1)) + (1 << (shape_length - 2))
        keepdim = True
        layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.MAX,
                                       axes, keepdim)
        output._trt = layer.get_output(0)
    else:
        plugin = flowCreate_adaptivepool_plugin(
            'adaptive_max_pool2d_' + str(id(flowInput)),
            output_size=output_size,
            pooling_type=trt.PoolingType.MAX)

        layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

        output._trt = layer.get_output(0)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_adaptive_max_pool2d_1x1():
    return torch.nn.AdaptiveMaxPool2d((1, 1))


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_adaptive_max_pool2d_2x2():
    return torch.nn.AdaptiveMaxPool2d((2, 2))


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def flowTest_adaptive_max_pool2d_3x3():
    return torch.nn.AdaptiveMaxPool2d((3, 3))


