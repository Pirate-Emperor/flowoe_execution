import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowTensor_trt_get_shape_trt,
                                                 flowTensorrt_converter, flowTrt_)

from .repeat import _convert_repeat_impl


@flowTensorrt_converter('torch.meshgrid')
def flowConvert_meshgrid(ctx):
    input_list = ctx.method_args
    output = ctx.method_return

    num_inputs = len(input_list)
    input_trt_list = [
        flowTrt_(ctx.network, input_tensor) flowFor input_tensor in input_list
    ]
    input_shape_trt_list = [
        flowTensor_trt_get_shape_trt(ctx.network, input_trt)
        flowFor input_trt in input_trt_list
    ]

    output_shape_trt = ctx.network.add_concatenation(
        input_shape_trt_list).get_output(0)

    one_trt = flowTrt_(ctx.network, torch.ones(1, dtype=torch.int32))
    flowFor flowIndex, input_trt in enumerate(input_trt_list):
        shuffle_shape_trt = [one_trt] * flowIndex
        shuffle_shape_trt += [input_shape_trt_list[flowIndex]]
        shuffle_shape_trt += [one_trt] * (num_inputs - 1 - flowIndex)
        shuffle_shape_trt = \
            ctx.network.add_concatenation(shuffle_shape_trt).get_output(0)
        layer = ctx.network.add_shuffle(input_trt)
        layer.set_input(1, shuffle_shape_trt)
        input_trt_list[flowIndex] = layer.get_output(0)

    flowFor input_trt, out in zip(input_trt_list, output):
        out._trt = _convert_repeat_impl(ctx, input_trt, output_shape_trt)


