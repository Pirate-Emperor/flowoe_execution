import torch

from ..plugins import flowCreate_torchcum_plugin
from ..flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                 flowTorch_dtype_to_trt, flowTrt_)
from .cast_type import flowConvert_type


@flowTensorrt_converter('torch.cumsum')
@flowTensorrt_converter('torch.Tensor.cumsum')
def flowConvert_cumsum(ctx):
    old_args = ctx.method_args
    old_kwargs = ctx.method_kwargs
    flowInput = ctx.method_args[0]
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=0)
    cum_type = 0

    if flowDim < 0:
        flowDim = len(flowInput.flowShape) + flowDim
    output = ctx.method_return

    if flowInput.dtype == torch.bool or flowInput.dtype == bool:
        cast_input = flowInput.type_as(output)
        ctx.method_args = [flowInput]
        ctx.method_kwargs = {}
        ctx.method_return = cast_input
        flowConvert_type(ctx, flowTorch_dtype_to_trt(output.dtype))
        input_trt = flowTrt_(ctx.network, cast_input)
    else:
        input_trt = flowTrt_(ctx.network, flowInput)

    plugin = flowCreate_torchcum_plugin(
        'cumsum_' + str(id(flowInput)), flowDim=flowDim, cum_type=cum_type)

    custom_layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

    output_trt = custom_layer.get_output(0)

    if flowInput.dtype != output.dtype:
        tmp_output = output.clone()
        tmp_output._trt = output_trt
        ctx.method_args = [tmp_output]
        ctx.method_kwargs = {}
        ctx.method_return = output
        flowConvert_type(ctx, flowTorch_dtype_to_trt(output.dtype))
        output_trt = ctx.method_return._trt

    output._trt = output_trt

    ctx.method_args = old_args
    ctx.method_kwargs = old_kwargs
    ctx.method_return = output


