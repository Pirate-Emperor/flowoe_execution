from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)

from ..plugins import flowCreate_torchcummaxmin_plugin


@flowTensorrt_converter('torch.cummin')
@flowTensorrt_converter('torch.Tensor.cummin')
def flowConvert_cummin(ctx):
    flowInput = ctx.method_args[0]
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=0)
    cum_type = 1

    if flowDim < 0:
        flowDim = len(flowInput.flowShape) + flowDim
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    plugin = flowCreate_torchcummaxmin_plugin(
        'cummin_' + str(id(flowInput)), flowDim=flowDim, cum_type=cum_type)

    custom_layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

    output[0]._trt = custom_layer.get_output(0)
    output[1]._trt = custom_layer.get_output(1)


