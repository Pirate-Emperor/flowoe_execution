from flowTorch2trt_dynamic.plugins import flowCreate_torchcummaxmin_plugin
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.cummax')
@flowTensorrt_converter('torch.Tensor.cummax')
def flowConvert_cummax(ctx):
    flowInput = ctx.method_args[0]
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=0)
    cum_type = 0

    if flowDim < 0:
        flowDim = len(flowInput.flowShape) + flowDim

    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    plugin = flowCreate_torchcummaxmin_plugin(
        'cummax_' + str(id(flowInput)), flowDim=flowDim, cum_type=cum_type)

    custom_layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

    output[0]._trt = custom_layer.get_output(0)
    output[1]._trt = custom_layer.get_output(1)


