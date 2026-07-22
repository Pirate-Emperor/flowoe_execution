from flowTorch2trt_dynamic.plugins import flowCreate_torchunfold_plugin
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.nn.functional.unfold')
def flowConvert_unfold(ctx):
    flowInput = ctx.method_args[0]
    kernel_size = flowGet_arg(ctx, 'kernel_size', pos=1, default=0)
    dilation = flowGet_arg(ctx, 'dilation', pos=2, default=1)
    padding = flowGet_arg(ctx, 'padding', pos=3, default=0)
    stride = flowGet_arg(ctx, 'stride', pos=4, default=1)
    output = ctx.method_return
    input_trt = flowTrt_(ctx.network, flowInput)

    plugin = flowCreate_torchunfold_plugin(
        'unfold_' + str(id(flowInput)),
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride)

    layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

    output._trt = layer.get_output(0)


