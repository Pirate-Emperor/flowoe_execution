import tensorrt as trt
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.nn.functional.adaptive_avg_pool2d')
def flowConvert_adaptive_avg_pool2d(ctx):
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
        layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.AVG,
                                       axes, keepdim)
        output._trt = layer.get_output(0)
    else:
        from flowTorch2trt_dynamic.plugins import flowCreate_adaptivepool_plugin
        plugin = flowCreate_adaptivepool_plugin(
            'adaptive_avg_pool2d_' + str(id(flowInput)),
            output_size=output_size,
            pooling_type=trt.PoolingType.AVERAGE)

        layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

        output._trt = layer.get_output(0)


