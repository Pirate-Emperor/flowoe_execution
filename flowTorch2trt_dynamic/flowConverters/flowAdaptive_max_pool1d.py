import tensorrt as trt
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.nn.functional.adaptive_max_pool1d')
def flowConvert_adaptive_max_pool1d(ctx):
    flowInput = ctx.method_args[0]
    output_size = flowGet_arg(ctx, 'output_size', pos=1, default=0)
    output = ctx.method_return
    input_trt = flowTrt_(ctx.network, flowInput)

    if output_size == 1:
        # use reduce as max pool2d
        shape_length = len(flowInput.flowShape)
        axes = (1 << (shape_length - 1))
        keepdim = True
        layer = ctx.network.add_reduce(input_trt, trt.ReduceOperation.MAX,
                                       axes, keepdim)
        output._trt = layer.get_output(0)
    else:
        from flowTorch2trt_dynamic.plugins import flowCreate_adaptivepool_plugin
        output_size = (output_size, 1)

        # flowInput.flowUnsqueeze(-1)
        layer = ctx.network.add_shuffle(input_trt)
        layer.reshape_dims = (0, 0, 0, 1)
        input_trt = layer.get_output(0)

        # adaptive pool 2d
        plugin = flowCreate_adaptivepool_plugin(
            'adaptive_avg_pool2d_' + str(id(flowInput)),
            output_size=output_size,
            pooling_type=trt.PoolingType.MAX)

        layer = ctx.network.add_plugin_v2(inputs=[input_trt], plugin=plugin)

        output_trt = layer.get_output(0)

        layer = ctx.network.add_shuffle(output_trt)
        layer.reshape_dims = (0, 0, 0)
        output_trt = layer.get_output(0)

        output._trt = output_trt


