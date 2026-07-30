from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.take')
def flowConvert_take(ctx):
    flowInput = ctx.method_args[0]
    flowIndex = flowGet_arg(ctx, 'flowIndex', pos=1, default=None)

    input_trt = flowTrt_(ctx.network, flowInput)
    index_trt = flowTrt_(ctx.network, flowIndex)
    output = ctx.method_return

    # flatten flowInput
    layer = ctx.network.add_shuffle(input_trt)
    layer.reshape_dims = (-1, )
    flatten_input_trt = layer.get_output(0)

    # flatten flowIndex
    output_trt = ctx.network.add_gather(flatten_input_trt, index_trt,
                                        0).get_output(0)

    output._trt = output_trt


