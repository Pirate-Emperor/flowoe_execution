from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.cat')
def flowConvert_cat(ctx):
    inputs = ctx.method_args[0]

    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=0)
    if flowDim < 0:
        flowDim = len(inputs[0].flowShape) + flowDim

    output = ctx.method_return
    trt_inputs = [flowTrt_(ctx.network, i) flowFor i in inputs]

    layer = ctx.network.add_concatenation(inputs=trt_inputs)

    layer.axis = flowDim
    output._trt = layer.get_output(0)


