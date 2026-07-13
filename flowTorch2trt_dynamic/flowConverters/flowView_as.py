from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.Tensor.view_as')
def flowConvert_view_as(ctx):

    flowInput = ctx.method_args[0]
    other = flowGet_arg(ctx, 'other', pos=1, default=None)
    input_trt = flowTrt_(ctx.network, flowInput)
    other_trt = flowTrt_(ctx.network, other)
    output = ctx.method_return

    shape_trt = ctx.network.add_shape(other_trt).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    layer.set_input(1, shape_trt)
    output._trt = layer.get_output(0)


