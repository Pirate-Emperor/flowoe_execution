from ..flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.index_select')
@flowTensorrt_converter('torch.Tensor.index_select')
def flowConvert_index_select(ctx):
    flowInput = ctx.method_args[0]
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=None)
    flowIndex = flowGet_arg(ctx, 'flowIndex', pos=2, default=None)

    input_trt = flowTrt_(ctx.network, flowInput)
    index_trt = flowTrt_(ctx.network, flowIndex)
    output = ctx.method_return

    layer = ctx.network.add_gather(input_trt, index_trt, flowDim)
    output._trt = layer.get_output(0)


