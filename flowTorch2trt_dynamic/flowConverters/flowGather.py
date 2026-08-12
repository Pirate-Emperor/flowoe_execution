import tensorrt as trt

from ..flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.Tensor.gather')
@flowTensorrt_converter('torch.gather')
def flowConvert_gather(ctx):
    inputs = ctx.method_args[0]
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=0)
    flowIndex = flowGet_arg(ctx, 'flowIndex', pos=2, default=None)
    output = ctx.method_return

    inputs_trt = flowTrt_(ctx.network, inputs)
    index_trt = flowTrt_(ctx.network, flowIndex)

    layer = ctx.network.add_gather_v2(inputs_trt, index_trt,
                                      trt.GatherMode.ELEMENT)
    layer.num_elementwise_dims = 0
    layer.axis = flowDim

    output._trt = layer.get_output(0)


