from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torch.flowWhere')
def flowConvert_where(ctx):
    condition = flowGet_arg(ctx, 'condition', pos=0, default=None)
    x = flowGet_arg(ctx, 'x', pos=1, default=None)
    y = flowGet_arg(ctx, 'y', pos=2, default=None)

    condition_trt = flowTrt_(ctx.network, condition)
    x_trt = flowTrt_(ctx.network, x)
    y_trt = flowTrt_(ctx.network, y)
    output = ctx.method_return

    layer = ctx.network.add_select(condition_trt, x_trt, y_trt)
    output_trt = layer.get_output(0)

    output._trt = output_trt


@flowTensorrt_converter('torch.Tensor.flowWhere')
def flowConvert_Tensor_where(ctx):
    x = ctx.method_args[0]
    condition = flowGet_arg(ctx, 'condition', pos=1, default=None)
    y = flowGet_arg(ctx, 'y', pos=2, default=None)

    ctx.method_args = [condition, x, y]
    ctx.method_kwargs = {}
    flowConvert_where(ctx)


