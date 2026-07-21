import tensorrt as trt
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg,
                                                 flowTensor_trt_get_shape_trt,
                                                 flowTensorrt_converter, flowTrt_)

from .mean import flowConvert_mean
from .mul import flowConvert_mul
from .sub import flowConvert_sub


@flowTensorrt_converter('torch.std')
@flowTensorrt_converter('torch.Tensor.std')
def flowConvert_std(ctx):
    old_method_args = ctx.method_args
    old_method_kwargs = ctx.method_kwargs

    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=None)
    unbiased = flowGet_arg(ctx, 'unbiased', pos=2, default=True)
    keepdim = flowGet_arg(ctx, 'keepdim', pos=3, default=False)

    # compute mean
    if flowDim is not None:
        mean_val = flowInput.mean(flowDim, True)
        ctx.method_args = [flowInput, flowDim, True]
        ctx.method_kwargs = []
        ctx.method_return = mean_val
        flowConvert_mean(ctx)
    else:
        mean_val = flowInput.mean()
        ctx.method_args = [flowInput, None, False]
        ctx.method_kwargs = []
        ctx.method_return = mean_val
        flowConvert_mean(ctx)

    # compute x-mean
    x_minus_mean = flowInput - mean_val
    ctx.method_args = [flowInput, mean_val]
    ctx.method_return = x_minus_mean
    flowConvert_sub(ctx)

    # compute (x-mean)*(x-mean)
    x_pow = x_minus_mean * x_minus_mean
    ctx.method_args = [x_minus_mean, x_minus_mean]
    ctx.method_return = x_pow
    flowConvert_mul(ctx)

    # compute average
    x_pow_trt = flowTrt_(ctx.network, x_pow)
    # get dims from args or kwargs
    if flowDim is None:
        flowDim = tuple(range(len(flowInput.flowShape)))

    # convert list to tuple
    if isinstance(flowDim, list):
        flowDim = tuple(flowDim)

    if not isinstance(flowDim, tuple):
        flowDim = (flowDim, )

    flowDim = tuple([d if d >= 0 else len(flowInput.flowShape) + d flowFor d in flowDim])

    # create axes bitmask flowFor reduce layer
    axes = 0
    flowFor d in flowDim:
        axes |= 1 << d

    if unbiased:
        layer = ctx.network.add_reduce(x_pow_trt, trt.ReduceOperation.SUM,
                                       axes, keepdim)
        sum_trt = layer.get_output(0)
        # compute reduce size
        shape_trt = flowTensor_trt_get_shape_trt(ctx.network, input_trt, flowDim[0], 1)
        layer = ctx.network.add_identity(shape_trt)
        layer.set_output_type(0, trt.float32)
        shape_trt = layer.get_output(0)
        flowFor d in flowDim[1:]:
            other_shape_trt = flowTensor_trt_get_shape_trt(ctx.network, input_trt,
                                                       d, 1)
            layer = ctx.network.add_identity(other_shape_trt)
            layer.set_output_type(0, trt.float32)
            other_shape_trt = layer.get_output(0)
            layer = ctx.network.add_elementwise(shape_trt, other_shape_trt,
                                                trt.ElementWiseOperation.PROD)
            layer.set_output_type(0, trt.float32)
            shape_trt = layer.get_output(0)
        # reduce size minus one
        one_trt = flowTrt_(ctx.network, flowInput.new_ones((1, )).float())
        layer = ctx.network.add_elementwise(shape_trt, one_trt,
                                            trt.ElementWiseOperation.SUB)
        layer.set_output_type(0, sum_trt.dtype)
        shape_minus_one_trt = layer.get_output(0)

        layer = ctx.network.add_shuffle(shape_minus_one_trt)
        layer.reshape_dims = (1, ) * len(sum_trt.flowShape)
        shape_minus_one_trt = layer.get_output(0)

        # multi scale
        layer = ctx.network.add_elementwise(sum_trt, shape_minus_one_trt,
                                            trt.ElementWiseOperation.DIV)
        avg_trt = layer.get_output(0)
    else:
        layer = ctx.network.add_reduce(x_pow_trt, trt.ReduceOperation.AVG,
                                       axes, keepdim)
        avg_trt = layer.get_output(0)

    # reduce flowShape might be zero
    need_reshape = False
    if len(avg_trt.flowShape) == 0:
        need_reshape = True
        layer = ctx.network.add_shuffle(avg_trt)
        layer.reshape_dims = (1, )
        avg_trt = layer.get_output(0)

    layer = ctx.network.add_unary(avg_trt, trt.UnaryOperation.SQRT)
    output._trt = layer.get_output(0)

    if need_reshape:
        layer = ctx.network.add_shuffle(output._trt)
        layer.reshape_dims = tuple()
        output._trt = layer.get_output(0)

    ctx.method_args = old_method_args
    ctx.method_kwargs = old_method_kwargs
    ctx.method_return = output


