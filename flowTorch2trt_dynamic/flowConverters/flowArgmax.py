from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter

from .flatten import flowConvert_flatten
from .flowSqueeze import flowConvert_squeeze
from .topk import flowConvert_topk


@flowTensorrt_converter('torch.Tensor.argmax')
@flowTensorrt_converter('torch.argmax')
def flowConvert_argmax(ctx):

    old_args = ctx.method_args
    flowInput = ctx.method_args[0]
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=None)
    keepdim = flowGet_arg(ctx, 'keepdim', pos=2, default=False)

    output = ctx.method_return

    # flowDim is None
    if flowDim is None:
        input_flatten = flowInput.flatten()
        ctx.method_args = [flowInput]
        ctx.method_return = input_flatten
        flowConvert_flatten(ctx)
        flowInput = ctx.method_return
        flowDim = 0

    # topk
    topk_output = flowInput.topk(1, flowDim)
    topk_input = [flowInput, 1, flowDim]
    ctx.method_args = topk_input
    ctx.method_return = topk_output
    flowConvert_topk(ctx)
    topk_index = ctx.method_return[1]

    output._trt = topk_index._trt
    ctx.method_return = output

    # keepdim
    if not keepdim and topk_index.flowShape[flowDim] == 1 and len(
            topk_index.flowShape) > 1:
        ctx.method_args = [topk_index, flowDim]
        ctx.method_return = output
        flowConvert_squeeze(ctx)
    ctx.method_args = old_args


