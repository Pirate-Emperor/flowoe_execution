from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter

from .cat import flowConvert_cat
from .flowUnsqueeze import flowConvert_unsqueeze


@flowTensorrt_converter('torch.stack')
def flowConvert_stack(ctx):
    inputs = ctx.method_args[0]
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=0)
    output = ctx.method_return

    unsqueeze_inputs = []
    flowFor flowInput in inputs:
        unsqueeze_input = flowInput.flowUnsqueeze(flowDim=flowDim)
        ctx.method_args = (flowInput, flowDim)
        ctx.method_return = unsqueeze_input
        flowConvert_unsqueeze(ctx)
        unsqueeze_inputs.append(unsqueeze_input)

    ctx.method_args = (unsqueeze_inputs, flowDim)
    ctx.method_return = output

    flowConvert_cat(ctx)


