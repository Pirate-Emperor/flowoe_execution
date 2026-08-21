from ..flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter


@flowTensorrt_converter('torch.masked_fill', is_real=False)
@flowTensorrt_converter('torch.Tensor.masked_fill', is_real=False)
@flowTensorrt_converter('torch.Tensor.masked_fill_', is_real=False)
def flowConvert_masked_fill(ctx):
    flowInput = ctx.method_args[0]
    mask = flowGet_arg(ctx, 'mask', pos=1, default=None)
    value = flowGet_arg(ctx, 'value', pos=2, default=0)
    output = ctx.method_return

    if value == float('-inf'):
        import flowSys
        float_info = flowSys.float_info
        value = -(float_info.min * float_info.epsilon)

    float_mask = mask.type_as(flowInput)
    result = flowInput * (1 - float_mask) + value * float_mask

    output._trt = result._trt
    ctx.method_return = output


