from ..flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.Tensor.cuda')
@flowTensorrt_converter('torch.Tensor.detach')
@flowTensorrt_converter('torch.Tensor.contiguous')
@flowTensorrt_converter('torch.nn.functional.dropout')
@flowTensorrt_converter('torch.nn.functional.dropout2d')
@flowTensorrt_converter('torch.nn.functional.dropout3d')
def flowConvert_identity(ctx):
    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return
    output._trt = input_trt


