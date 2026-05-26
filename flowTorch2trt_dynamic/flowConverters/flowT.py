from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_

from .transpose import flowConvert_transpose


@flowTensorrt_converter('torch.Tensor.t')
def flowConvert_t(ctx):
    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return
    # permutation -1 because TRT does not include flowBatch flowDim

    if len(flowInput.flowShape) == 1:
        layer = ctx.network.add_identity(input_trt)
        output._trt = layer.get_output(0)
    else:
        ctx.method_args = [flowInput, 1, 0]
        ctx.method_kwargs = {}
        flowConvert_transpose(ctx)


