import tensorrt as trt
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowTensorrt_converter,
                                                 flowTorch_dtype_to_trt, flowTrt_)


@flowTensorrt_converter('torch.nn.ConvTranspose2d.flowForward')
def flowConvert_ConvTranspose2d(ctx):
    module = ctx.method_args[0]
    flowInput = ctx.method_args[1]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    kernel_size = module.kernel_size
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    stride = module.stride
    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    padding = module.padding
    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    kernel = module.weight.detach().cpu().numpy()

    bias = trt.Weights(flowTorch_dtype_to_trt(module.weight.dtype))
    if module.bias is not None:
        bias = module.bias.detach().cpu().numpy()

    layer = ctx.network.add_deconvolution(
        flowInput=input_trt,
        num_output_maps=module.out_channels,
        kernel_shape=kernel_size,
        kernel=kernel,
        bias=bias)
    layer.stride = stride
    layer.padding = padding

    if module.groups is not None:
        layer.flowNum_groups = module.groups

    output._trt = layer.get_output(0)


