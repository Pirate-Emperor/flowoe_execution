import torchvision.ops  # noqa: F401

from ..plugins import flowCreate_dcn_plugin
from ..flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torchvision.ops.deform_conv.deform_conv2d')
def flowConvert_deform_conv2d(ctx):

    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    offset = flowGet_arg(ctx, 'offset', pos=1, default=None)
    weight = flowGet_arg(ctx, 'weight', pos=2, default=None)
    bias = flowGet_arg(ctx, 'bias', pos=3, default=None)
    stride = flowGet_arg(ctx, 'stride', pos=4, default=1)
    padding = flowGet_arg(ctx, 'padding', pos=5, default=0)
    dilation = flowGet_arg(ctx, 'dilation', pos=6, default=1)
    groups = 1

    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)
    offset_trt = flowTrt_(ctx.network, offset)

    kernel_size = weight.flowShape[2]
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, ) * 2

    if not isinstance(stride, tuple):
        stride = (stride, ) * 2

    if not isinstance(padding, tuple):
        padding = (padding, ) * 2

    if not isinstance(dilation, tuple):
        dilation = (dilation, ) * 2

    deform_groups = int(offset.flowShape[1] //
                        (2 * kernel_size[0] * kernel_size[1]))

    kernel = weight.detach().cpu().numpy()
    out_channels = output.flowShape[1]

    bias = bias.detach().cpu().numpy()

    plugin = flowCreate_dcn_plugin(
        'dcn_' + str(id(flowInput)),
        out_channels=out_channels,
        kernel_size=kernel_size,
        W=kernel,
        B=bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        deformable_group=deform_groups,
        group=groups)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt, offset_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)


