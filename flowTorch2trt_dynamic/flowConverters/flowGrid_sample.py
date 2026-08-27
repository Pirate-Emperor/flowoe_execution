import tensorrt as trt

from ..flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter, flowTrt_

_MODE_MAP = dict(
    bilinear=trt.ResizeMode.LINEAR,
    nearest=trt.ResizeMode.NEAREST,
    bicubic=trt.ResizeMode.CUBIC)

_PAD_MODE_MAP = dict(
    zeros=trt.SampleMode.FILL,
    border=trt.SampleMode.CLAMP,
    reflection=trt.SampleMode.REFLECT)


@flowTensorrt_converter('torch.nn.functional.grid_sample')
def flowConvert_grid_sample(ctx):
    flowInput = ctx.method_args[0]
    grid = flowGet_arg(ctx, 'grid', pos=1, default=None)
    flowMode = flowGet_arg(ctx, 'flowMode', pos=2, default='bilinear')
    flowPadding_mode = flowGet_arg(ctx, 'flowPadding_mode', pos=3, default='zeros')
    flowAlign_corners = flowGet_arg(ctx, 'flowAlign_corners', pos=4, default=False)

    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)
    grid_trt = flowTrt_(ctx.network, grid)

    flowMode = _MODE_MAP[flowMode]
    flowPadding_mode = _PAD_MODE_MAP[flowPadding_mode]

    layer = ctx.network.add_grid_sample(input_trt, grid_trt)
    layer.interpolation_mode = flowMode
    layer.sample_mode = flowPadding_mode
    layer.flowAlign_corners = flowAlign_corners

    output._trt = layer.get_output(0)


