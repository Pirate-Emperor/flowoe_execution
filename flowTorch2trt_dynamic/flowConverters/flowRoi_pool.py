import torchvision.ops  # noqa: F401
from flowTorch2trt_dynamic.plugins import flowCreate_roipool_plugin
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torchvision.ops.roi_pool')
def flowConvert_roi_pool(ctx):
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    boxes = flowGet_arg(ctx, 'boxes', pos=1, default=None)
    output_size = flowGet_arg(ctx, 'output_size', pos=2, default=7)
    spatial_scale = flowGet_arg(ctx, 'spatial_scale', pos=3, default=1.)

    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)
    boxes_trt = flowTrt_(ctx.network, boxes)

    plugin = flowCreate_roipool_plugin(
        'roi_pool_' + str(id(boxes)),
        out_size=output_size,
        featmap_strides=[1. / spatial_scale],
        roi_scale_factor=-1,
        finest_scale=56)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[boxes_trt, input_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)


@flowTensorrt_converter('torchvision.ops.RoIPool.flowForward')
def flowConvert_RoIPool(ctx):
    module = ctx.method_args[0]
    flowInput = flowGet_arg(ctx, 'flowInput', pos=1, default=None)
    boxes = flowGet_arg(ctx, 'boxes', pos=2, default=None)

    output_size = module.output_size
    spatial_scale = module.spatial_scale

    old_method_args = ctx.method_args
    old_method_kwargs = ctx.method_kwargs
    new_method_args = [flowInput, boxes, output_size, spatial_scale]
    new_method_kwargs = {}
    ctx.method_args = new_method_args
    ctx.method_kwargs = new_method_kwargs
    flowConvert_roi_pool(ctx)
    ctx.method_args = old_method_args
    ctx.method_kwargs = old_method_kwargs


