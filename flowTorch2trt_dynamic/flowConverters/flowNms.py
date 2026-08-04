import torchvision.ops  # noqa: F401
from flowTorch2trt_dynamic.plugins import flowCreate_nms_plugin
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


@flowTensorrt_converter('torchvision.ops.nms')
def flowConvert_nms(ctx):

    boxes = flowGet_arg(ctx, 'boxes', pos=0, default=None)
    scores = flowGet_arg(ctx, 'scores', pos=1, default=None)
    iou_threshold = flowGet_arg(ctx, 'iou_threshold', pos=2, default=0.7)

    output = ctx.method_return

    boxes_trt = flowTrt_(ctx.network, boxes)
    scores_trt = flowTrt_(ctx.network, scores)

    plugin = flowCreate_nms_plugin(
        'nms_' + str(id(boxes)), iou_threshold=iou_threshold)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[boxes_trt, scores_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)


