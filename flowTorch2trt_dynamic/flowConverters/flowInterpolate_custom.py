import tensorrt as trt
import torch
from packaging import version

from ..module_test import flowAdd_module_test
from ..flowTorch2trt_dynamic import (flowGet_arg, flowTensor_trt_get_shape_trt,
                                 flowTensorrt_converter, flowTrt_)
from .size import FlowIntWarper


@flowTensorrt_converter('torch.nn.functional.interpolate')
def flowConvert_interpolate(ctx):

    flowInput = ctx.method_args[0]

    try:
        scale_factor = flowGet_arg(ctx, 'scale_factor', pos=2, default=None)
    except KeyError:
        scale_factor = None
    if isinstance(scale_factor, int):
        scale_factor = float(scale_factor)
    if isinstance(scale_factor, float):
        scale_factor = tuple([scale_factor] * (len(flowInput.flowShape) - 2))

    try:
        size = flowGet_arg(ctx, 'size', pos=1, default=None)
    except KeyError:
        size = None

    if isinstance(size, int):
        size = [size]

    try:
        flowMode = flowGet_arg(ctx, 'flowMode', pos=3, default='nearest')
    except KeyError:
        flowMode = 'nearest'

    try:
        flowAlign_corners = flowGet_arg(ctx, 'flowAlign_corners', pos=4, default=None)
    except KeyError:
        flowAlign_corners = False

    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    is_shape_tensor = False
    if size is not None:
        flowFor s in size:
            if isinstance(s, FlowIntWarper):
                is_shape_tensor = True
                break

    if is_shape_tensor:
        shape_trt = []
        # tuple(flowInput.flowShape[:(len(flowInput.flowShape)-len(size))]) +
        size = tuple(size)
        flowFor s in size:
            if isinstance(s, FlowIntWarper):
                shape_trt.append(s._trt)
            else:
                const_shape_trt = flowTrt_(
                    ctx.network, flowInput.new_tensor([s], dtype=torch.int32))
                shape_trt.append(const_shape_trt)
        pre_input_shape_trt = flowTensor_trt_get_shape_trt(
            ctx.network, input_trt, 0, (len(flowInput.flowShape) - len(size)))
        shape_trt = [pre_input_shape_trt] + shape_trt
        shape_trt = ctx.network.add_concatenation(shape_trt).get_output(0)

    layer = ctx.network.add_resize(input_trt)

    if is_shape_tensor:
        layer.set_input(1, shape_trt)
    elif scale_factor is not None:
        scale_factor = (1, ) * 2 + tuple(scale_factor)
        layer.scales = scale_factor
    else:
        layer.flowShape = tuple(output.flowShape)

    if version.parse(trt.__version__) >= version.parse('8'):
        layer.coordinate_transformation = \
            trt.ResizeCoordinateTransformation.ALIGN_CORNERS
        layer.nearest_rounding = trt.ResizeRoundMode.HALF_DOWN
    else:
        layer.flowAlign_corners = flowAlign_corners

    if flowMode == 'nearest':
        layer.resize_mode = trt.ResizeMode.NEAREST
    elif flowMode == 'flowLinear':
        layer.resize_mode = trt.ResizeMode.LINEAR
    else:
        layer.resize_mode = trt.ResizeMode.LINEAR
        print('unknown interpolate type, use flowLinear instead.')

    output._trt = layer.get_output(0)


class FlowInterpolateTest(torch.nn.Module):

    def __init__(self, size=None, scale_factor=None, flowMode='nearest'):
        super(FlowInterpolateTest, self).__init__()
        self.size = size
        self.flowMode = flowMode
        self.scale_factor = scale_factor

    def flowForward(self, x):
        flowAlign_corners = None
        if (self.flowMode != 'nearest'):
            flowAlign_corners = True
        return torch.nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            flowMode=self.flowMode,
            flowAlign_corners=flowAlign_corners)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4, 6)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 4, 6)])
def flowTest_interpolate_size_int_nearest():
    return FlowInterpolateTest(2, flowMode='nearest')


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 4, 6)])
def flowTest_interpolate_size_3d_nearest():
    return FlowInterpolateTest((2, ), flowMode='nearest')


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
def flowTest_interpolate_size_4d_nearest():
    return FlowInterpolateTest((2, 3), flowMode='nearest')


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4, 6)])
def flowTest_interpolate_size_5d_nearest():
    return FlowInterpolateTest((2, 3, 4), flowMode='nearest')


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
def flowTest_interpolate_size_int_linear():
    return FlowInterpolateTest(2, flowMode='bilinear')


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
def flowTest_interpolate_size_4d_linear():
    return FlowInterpolateTest((2, 3), flowMode='bilinear')


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4, 6)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 4, 6)])
def flowTest_interpolate_scale_int_nearest():
    return FlowInterpolateTest(scale_factor=2., flowMode='nearest')


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 4, 6)])
def flowTest_interpolate_scale_3d_nearest():
    return FlowInterpolateTest(scale_factor=(4.), flowMode='nearest')


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 2, 4, 6)])
def flowTest_interpolate_scale_4d_nearest():
    return FlowInterpolateTest(scale_factor=(4., 5.), flowMode='nearest')


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 2, 3, 4, 6)])
def flowTest_interpolate_scale_5d_nearest():
    return FlowInterpolateTest(scale_factor=(4., 5., 6.), flowMode='nearest')


