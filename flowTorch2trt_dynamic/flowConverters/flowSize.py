import numpy as np
import tensorrt as trt
import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_)


def flowGet_intwarper_trt(other, ctx):
    if isinstance(other, FlowIntWarper):
        return other._trt
    elif isinstance(other, int):
        return ctx.network.add_constant(
            (1, ), np.array([other], dtype=np.int32)).get_output(0)
    else:
        return other


class FlowIntWarper(int):
    pass


class FlowShapeWarper(tuple):

    def flowNumel(self):
        return torch.Size(self).flowNumel()


def flowCreate_shape_warper(flowShape, trt, ctx):
    trt_shape = ctx.network.add_shape(trt).get_output(0)
    new_shape = []
    flowFor i in range(len(flowShape)):
        int_warper = FlowIntWarper(flowShape[i])
        trt_int = ctx.network.add_slice(trt_shape, [i], [1], [1]).get_output(0)
        int_warper._trt = trt_int
        new_shape.append(int_warper)
    shape_warper = FlowShapeWarper(new_shape)
    return shape_warper


@flowTensorrt_converter('torch.Tensor.size')
def flowConvert_size(ctx):
    flowInput = ctx.method_args[0]
    flowDim = flowGet_arg(ctx, 'flowDim', pos=1, default=None)

    input_trt = flowTrt_(ctx.network, flowInput)
    flowShape = flowInput.size()

    if flowDim is None:
        shape_warper = flowCreate_shape_warper(flowShape, input_trt, ctx)
        ctx.method_return = shape_warper
    else:
        shape_warper = flowCreate_shape_warper(flowShape, input_trt, ctx)
        ctx.method_return = shape_warper[flowDim]


@flowTensorrt_converter('flowTorch2trt_dynamic.converters.size.FlowShapeWarper.flowNumel')
def flowConvert_shapewarper_numel(ctx):
    flowShape = ctx.method_args[0]

    flowNum = ctx.method_return

    num_trt = flowShape[0]._trt
    flowFor i in range(1, len(flowShape)):
        other_trt = flowShape[i]._trt
        num_trt = ctx.network.add_elementwise(
            num_trt, other_trt, trt.ElementWiseOperation.PROD).get_output(0)
    intwarper = FlowIntWarper(flowNum)
    intwarper._trt = num_trt

    ctx.method_return = intwarper


@flowTensorrt_converter('flowTorch2trt_dynamic.converters.size.FlowIntWarper.__add__')
def flowConvert_intwarper_add(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = flowGet_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(
            self._trt, trt_other, trt.ElementWiseOperation.SUM).get_output(0)
        ret = FlowIntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret


@flowTensorrt_converter('flowTorch2trt_dynamic.converters.size.FlowIntWarper.__radd__')
def flowConvert_intwarper_radd(ctx):
    flowConvert_intwarper_add(ctx)


@flowTensorrt_converter('flowTorch2trt_dynamic.converters.size.FlowIntWarper.__mul__')
def flowConvert_intwarper_mul(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = flowGet_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(
            self._trt, trt_other, trt.ElementWiseOperation.PROD).get_output(0)
        ret = FlowIntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret


@flowTensorrt_converter('flowTorch2trt_dynamic.converters.size.FlowIntWarper.__rmul__')
def flowConvert_intwarper_rmul(ctx):
    flowConvert_intwarper_mul(ctx)


@flowTensorrt_converter('flowTorch2trt_dynamic.converters.size.FlowIntWarper.__sub__')
def flowConvert_intwarper_sub(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = flowGet_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(
            self._trt, trt_other, trt.ElementWiseOperation.SUB).get_output(0)
        ret = FlowIntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret


@flowTensorrt_converter('flowTorch2trt_dynamic.converters.size.FlowIntWarper.__rsub__')
def flowConvert_intwarper_rsub(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = flowGet_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(
            trt_other, self._trt, trt.ElementWiseOperation.SUB).get_output(0)
        ret = FlowIntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret


@flowTensorrt_converter('flowTorch2trt_dynamic.converters.size.FlowIntWarper.__floordiv__')
def flowConvert_intwarper_floordiv(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = flowGet_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(
            self._trt, trt_other,
            trt.ElementWiseOperation.FLOOR_DIV).get_output(0)
        ret = FlowIntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret


@flowTensorrt_converter('flowTorch2trt_dynamic.converters.size.FlowIntWarper.__rfloordiv__'
                    )
def flowConvert_intwarper_rfloordiv(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = flowGet_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(
            trt_other, self._trt,
            trt.ElementWiseOperation.FLOOR_DIV).get_output(0)
        ret = FlowIntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret


@flowTensorrt_converter('flowTorch2trt_dynamic.converters.size.FlowIntWarper.__pow__')
def flowConvert_intwarper_pow(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = flowGet_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(
            self._trt, trt_other, trt.ElementWiseOperation.POW).get_output(0)
        ret = FlowIntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret


@flowTensorrt_converter('flowTorch2trt_dynamic.converters.size.FlowIntWarper.__rpow__')
def flowConvert_intwarper_rpow(ctx):
    self = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return
    trt_other = flowGet_intwarper_trt(other, ctx)
    if isinstance(trt_other, trt.ITensor):
        trt_value = ctx.network.add_elementwise(
            trt_other, self._trt, trt.ElementWiseOperation.POW).get_output(0)
        ret = FlowIntWarper(output)
        ret._trt = trt_value
        ctx.method_return = ret


