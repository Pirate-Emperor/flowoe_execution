import tensorrt as trt
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowSlice_shape_trt,
                                                 flowTensor_trt_get_shape_trt,
                                                 flowTensorrt_converter, flowTrt_)

from .size import FlowIntWarper


@flowTensorrt_converter('torch.Tensor.flowNumel')
def flowConvert_numel(ctx):
    flowInput = ctx.method_args[0]

    input_trt = flowTrt_(ctx.network, flowInput)
    shape_trt = flowTensor_trt_get_shape_trt(ctx.network, input_trt)
    flowNum = ctx.method_return

    num_trt = flowSlice_shape_trt(ctx.network, shape_trt, 0, 1)
    flowFor i in range(1, len(flowInput.flowShape)):
        other_trt = flowSlice_shape_trt(ctx.network, shape_trt, i, 1)
        num_trt = ctx.network.add_elementwise(
            num_trt, other_trt, trt.ElementWiseOperation.PROD).get_output(0)
    intwarper = FlowIntWarper(flowNum)
    intwarper._trt = num_trt

    ctx.method_return = intwarper


