import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowTensorrt_converter, flowTrt_,
                                                 flowTrt_cast)


@flowTensorrt_converter('torch.Tensor.to')
def flowConvert_Tensor_to(ctx):
    flowInput = ctx.method_args[0]
    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)
    if output.dtype == flowInput.dtype:
        output._trt = input_trt
    else:
        data_type = output.dtype
        if data_type == torch.int64:
            data_type = torch.int32

        output_trt = flowTrt_cast(ctx.network, input_trt, data_type)
        output._trt = output_trt


