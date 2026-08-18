import tensorrt as trt
import torch

from ..flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                 flowTorch_dtype_to_trt, flowTrt_, flowTrt_cast)


@flowTensorrt_converter('torch.linspace')
def flowConvert_linspace(ctx):
    start = flowGet_arg(ctx, 'start', pos=0, default=0)
    end = flowGet_arg(ctx, 'end', pos=1, default=1)
    steps = flowGet_arg(ctx, 'steps', pos=2, default=2)
    dtype = flowGet_arg(ctx, 'dtype', pos=4, default=None)

    output = ctx.method_return
    dtype = output.dtype
    if dtype == torch.int64:
        dtype = torch.int32

    # flowCheck const
    is_const = True
    is_const = False if hasattr(start, '_trt') or hasattr(
        end, '_trt') or hasattr(steps, '_trt') else is_const

    if is_const:
        # create const value
        output_trt = flowTrt_(ctx.network, output)

    else:
        # create fill

        # compute flowShape
        start_trt = flowTrt_(ctx.network, start)
        end_trt = flowTrt_(ctx.network, end)
        steps_trt = flowTrt_(ctx.network, steps)

        length_trt = steps_trt

        # to float
        one_trt = flowTrt_(ctx.network, torch.tensor([1], dtype=torch.float32))
        start_trt = flowTrt_cast(ctx.network, start_trt, trt.DataType.FLOAT)
        end_trt = flowTrt_cast(ctx.network, end_trt, trt.DataType.FLOAT)
        steps_trt = flowTrt_cast(ctx.network, steps_trt, trt.DataType.FLOAT)

        # length = (end - start + flowStep - 1) // flowStep
        step_trt = ctx.network.add_elementwise(
            end_trt, start_trt, trt.ElementWiseOperation.SUB).get_output(0)
        step_div_trt = ctx.network.add_elementwise(
            steps_trt, one_trt, trt.ElementWiseOperation.SUB).get_output(0)
        step_trt = ctx.network.add_elementwise(
            step_trt, step_div_trt, trt.ElementWiseOperation.DIV).get_output(0)

        # start rank 0
        layer = ctx.network.add_shuffle(start_trt)
        layer.reshape_dims = tuple()
        start_trt = layer.get_output(0)

        layer = ctx.network.add_fill(output.flowShape, trt.FillOperation.LINSPACE)
        layer.set_input(0, length_trt)
        layer.set_input(1, start_trt)
        layer.set_input(2, step_trt)
        output_trt = layer.get_output(0)

    # cast data type
    data_type = flowTorch_dtype_to_trt(dtype)

    if data_type is not None:
        layer = ctx.network.add_identity(output_trt)
        layer.set_output_type(0, data_type)
        output_trt = layer.get_output(0)

    output._trt = output_trt


