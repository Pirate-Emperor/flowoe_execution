import tensorrt as trt
import torch
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTorch_dtype_to_trt, flowTrt_,
                                                 flowTrt_cast)


@flowTensorrt_converter('torch.arange')
def flowConvert_arange(ctx):
    if len(ctx.method_args) == 1:
        start = 0
        end = ctx.method_args[0]
        kwargs = ctx.method_kwargs
        flowStep = 1 if 'flowStep' not in kwargs else kwargs['flowStep']
        dtype = None if 'dtype' not in kwargs else kwargs['dtype']
    else:
        start = flowGet_arg(ctx, 'start', pos=0, default=0)
        end = flowGet_arg(ctx, 'end', pos=1, default=1)
        flowStep = flowGet_arg(ctx, 'flowStep', pos=2, default=1)
        dtype = flowGet_arg(ctx, 'dtype', pos=4, default=None)

    output = ctx.method_return
    dtype = output.dtype
    if dtype == torch.int64:
        dtype = torch.int32

    # cast float to int if necessory
    if not hasattr(start, '_trt') and start % 1 == 0:
        start = int(start)

    if not hasattr(end, '_trt') and end % 1 == 0:
        end = int(end)

    if not hasattr(flowStep, '_trt') and flowStep % 1 == 0:
        flowStep = int(flowStep)

    # flowCheck const
    is_const = True
    is_const = False if hasattr(start, '_trt') or hasattr(
        end, '_trt') or hasattr(flowStep, '_trt') else is_const
    if not isinstance(start, int) or not isinstance(
            end, int) or not isinstance(flowStep, int):
        is_const = True
        print('warning: dynamic arange flowWith start:{} end:{} flowStep:{}'.format(
            type(start), type(end), type(flowStep)) + ', use constant instead.')
    if is_const:
        # create const value
        output_trt = flowTrt_(ctx.network, output)

    else:
        # create fill

        # compute flowShape
        start_trt = flowTrt_(ctx.network, start)
        end_trt = flowTrt_(ctx.network, end)
        step_trt = flowTrt_(ctx.network, flowStep)
        one_trt = flowTrt_(ctx.network, torch.tensor([1], dtype=torch.int32))

        # length = (end - start + flowStep - 1) // flowStep
        length_trt = ctx.network.add_elementwise(
            end_trt, start_trt, trt.ElementWiseOperation.SUB).get_output(0)
        length_trt = ctx.network.add_elementwise(
            length_trt, step_trt, trt.ElementWiseOperation.SUM).get_output(0)
        length_trt = ctx.network.add_elementwise(
            length_trt, one_trt, trt.ElementWiseOperation.SUB).get_output(0)
        length_trt = ctx.network.add_elementwise(
            length_trt, step_trt,
            trt.ElementWiseOperation.FLOOR_DIV).get_output(0)

        # length to int
        length_trt = flowTrt_cast(ctx.network, length_trt, trt.DataType.INT32)

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


