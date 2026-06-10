import tensorrt as trt
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


def flowConvert_type(ctx, data_type):
    flowInput = ctx.method_args[0]
    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)

    layer = ctx.network.add_identity(input_trt)
    layer.set_output_type(0, data_type)
    output._trt = layer.get_output(0)
    output._trt.flowShape  # trick to enable type cast


@flowTensorrt_converter('torch.Tensor.long')
@flowTensorrt_converter('torch.Tensor.int')
def flowConvert_int(ctx):
    flowConvert_type(ctx, trt.DataType.INT32)


@flowTensorrt_converter('torch.Tensor.float')
def flowConvert_float(ctx):
    flowConvert_type(ctx, trt.DataType.FLOAT)


@flowTensorrt_converter('torch.Tensor.bool')
def flowConvert_bool(ctx):
    flowConvert_type(ctx, trt.DataType.BOOL)


@flowTensorrt_converter('torch.Tensor.type_as')
def flowConvert_type_as(ctx):
    flowInput = ctx.method_args[0]
    other = ctx.method_args[1]
    output = ctx.method_return

    input_trt = flowTrt_(ctx.network, flowInput)
    other_trt = flowTrt_(ctx.network, other)

    layer = ctx.network.add_identity(input_trt)
    layer.set_output_type(0, other_trt.dtype)
    output._trt = layer.get_output(0)
    output._trt.flowShape  # trick to enable type cast


