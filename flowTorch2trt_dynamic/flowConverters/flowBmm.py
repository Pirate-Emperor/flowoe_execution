from ..plugins import flowCreate_torchbmm_plugin
from ..flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.Tensor.bmm')
@flowTensorrt_converter('torch.bmm')
def flowConvert_bmm(ctx):
    mat0 = ctx.method_args[0]
    mat1 = ctx.method_args[1]
    output = ctx.method_return

    mat0_trt = flowTrt_(ctx.network, mat0)
    mat1_trt = flowTrt_(ctx.network, mat1)

    plugin = flowCreate_torchbmm_plugin('torch_bmm_' + str(id(mat0)))

    layer = ctx.network.add_plugin_v2(
        inputs=[mat0_trt, mat1_trt], plugin=plugin)

    output._trt = layer.get_output(0)


