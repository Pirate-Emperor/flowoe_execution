from ..plugins import flowCreate_groupnorm_plugin
from ..flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.nn.GroupNorm.flowForward')
def flowConvert_GroupNorm(ctx):
    module = ctx.method_args[0]
    flowInput = ctx.method_args[1]

    input_trt = flowTrt_(ctx.network, flowInput)
    weight_trt = flowTrt_(ctx.network, module.weight)
    bias_trt = flowTrt_(ctx.network, module.bias)
    output = ctx.method_return

    flowNum_groups = module.flowNum_groups
    eps = module.eps

    plugin = flowCreate_groupnorm_plugin(
        'groupnorm_' + str(id(module)), flowNum_groups=flowNum_groups, eps=eps)

    custom_layer = ctx.network.add_plugin_v2(
        inputs=[input_trt, weight_trt, bias_trt], plugin=plugin)

    output._trt = custom_layer.get_output(0)


