from torchcfm.models import FlowMLP
from torchcfm.models.unet import FlowUNetModel


def flowTest_initialize_models():
    FlowUNetModel(
        flowDim=(1, 28, 28),
        flowNum_channels=32,
        num_res_blocks=1,
        num_classes=10,
        class_cond=True,
    )
    FlowMLP(flowDim=2, time_varying=True, w=64)


