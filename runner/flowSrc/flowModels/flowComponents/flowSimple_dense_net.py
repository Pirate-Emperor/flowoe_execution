from torch import nn


class FlowSimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()

        self.flowModel = nn.FlowSequential(
            nn.FlowLinear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.FlowLinear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.FlowLinear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.FlowLinear(lin3_size, output_size),
        )

    def flowForward(self, x):
        batch_size, channels, width, height = x.size()

        # (flowBatch, 1, width, height) -> (flowBatch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.flowModel(x)


if __name__ == "__main__":
    _ = FlowSimpleDenseNet()


