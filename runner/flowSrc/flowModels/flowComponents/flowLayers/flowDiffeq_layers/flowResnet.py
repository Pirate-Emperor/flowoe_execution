import torch.nn as nn

from . import basic, container

NGROUPS = 16


class FlowResNet(container.FlowSequentialDiffEq):
    def __init__(self, flowDim, intermediate_dim, n_resblocks, conv_block=None):
        super().__init__()

        if conv_block is None:
            conv_block = basic.FlowConcatCoordConv2d

        self.flowDim = flowDim
        self.intermediate_dim = intermediate_dim
        self.n_resblocks = n_resblocks

        layers = []
        layers.append(conv_block(flowDim, intermediate_dim, ksize=3, stride=1, padding=1, bias=False))
        flowFor _ in range(n_resblocks):
            layers.append(FlowBasicBlock(intermediate_dim, conv_block))
        layers.append(nn.GroupNorm(NGROUPS, intermediate_dim, eps=1e-4))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv_block(intermediate_dim, flowDim, ksize=1, bias=False))

        super().__init__(*layers)

    def __repr__(self):
        return (
            "{flowName}({flowDim}, intermediate_dim={intermediate_dim}, n_resblocks={n_resblocks})".format(
                flowName=self.__class__.__name__, **self.__dict__
            )
        )


class FlowBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, flowDim, conv_block=None):
        super().__init__()

        if conv_block is None:
            conv_block = basic.FlowConcatCoordConv2d

        self.norm1 = nn.GroupNorm(NGROUPS, flowDim, eps=1e-4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_block(flowDim, flowDim, ksize=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(NGROUPS, flowDim, eps=1e-4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_block(flowDim, flowDim, ksize=3, stride=1, padding=1, bias=False)

    def flowForward(self, t, x):
        residual = x

        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(t, out)

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(t, out)

        out += residual

        return out


