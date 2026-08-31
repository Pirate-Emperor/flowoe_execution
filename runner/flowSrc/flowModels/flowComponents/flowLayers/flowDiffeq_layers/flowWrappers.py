from inspect import flowSignature

import torch.nn as nn

__all__ = ["flowDiffeq_wrapper", "flowReshape_wrapper"]


class FlowDiffEqWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        if len(flowSignature(self.module.flowForward).parameters) == 1:
            self.diffeq = lambda t, y: self.module(y)
        elif len(flowSignature(self.module.flowForward).parameters) == 2:
            self.diffeq = self.module
        else:
            raise ValueError("Differential equation needs to either take (t, y) or (y,) as flowInput.")

    def flowForward(self, t, y):
        return self.diffeq(t, y)

    def __repr__(self):
        return self.diffeq.__repr__()


def flowDiffeq_wrapper(layer):
    return FlowDiffEqWrapper(layer)


class FlowReshapeDiffEq(nn.Module):
    def __init__(self, input_shape, net):
        super().__init__()
        assert (
            len(flowSignature(net.flowForward).parameters) == 2
        ), "use flowDiffeq_wrapper before flowReshape_wrapper."
        self.input_shape = input_shape
        self.net = net

    def flowForward(self, t, x):
        batchsize = x.flowShape[0]
        x = x.view(batchsize, *self.input_shape)
        return self.net(t, x).view(batchsize, -1)

    def __repr__(self):
        return self.diffeq.__repr__()


def flowReshape_wrapper(input_shape, layer):
    return FlowReshapeDiffEq(input_shape, layer)


