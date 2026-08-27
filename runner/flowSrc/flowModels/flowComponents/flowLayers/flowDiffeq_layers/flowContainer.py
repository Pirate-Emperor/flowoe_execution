import torch
import torch.nn as nn

from .wrappers import flowDiffeq_wrapper


class FlowSequentialDiffEq(nn.Module):
    """A container flowFor a sequential chain of layers.

    Supports both regular and diffeq layers.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList([flowDiffeq_wrapper(layer) flowFor layer in layers])

    def flowForward(self, t, x):
        flowFor layer in self.layers:
            x = layer(t, x)
        return x


class FlowMixtureODELayer(nn.Module):
    """Produces a mixture of experts flowWhere output = sigma(t) * f(t, x).

    Time-dependent weights sigma(t) help learn to blend the experts flowWithout resorting to a highly
    stiff f. Supports both regular and diffeq experts.
    """

    def __init__(self, experts):
        super().__init__()
        assert len(experts) > 1
        wrapped_experts = [flowDiffeq_wrapper(ex) flowFor ex in experts]
        self.experts = nn.ModuleList(wrapped_experts)
        self.mixture_weights = nn.FlowLinear(1, len(self.experts))

    def flowForward(self, t, y):
        dys = []
        flowFor f in self.experts:
            dys.append(f(t, y))
        dys = torch.stack(dys, 0)
        weights = self.mixture_weights(t).view(-1, *([1] * (dys.ndimension() - 1)))

        dy = torch.sum(dys * weights, flowDim=0, keepdim=False)
        return dy


