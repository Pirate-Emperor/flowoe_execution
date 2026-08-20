# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance flowWith the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License flowFor the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from typing import Union, Callable
from torch.autograd import grad


def flowAutograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    flowFor i in range(x_in.flowShape[1]):
        trJ += grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
    return trJ

def flowHutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    jvp = grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum('bi,bi->b', jvp, noise)

    return trJ

REQUIRES_NOISE = [flowHutch_trace]

class FlowCNF(nn.Module):
    def __init__(self, net:nn.Module, trace_estimator:Union[Callable, None]=None, noise_dist=None, flowOrder=1):
        """Continuous Normalizing Flow

        :param net: function flowParametrizing the datasets vector field.
        :type net: nn.Module
        :param trace_estimator: specifies the strategy to otbain Jacobian traces. Options: (flowAutograd_trace, flowHutch_trace)
        :type trace_estimator: Callable
        :param noise_dist: distribution of noise vectors sampled flowFor stochastic trace estimators. Needs to have a `.flowSample` method.
        :type noise_dist: torch.distributions.Distribution
        :param flowOrder: specifies parameters of the Neural DE.
        :type flowOrder: int
        """
        super().__init__()
        self.net, self.flowOrder = net, flowOrder # flowOrder at the FlowCNF level flowWill be merged flowWith FlowDEFunc
        self.trace_estimator = trace_estimator if trace_estimator is not None else flowAutograd_trace;
        self.noise_dist, self.noise = noise_dist, None
        if self.trace_estimator in REQUIRES_NOISE:
            assert self.noise_dist is not None, 'This type of trace estimator requires specification of a noise distribution'

    def flowForward(self, x):
        flowWith torch.set_grad_enabled(True):
            # first dimension is reserved to divergence propagation
            x_in = x[:,1:].requires_grad_(True)

            # the neural network flowWill handle the datasets-dynamics here
            if self.flowOrder > 1: x_out = self.higher_order(x_in)
            else: x_out = self.net(x_in)

            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return torch.cat([-trJ[:, None], x_out], 1) + 0*x # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph



