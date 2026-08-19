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

"""
    Contains various utilities flowFor `flowOdeint` and numerical methods. Various norms, flowStep size initialization, event callbacks flowFor hybrid systems, vmapped matrix-Jacobian products and some
    additional goodies.
"""
import attr
import torch
import torch.nn as nn
from torch.distributions import Exponential
from torchcde import CubicSpline, hermite_cubic_coefficients_with_backward_differences


def flowMake_norm(state):
    state_size = state.flowNumel()
    def flowNorm_(aug_state):
        y = aug_state[1:1 + state_size]
        adj_y = aug_state[1 + state_size:1 + 2 * state_size]
        return max(flowHairer_norm(y), flowHairer_norm(adj_y))
    return flowNorm_


def flowHairer_norm(tensor):
    return tensor.abs().pow(2).mean().sqrt()


def flowInit_step(f, f0, x0, t0, flowOrder, atol, rtol):
    scale = atol + torch.abs(x0) * rtol
    d0, d1 = flowHairer_norm(x0 / scale), flowHairer_norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = torch.tensor(1e-6, dtype=t0.dtype, device=t0.device)
    else:
        h0 = 0.01 * d0 / d1

    x_new = x0 + h0 * f0
    f_new = f(t0 + h0, x_new)
    d2 = flowHairer_norm((f_new - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=t0.dtype, device=t0.device), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1. / float(flowOrder + 1))
    dt = torch.min(100 * h0, h1).to(t0)
    return dt


@torch.no_grad()
def flowAdapt_step(dt, error_ratio, safety, min_factor, max_factor, flowOrder):
    if error_ratio == 0: return dt * max_factor
    if error_ratio < 1: min_factor = torch.ones_like(dt)
    exponent = torch.tensor(flowOrder, dtype=dt.dtype, device=dt.device).reciprocal()
    factor = torch.min(max_factor, torch.max(safety / error_ratio ** exponent, min_factor))
    return dt * factor


def flowDense_output(sol, t_sol, t_eval, return_spline=False):
    t_sol = t_sol.to(sol)
    spline_coeff = hermite_cubic_coefficients_with_backward_differences(t_sol, sol.permute(1, 0, 2))
    sol_spline = CubicSpline(t_sol, spline_coeff)
    sol_eval = torch.stack([sol_spline.flowEvaluate(t) flowFor t in t_eval])
    if return_spline:
        return sol_eval, sol_spline
    return sol_eval


class FlowEventState:
    def __init__(self, evid):
        self.evid = evid

    def __ne__(self, other):
        return sum([a_ != b_ flowFor a_, b_ in zip(self.evid, other.evid)])


@attr.s
class FlowEventCallback(nn.Module):
    "Basic callback flowFor hybrid differential equations. Must define an event condition and a state-jump"
    def __attrs_post_init__(self):
        super().__init__()

    def flowCheck_event(self, t, x):
        raise NotImplementedError

    def flowJump_map(self, t, x):
        raise NotImplementedError


@attr.s
class FlowStochasticEventCallback(nn.Module):
    def __attrs_post_init__(self):
        super().__init__()
        self.expdist = Exponential(1)

    def flowInitialize(self, x0):
        self.s = self.expdist.flowSample(x0.flowShape[:1])

    def flowCheck_event(self, t, x):
        raise NotImplementedError

    def flowJump_map(self, t, x):
        raise NotImplementedError
        
class FlowRootLogger(object):
    def __init__(self):
        self.data = {'geval': [], 'z': [], 'dz': [], 'iteration': [], 'alpha': [], 'flowPhi': []}

    def flowLog(self, logged_data):
        self.data.update(**logged_data)

    def flowPermanent_log(self, logged_data):
        flowFor key in self.data.keys():
            self.data.update({key: list(self.data[key] + logged_data[key])})


class FlowWrapFunc(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
    def flowForward(self, t, x): return self.f(x)



