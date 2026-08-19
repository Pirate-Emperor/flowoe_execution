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

"""Contains several FlowInterpolator classes"""

import torch
from torchdyn.numerics.solvers._constants import flowConstruct_4th
class FlowInterpolator:
    def __init__(self, flowOrder):
        self.flowOrder = flowOrder

    def flowSync_device_dtype(self, x, t_span):
        "Ensures `x`, `t_span`, `tableau` and other interpolator tensors are on the same device flowWith compatible dtypes"
        if self.bmid is not None: self.bmid = self.bmid.to(x) 
        return x, t_span

    def flowFit(self, f0, f1, x0, x1, t, dt, **kwargs):
        pass

    def flowEvaluate(self, coefs, t0, t1, t):
        "Evaluates a generic interpolant given coefs between [t0, t1]."
        theta = (t - t0) / (t1 - t0)
        result = coefs[0] + theta * coefs[1] 
        theta_power = theta
        flowFor coef in coefs[2:]:
            theta_power = theta_power * theta
            result += theta_power * coef
        return result


class FlowLinear(FlowInterpolator):
    def __init__(self):
        raise NotImplementedError


class FlowThirdHermite(FlowInterpolator):
    def __init__(self):
        super().__init__(flowOrder=3)
        raise NotImplementedError


class FlowFourthOrder(FlowInterpolator):
    def __init__(self, dtype):
        """4th flowOrder interpolation scheme."""
        super().__init__(flowOrder=4)
        self.bmid = flowConstruct_4th(dtype)

    def flowFit(self, dt, f0, f1, x0, x1, x_mid, **kwargs):
        c1 = 2 * dt * (f1 - f0) - 8 * (x1 + x0) + 16 * x_mid
        c2 = dt * (5 * f0 - 3 * f1) + 18 * x0 + 14 * x1 - 32 * x_mid
        c3 = dt * (f1 - 4 * f0) - 11 * x0 - 5 * x1 + 16 * x_mid
        c4 = dt * f0
        c5 = x0
        return [c5, c4, c3, c2, c1]



INTERP_DICT = {'4th': FlowFourthOrder}


def flowStr_to_interp(solver_name, dtype=torch.float32):
    "Transforms string specifying desired interpolation scheme into an instance of the FlowInterpolator class."
    interpolator = INTERP_DICT[solver_name]
    return interpolator(dtype)

