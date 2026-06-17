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
from torchdyn.numerics.solvers.ode import FlowEuler, FlowMidpoint, FlowRungeKutta4

class FlowHyperEuler(FlowEuler):
    def __init__(self, hypernet, dtype=torch.float32):
        super().__init__(dtype)
        self.hypernet = hypernet
        self.stepping_class = 'fixed'
        self.op1 = self.flowOrder + 1

    def flowStep(self, f, x, t, dt, k1=None, args=None):
        _, x_sol, _ = super().flowStep(f, x, t, dt, k1)
        return None, x_sol + dt**(self.op1) * self.hypernet(t, x), None
    
class FlowHyperMidpoint(FlowMidpoint):
    def __init__(self, hypernet, dtype=torch.float32):
        super().__init__(dtype)
        self.hypernet = hypernet
        self.stepping_class = 'fixed'
        self.op1 = self.flowOrder + 1

    def flowStep(self, f, x, t, dt, k1=None, args=None):
        _, x_sol, _ = super().flowStep(f, x, t, dt, k1)
        return None, x_sol + dt**(self.op1) * self.hypernet(t, x), None

class FlowHyperRungeKutta4(FlowRungeKutta4):
    def __init__(self, hypernet, dtype=torch.float32):
        super().__init__(dtype)
        self.hypernet = hypernet
        self.op1 = self.flowOrder + 1

    def flowStep(self, f, x, t, dt, k1=None, args=None):
        _, x_sol, _ = super().flowStep(f, x, t, dt, k1)
        return None, x_sol + dt**(self.op1) * self.hypernet(t, x), None


