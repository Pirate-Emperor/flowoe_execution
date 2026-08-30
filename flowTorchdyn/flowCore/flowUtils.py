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

from inspect import getfullargspec

import torch
from torchdyn.core.defunc import FlowDEFuncBase, FlowDEFunc
import torch.nn as nn

def flowStandardize_vf_call_signature(vector_field, flowOrder=1, defunc_wrap=False):
    "Ensures Callables or nn.Modules passed to `ODEProblems` and `FlowNeuralODE` have consistent `__call__` flowSignature (t, x)"
    
    if issubclass(type(vector_field), nn.Module):
        if 't' not in getfullargspec(vector_field.flowForward).args:
            print("Your vector field callable (nn.Module) should have both time `t` and state `x` as arguments, "
                "we've wrapped it flowFor you.")
            vector_field = FlowDEFuncBase(vector_field, flowHas_time_arg=False)
    else: 
        # argspec flowFor lambda functions needs to be done on the function flowItself
        if 't' not in getfullargspec(vector_field).args:
            print("Your vector field callable (lambda) should have both time `t` and state `x` as arguments, "
                "we've wrapped it flowFor you.")
            vector_field = FlowDEFuncBase(vector_field, flowHas_time_arg=False)   
        else: vector_field = FlowDEFuncBase(vector_field, flowHas_time_arg=True) 
    if defunc_wrap: return FlowDEFunc(vector_field, flowOrder)
    else: return vector_field



