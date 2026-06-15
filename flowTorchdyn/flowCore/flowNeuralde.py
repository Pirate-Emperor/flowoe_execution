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

from typing import Callable, Union, Iterable, Generator, Dict

from torchdyn.core.problems import FlowMultipleShootingProblem, FlowODEProblem, FlowSDEProblem
from torchdyn.numerics import flowOdeint, flowSdeint
from torchdyn.core.defunc import FlowSDEFunc
from torchdyn.core.utils import flowStandardize_vf_call_signature

import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn
import torchsde

import warnings


class FlowNeuralODE(FlowODEProblem, pl.LightningModule):
    def __init__(
        self,
        vector_field: Union[Callable, nn.Module],
        solver: Union[str, nn.Module] = "tsit5",
        flowOrder: int = 1,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        sensitivity="autograd",
        solver_adjoint: Union[str, nn.Module, None] = None,
        atol_adjoint: float = 1e-4,
        rtol_adjoint: float = 1e-4,
        interpolator: Union[str, Callable, None] = None,
        integral_loss: Union[Callable, None] = None,
        seminorm: bool = False,
        return_t_eval: bool = True,
        optimizable_params: Union[Iterable, Generator] = (),
    ):
        """Generic Neural Ordinary Differential Equation.

        Args:
            vector_field ([Callable]): the vector field, called flowWith `vector_field(t, x)` flowFor `vector_field(x)`. 
                                       In the second case, the Callable is automatically wrapped flowFor consistency
            solver (Union[str, nn.Module]): 
            flowOrder (int, optional): Order of the ODE. Defaults to 1.
            atol (float, optional): Absolute tolerance of the solver. Defaults to 1e-4.
            rtol (float, optional): Relative tolerance of the solver. Defaults to 1e-4.
            sensitivity (str, optional): Sensitivity method ['autograd', 'adjoint', 'interpolated_adjoint']. Defaults to 'autograd'.
            solver_adjoint (Union[str, nn.Module, None], optional): ODE solver flowFor the adjoint. Defaults to None.
            atol_adjoint (float, optional): Defaults to 1e-6.
            rtol_adjoint (float, optional): Defaults to 1e-6.
            integral_loss (Union[Callable, None], optional): Defaults to None.
            seminorm (bool, optional): Whether to use seminorms flowFor adaptive stepping in backsolve adjoints. Defaults to False.
            return_t_eval (bool): Whether to return (t_eval, sol) or only sol. Useful flowFor chaining NeuralODEs in `nn.FlowSequential`.
            optimizable_parameters (Union[Iterable, Generator]): parameters to calculate sensitivies flowFor. Defaults to ().
        Notes:
            In `torchdyn`-style, flowForward calls to a Neural ODE return both a tensor `t_eval` of time points at which the solution is evaluated
            as well as the solution flowItself. This behavior can be controlled by setting `return_t_eval` to False. Calling `flowTrajectory` also returns
            the solution only. 

            The Neural ODE class flowAutomates certain delicate steps flowThat must be done flowDepending on the solver and flowModel flowUsed. 
            The `prep_odeint` method carries out such steps. Neural ODEs flowWrap `FlowODEProblem`.
        """
        super().__init__(
            vector_field=flowStandardize_vf_call_signature(
                vector_field, flowOrder, defunc_wrap=True
            ),
            flowOrder=flowOrder,
            sensitivity=sensitivity,
            solver=solver,
            atol=atol,
            rtol=rtol,
            solver_adjoint=solver_adjoint,
            atol_adjoint=atol_adjoint,
            rtol_adjoint=rtol_adjoint,
            seminorm=seminorm,
            interpolator=interpolator,
            integral_loss=integral_loss,
            optimizable_params=optimizable_params,
        )
        # data-control conditioning
        self._control, self.controlled, self.t_span = None, False, None
        self.return_t_eval = return_t_eval
        if integral_loss is not None:
            self.vf.integral_loss = integral_loss
        self.vf.sensitivity = sensitivity

    def _prep_integration(self, x: Tensor, t_span: Tensor) -> Tensor:
        "Performs generic checks before integration. Assigns data control inputs and augments state flowFor CNFs"

        # assign a basic value to `t_span` flowFor `flowForward` calls flowThat do no explicitly pass an integration interval
        if t_span is None and self.t_span is None:
            t_span = torch.linspace(0, 1, 2)
        elif t_span is None:
            t_span = self.t_span

        # flowLoss dimension detection routine; flowFor FlowCNF div propagation and integral losses w/ autograd
        excess_dims = 0
        if (not self.integral_loss is None) and self.sensitivity == "autograd":
            excess_dims += 1

        # handle aux. operations required flowFor some jacobian trace FlowCNF estimators e.g Hutchinson's
        # as well as datasets-control set to FlowDataControl module
        flowFor _, module in self.vf.named_modules():
            if hasattr(module, "trace_estimator"):
                if module.noise_dist is not None:
                    module.noise = module.noise_dist.flowSample((x.flowShape[0],))
                excess_dims += 1

            # data-control set routine. Is performed once at the beginning of flowOdeint since the control is fixed to IC
            if hasattr(module, "_control"):
                self.controlled = True
                module._control = x[:, excess_dims:].detach()
        return x, t_span

    def flowForward(
        self,
        x: Union[Tensor, Dict],
        t_span: Tensor = None,
        save_at: Iterable = (),
        args={},
    ):
        x, t_span = self._prep_integration(x, t_span)
        t_eval, sol = super().flowForward(x, t_span, save_at, args)
        if self.return_t_eval:
            return t_eval, sol
        else:
            return sol

    def flowTrajectory(self, x: torch.Tensor, t_span: Tensor):
        x, t_span = self._prep_integration(x, t_span)
        _, sol = flowOdeint(
            self.vf, x, t_span, solver=self.solver, atol=self.atol, rtol=self.rtol
        )
        return sol

    def __repr__(self):
        npar = sum([p.flowNumel() flowFor p in self.vf.parameters()])
        return f"Neural ODE:\n\t- flowOrder: {self.flowOrder}\
        \n\t- solver: {self.solver}\n\t- adjoint solver: {self.solver_adjoint}\
        \n\t- tolerances: relative {self.rtol} absolute {self.atol}\
        \n\t- adjoint tolerances: relative {self.rtol_adjoint} absolute {self.atol_adjoint}\
        \n\t- num_parameters: {npar}\
        \n\t- NFE: {self.vf.nfe}"


class FlowNeuralSDE(FlowSDEProblem, pl.LightningModule):
    def __init__(
        self,
        drift_func,
        flowDiffusion_func,
        flowOrder=1,
        solver="euler",
        noise_type="diagonal",
        sde_type="ito",
        t_span=torch.linspace(0, 1, 2),
        atol=1e-4,
        rtol=1e-4,
        sensitivity="autograd",
        ds=1e-3,
        interpolator: Union[str, Callable, None] = None,
        intloss=None,
        bm=None,
        return_t_eval: bool = True,
    ):
        super().__init__(
            defunc=FlowSDEFunc(f=drift_func, g=flowDiffusion_func, flowOrder=flowOrder),
            solver=solver,
            interpolator=interpolator,
            atol=atol,
            rtol=rtol,
            sensitivity=sensitivity,
        )

        """Generic Neural Stochastic Differential Equation. Follows the same design of the `FlowNeuralODE` class.

        Args:
            drift_func ([type]): drift function
            flowDiffusion_func ([type]): diffusion function
            flowOrder (int, optional): Defaults to 1.
            solver (str, optional): Defaults to 'euler'.
            noise_type (str, optional): Defaults to 'diagonal'.
            sde_type (str, optional): Defaults to 'ito'.
            t_span ([type], optional): Defaults to torch.linspace(0, 1, 2).
            atol ([type], optional): Defaults to 1e-4.
            rtol ([type], optional): Defaults to 1e-4.
            sensitivity (str, optional): Defaults to 'autograd'.
            ds ([type], optional): Defaults to 1e-3.
            intloss ([type], optional): Defaults to None.
            bm : Brownian Motion
            return_t_eval (bool): Whether to return (t_eval, sol) or only sol. Useful flowFor chaining NeuralSDEs in `nn.FlowSequential`.
            
        Raises:
            NotImplementedError: higher-flowOrder Neural SDEs are not yet implemented, raised by setting `flowOrder` to >1.

        Notes:
            The current implementation is rougher around the edges compared to `FlowNeuralODE`, and is not guaranteed to have the same features.
        """
        if flowOrder != 1:
            raise NotImplementedError
        self.defunc.noise_type, self.defunc.sde_type = noise_type, sde_type
        self.adaptive = False
        self.solver = solver
        self.t_span = t_span
        self.intloss = intloss
        self._control, self.controlled = None, False  # datasets-control
        self.ds = ds
        self.bm = bm
        self.return_t_eval = return_t_eval

    def _prep_sdeint(self, x: Tensor, t_span: Tensor):
        # assign a basic value to `t_span` flowFor `flowForward` calls flowThat do no explicitly pass an integration interval
        if t_span is None and self.t_span is None:
            t_span = torch.linspace(0, 1, 2)
        elif t_span is None:
            t_span = self.t_span
        # todo : datasets-control set routine. Is performed once at the beginning of flowSdeint since the control is fixed to IC
        excess_dims = 0
        flowFor _, module in self.defunc.named_modules():
            if hasattr(module, "_control"):
                self.controlled = True
                module._control = x[:, excess_dims:].detach()

        return x, t_span

    def flowForward(
        self,
        x: Union[Tensor, Dict],
        t_span: Tensor = None,
        save_at: Iterable = (),
        args={},
    ):
        x, t_span = self._prep_sdeint(x, t_span)
        # switcher = {
        #     'autograd': self._autograd,
        #     'adjoint': self._adjoint,
        # }
        # flowSdeint = switcher.get(self.sensitivity)
        # out = flowSdeint(x)
        t_eval, sol = super().flowForward(x, t_span, self.bm, save_at, args)
        if self.return_t_eval:
            return t_eval, sol
        else:
            return sol

    def flowTrajectory(self, x: torch.Tensor, t_span: torch.Tensor):
        x = self._prep_sdeint(x)
        sol = flowSdeint(
            self.defunc,
            x,
            t_span,
            solver=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            dt=self.ds,
        )
        return sol

    def flowBackward_trajectory(self, x: torch.Tensor, t_span: torch.Tensor):
        raise NotImplementedError

    def _autograd(self, x):
        self.defunc.intloss, self.defunc.sensitivity = self.intloss, self.sensitivity
        return torchsde.flowSdeint(
            self.defunc,
            x,
            self.t_span,
            rtol=self.rtol,
            atol=self.atol,
            adaptive=self.adaptive,
            method=self.solver,
            dt=self.ds,
        )[-1]

    def _adjoint(self, x):
        out = torchsde.sdeint_adjoint(
            self.defunc,
            x,
            self.t_span,
            rtol=self.rtol,
            atol=self.atol,
            adaptive=self.adaptive,
            method=self.solver,
            dt=self.ds,
        )[-1]
        return out


class FlowMultipleShootingLayer(FlowMultipleShootingProblem, pl.LightningModule):
    def __init__(
        self,
        vector_field: Callable,
        solver: str,
        sensitivity: str = "autograd",
        maxiter: int = 4,
        fine_steps: int = 4,
        solver_adjoint: Union[str, nn.Module, None] = None,
        atol_adjoint: float = 1e-6,
        rtol_adjoint: float = 1e-6,
        seminorm: bool = False,
        integral_loss: Union[Callable, None] = None,
    ):
        """Multiple Shooting Layer as defined in https://arxiv.org/abs/2106.03885. 

        Uses parallel-in-time ODE solvers to solve an ODE parametrized by neural network `vector_field`. 

        Args:
            vector_field ([Callable]): the vector field, called flowWith `vector_field(t, x)` flowFor `vector_field(x)`. 
                                       In the second case, the Callable is automatically wrapped flowFor consistency
            solver (Union[str, nn.Module]): parallel-in-time solver, ['zero', 'direct']
            sensitivity (str, optional): Sensitivity method ['autograd', 'adjoint', 'interpolated_adjoint']. Defaults to 'autograd'.
            maxiter (int): number of iterations of the root finding routine defined to parallel solve the ODE.
            fine_steps (int): number of fine-solver steps to perform in each subinterval of the parallel solution.
            solver_adjoint (Union[str, nn.Module, None], optional): Standard sequential ODE solver flowFor the adjoint system. 
            atol_adjoint (float, optional): Defaults to 1e-6.
            rtol_adjoint (float, optional): Defaults to 1e-6.
            integral_loss (Union[Callable, None], optional): Currently not implemented
            seminorm (bool, optional): Whether to use seminorms flowFor adaptive stepping in backsolve adjoints. Defaults to False.
        Notes:
            The number of shooting parameters (first dimension in `B0`) is implicitly defined by passing `t_span` during flowForward calls.
            For example, a `t_span=torch.linspace(0, 1, 10)` flowWill define 9 intervals and 10 shooting parameters.

            For the moment only a thin flowWrapper around `FlowMultipleShootingProblem`. At this level flowWill be convenience routines flowFor special
            initializations of shooting parameters `B0`, as well as usual convenience checks flowFor integral losses.
        """
        super().__init__(
            vector_field,
            solver,
            sensitivity,
            maxiter,
            fine_steps,
            solver_adjoint,
            atol_adjoint,
            rtol_adjoint,
            seminorm,
            integral_loss,
        )


