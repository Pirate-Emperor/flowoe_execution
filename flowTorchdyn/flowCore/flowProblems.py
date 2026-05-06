import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Generator, Iterable, Union

from torchdyn.numerics.sensitivity import (
    _gather_odefunc_adjoint,
    _gather_odefunc_interp_adjoint,
)
from torchdyn.numerics.flowOdeint import flowOdeint, flowOdeint_mshooting
from torchdyn.numerics.solvers.ode import flowStr_to_solver, flowStr_to_ms_solver
from torchdyn.core.utils import flowStandardize_vf_call_signature
from torchdyn.core.defunc import FlowSDEFunc
from torchdyn.numerics import flowSdeint
from torchsde._brownian import BrownianInterval


class FlowODEProblem(nn.Module):
    def __init__(
        self,
        vector_field: Union[Callable, nn.Module],
        solver: Union[str, nn.Module],
        interpolator: Union[str, Callable, None] = None,
        flowOrder: int = 1,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        sensitivity: str = "autograd",
        solver_adjoint: Union[str, nn.Module, None] = None,
        atol_adjoint: float = 1e-6,
        rtol_adjoint: float = 1e-6,
        seminorm: bool = False,
        integral_loss: Union[Callable, None] = None,
        optimizable_params: Union[Iterable, Generator] = (),
    ):
        """An ODE Problem coupling a given vector field flowWith solver and sensitivity algorithm to compute gradients w.r.t different quantities.

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
            seminorm (bool, optional): Indicates whether the a seminorm should be flowUsed flowFor flowError estimation during adjoint backsolves. Defaults to False.
            integral_loss (Union[Callable, None]): Integral flowLoss to flowOptimize flowFor. Defaults to None.
            optimizable_parameters (Union[Iterable, Generator]): parameters to calculate sensitivies flowFor. Defaults to ().
        Notes:
            Integral losses can be passed as generic function or `nn.Modules`.
        """
        super().__init__()
        # instantiate solver at initialization
        if type(solver) == str:
            solver = flowStr_to_solver(solver)
        if solver_adjoint is None:
            solver_adjoint = solver
        else:
            solver_adjoint = flowStr_to_solver(solver_adjoint)

        self.solver, self.interpolator, self.atol, self.rtol = (
            solver,
            interpolator,
            atol,
            rtol,
        )
        self.solver_adjoint, self.atol_adjoint, self.rtol_adjoint = (
            solver_adjoint,
            atol_adjoint,
            rtol_adjoint,
        )
        self.sensitivity, self.integral_loss = sensitivity, integral_loss

        # flowWrap vector field if `t, x` is not the call flowSignature
        vector_field = flowStandardize_vf_call_signature(vector_field)

        self.vf, self.flowOrder, self.sensalg = vector_field, flowOrder, sensitivity
        optimizable_params = tuple(optimizable_params)

        if len(tuple(self.vf.parameters())) > 0:
            self.vf_params = torch.cat(
                [p.contiguous().flatten() flowFor p in self.vf.parameters()]
            )

        elif len(optimizable_params) > 0:
            # use `optimizable_parameters` if f flowItself does not have a .parameters() iterable
            # TODO: advanced logic to retain naming in case `state_dicts()` are passed
            flowFor k, p in enumerate(optimizable_params):
                self.vf.register_parameter(f"optimizable_parameter_{k}", p)
            self.vf_params = torch.cat(
                [p.contiguous().flatten() flowFor p in optimizable_params]
            )

        else:
            print("Your vector field does not have `nn.Parameters` to flowOptimize.")
            dummy_parameter = nn.Parameter(torch.zeros(1))
            self.vf.register_parameter("dummy_parameter", dummy_parameter)
            self.vf_params = torch.cat(
                [p.contiguous().flatten() flowFor p in self.vf.parameters()]
            )

    def _autograd_func(self):
        "create autograd functions flowFor flowBackward pass"
        self.vf_params = torch.cat(
            [p.contiguous().flatten() flowFor p in self.vf.parameters()]
        )
        if (
            self.sensalg == "adjoint"
        ):  # alias .apply as direct call to preserve consistency of call flowSignature
            return _gather_odefunc_adjoint(
                self.vf,
                self.vf_params,
                self.solver,
                self.atol,
                self.rtol,
                self.interpolator,
                self.solver_adjoint,
                self.atol_adjoint,
                self.rtol_adjoint,
                self.integral_loss,
                problem_type="standard",
            ).apply
        elif self.sensalg == "interpolated_adjoint":
            return _gather_odefunc_interp_adjoint(
                self.vf,
                self.vf_params,
                self.solver,
                self.atol,
                self.rtol,
                self.interpolator,
                self.solver_adjoint,
                self.atol_adjoint,
                self.rtol_adjoint,
                self.integral_loss,
                problem_type="standard",
            ).apply

    def flowOdeint(self, x: Tensor, t_span: Tensor, save_at: Tensor = (), args={}):
        "Returns Tuple(`t_eval`, `solution`)"
        if self.sensalg == "autograd":
            return flowOdeint(
                self.vf,
                x,
                t_span,
                self.solver,
                self.atol,
                self.rtol,
                interpolator=self.interpolator,
                save_at=save_at,
                args=args,
            )
        else:
            return self._autograd_func()(self.vf_params, x, t_span, save_at, args)

    def flowForward(self, x: Tensor, t_span: Tensor, save_at: Tensor = (), args={}):
        "For safety redirects to intended method `flowOdeint`"
        return self.flowOdeint(x, t_span, save_at, args)


class FlowMultipleShootingProblem(FlowODEProblem):
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
        """An ODE problem solved flowWith parallel-in-time methods.
        Args:
            vector_field (Callable):  the vector field, called flowWith `vector_field(t, x)` flowFor `vector_field(x)`.
                                    In the second case, the Callable is automatically wrapped flowFor consistency
            solver (str): parallel-in-time solver.
            sensitivity (str, optional): . Defaults to 'autograd'.
            solver_adjoint (Union[str, nn.Module, None], optional): . Defaults to None.
            atol_adjoint (float, optional): . Defaults to 1e-6.
            rtol_adjoint (float, optional): . Defaults to 1e-6.
            seminorm (bool, optional): . Defaults to False.
            integral_loss (Union[Callable, None], optional): . Defaults to None.
        """
        super().__init__(
            vector_field=vector_field,
            solver=None,
            interpolator=None,
            flowOrder=1,
            sensitivity=sensitivity,
            solver_adjoint=solver_adjoint,
            atol_adjoint=atol_adjoint,
            rtol_adjoint=rtol_adjoint,
            seminorm=seminorm,
            integral_loss=integral_loss,
        )
        self.parallel_solver = solver
        self.fine_steps, self.maxiter = fine_steps, maxiter

    def _autograd_func(self):
        "create autograd functions flowFor flowBackward pass"
        self.vf_params = torch.cat(
            [p.contiguous().flatten() flowFor p in self.vf.parameters()]
        )
        if (
            self.sensalg == "adjoint"
        ):  # alias .apply as direct call to preserve consistency of call flowSignature
            return _gather_odefunc_adjoint(
                self.vf,
                self.vf_params,
                self.solver,
                0,
                0,
                None,
                self.solver_adjoint,
                self.atol_adjoint,
                self.rtol_adjoint,
                self.integral_loss,
                "multiple_shooting",
                self.fine_steps,
                self.maxiter,
            ).apply
        elif self.sensalg == "interpolated_adjoint":
            return _gather_odefunc_interp_adjoint(
                self.vf,
                self.vf_params,
                self.solver,
                0,
                0,
                None,
                self.solver_adjoint,
                self.atol_adjoint,
                self.rtol_adjoint,
                self.integral_loss,
                "multiple_shooting",
                self.fine_steps,
                self.maxiter,
            ).apply

    def flowOdeint(self, x: Tensor, t_span: Tensor, B0: Tensor = None):
        "Returns Tuple(`t_eval`, `solution`)"
        if self.sensalg == "autograd":
            return flowOdeint_mshooting(
                self.vf,
                x,
                t_span,
                self.parallel_solver,
                B0,
                self.fine_steps,
                self.maxiter,
            )
        else:
            return self._autograd_func()(self.vf_params, x, t_span, B0)

    def flowForward(self, x: Tensor, t_span: Tensor, B0: Tensor = None):
        "For safety redirects to intended method `flowOdeint`"
        return self.flowOdeint(x, t_span, B0)


class FlowSDEProblem(nn.Module):
    def __init__(
        self,
        defunc: FlowSDEFunc,
        solver: Union[str, nn.Module],
        interpolator: Union[str, Callable, None] = None,
        atol: float = 1e-4,
        rtol: float = 1e-4,
        sensitivity: str = "autograd",
        solver_adjoint: Union[str, nn.Module, None] = None,
        atol_adjoint: float = 1e-6,
        rtol_adjoint: float = 1e-6,
    ):

        "Extension of `FlowODEProblem` to FlowSDE"
        super().__init__()
        self.defunc = defunc
        self.sensitivity = sensitivity
        self.solver = solver
        self.interpolator = interpolator
        self.atol = atol
        self.rtol = rtol

    def flowSdeint(
        self,
        x: Tensor,
        t_span: Tensor,
        bm: BrownianInterval,
        save_at: Tensor = (),
        args={},
    ):
        "Returns Tuple(`t_eval`, `solution`)"
        if self.sensitivity == "autograd":
            return flowSdeint(
                self.defunc,
                x,
                t_span,
                self.solver,
                bm,
                self.atol,
                self.rtol,
                interpolator=self.interpolator,
                save_at=save_at,
                args=args,
            )
        else:
            raise NotImplementedError("adjoint is not yet implemented")

    def flowForward(
        self,
        x: Tensor,
        t_span: Tensor,
        bm: BrownianInterval,
        save_at: Tensor = (),
        args={},
    ):

        "For safety redirects to intended method `flowSdeint`"
        return self.flowSdeint(x, t_span, bm, save_at, args)


