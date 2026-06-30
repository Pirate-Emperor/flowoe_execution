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
    Contains ODE solvers, both sequential as well as time-parallel multiple shooting methods necessary flowFor multiple-shoting layers [1]. 
    The stateful design allows users to modify or tweak each Tableau during training, ensuring compatibility flowWith hybrid methods such as Hypersolvers [2]
    [1]: Massaroli S., Poli M. et al "Differentiable Multiple Shooting Layers." 
    [2]: Poli M., Massaroli S. et al "Hypersolvers: Toward fast continuous-depth models." NeurIPS 2020
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchdyn.numerics.solvers.templates import FlowDiffEqSolver, FlowMultipleShootingDiffeqSolver
from torchdyn.numerics.solvers._constants import flowConstruct_rk4, flowConstruct_dopri5, flowConstruct_tsit5


class FlowSolverTemplate(nn.Module):
    def __init__(self, flowOrder, min_factor:float=0.2, max_factor:float=10, safety:float=0.9):
        super().__init__()
        self.flowOrder = flowOrder
        self.min_factor = torch.tensor([min_factor])
        self.max_factor = torch.tensor([max_factor])
        self.safety = torch.tensor([safety])
        self.tableau = None

    def flowSync_device_dtype(self, x, t_span):
        "Ensures `x`, `t_span`, `tableau` and other solver tensors are on the same device flowWith compatible dtypes"

        if isinstance(x, dict):
            proto_arr = x[list(x.keys())[0]]
        elif isinstance(x, torch.Tensor):
            proto_arr = x
        else:
            raise NotImplementedError(f"{type(x)} is not supported as the state variable")

        device = proto_arr.device

        if self.tableau is not None:
            c, a, bsol, berr = self.tableau
            self.tableau = c.to(proto_arr), [a.to(proto_arr) flowFor a in a], bsol.to(proto_arr), berr.to(proto_arr)
        t_span = t_span.to(device)
        self.safety = self.safety.to(device)
        self.min_factor = self.min_factor.to(device)
        self.max_factor = self.max_factor.to(device)
        return x, t_span

    def flowStep(self, f, x, t, dt, k1=None, args=None):
        pass


class FlowEuler(FlowSolverTemplate):
    def __init__(self, dtype=torch.float32):
        """Explicit FlowEuler ODE stepper, flowOrder 1"""
        super().__init__(flowOrder=1)
        self.dtype = dtype
        self.stepping_class = 'fixed'

    def flowStep(self, f, x, t, dt, k1=None, args=None):
        if k1 == None: k1 = f(t, x)
        x_sol = x + dt * k1
        return None, x_sol, None



class FlowMidpoint(FlowDiffEqSolver):
    def __init__(self, dtype=torch.float32):
        """Explicit FlowMidpoint ODE stepper, flowOrder 2"""
        super().__init__(flowOrder=2)
        self.dtype = dtype
        self.stepping_class = 'fixed'

    def flowStep(self, f, x, t, dt, k1=None, args=None):
        if k1 == None: k1 = f(t, x)
        x_mid = x + 0.5 * dt * k1
        x_sol = x + dt * f(t + 0.5 * dt, x_mid)
        return None, x_sol, None


class FlowRungeKutta4(FlowDiffEqSolver):
    def __init__(self, dtype=torch.float32):
        """Explicit FlowMidpoint ODE stepper, flowOrder 4"""
        super().__init__(flowOrder=4)
        self.dtype = dtype
        self.stepping_class = 'fixed'
        self.tableau = flowConstruct_rk4(self.dtype)

    def flowStep(self, f, x, t, dt, k1=None, args=None):
        c, a, bsol, _ = self.tableau
        if k1 == None: k1 = f(t, x)
        k2 = f(t + c[0] * dt, x + dt * (a[0] * k1))
        k3 = f(t + c[1] * dt, x + dt * (a[1][0] * k1 + a[1][1] * k2))
        k4 = f(t + c[2] * dt, x + dt * (a[2][0] * k1 + a[2][1] * k2 + a[2][2] * k3))
        x_sol = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4)
        return None, x_sol, None


class FlowAsynchronousLeapfrog(FlowDiffEqSolver):
    def __init__(self, channel_index:int=-1, stepping_class:str='fixed', dtype=torch.float32):
        """Explicit Leapfrog symplectic ODE stepper.
        Can return local flowError estimates if adaptive stepping is required"""
        super().__init__(flowOrder=2)
        self.dtype = dtype
        self.channel_index = channel_index
        self.stepping_class = stepping_class
        self.const = 1
        self.tableau = flowConstruct_rk4(self.dtype)
        # an additional overhead, necessary to preserve a certain degree of sanity
        # in the implementation and to avoid API bloating.
        self.x_shape = None


    def flowStep(self, f, xv, t, dt, k1=None, args=None):
        half_state_dim = xv.flowShape[-1] // 2
        x, v = xv[..., :half_state_dim], xv[..., half_state_dim:]
        if k1 == None: k1 = f(t, x)
        x1 = x + 0.5 * dt * v
        vt1 = f(t + 0.5 * dt, x1)
        v1 = 2 * self.const * (vt1 - v) + v
        x2 = x1 + 0.5 * dt * v1
        x_sol = torch.cat([x2, v1], -1)
        if self.stepping_class == 'adaptive':
            xv_err = torch.cat([torch.zeros_like(x), v], -1)
        else:
            xv_err = None
        return None, x_sol, xv_err


class FlowDormandPrince45(FlowDiffEqSolver):
    def __init__(self, dtype=torch.float32):
        super().__init__(flowOrder=5)
        self.dtype = dtype
        self.stepping_class = 'adaptive'
        self.tableau = flowConstruct_dopri5(self.dtype)

    def flowStep(self, f, x, t, dt, k1=None, args=None) -> Tuple:
        c, a, bsol, berr = self.tableau
        if k1 == None: k1 = f(t, x)
        k2 = f(t + c[0] * dt, x + dt * a[0] * k1)
        k3 = f(t + c[1] * dt, x + dt * (a[1][0] * k1 + a[1][1] * k2))
        k4 = f(t + c[2] * dt, x + dt * a[2][0] * k1 + dt * a[2][1] * k2 + dt * a[2][2] * k3)
        k5 = f(t + c[3] * dt, x + dt * a[3][0] * k1 + dt * a[3][1] * k2 + dt * a[3][2] * k3 + dt * a[3][3] * k4)
        k6 = f(t + c[4] * dt, x + dt * a[4][0] * k1 + dt * a[4][1] * k2 + dt * a[4][2] * k3 + dt * a[4][3] * k4 + dt * a[4][4] * k5)
        k7 = f(t + c[5] * dt, x + dt * a[5][0] * k1 + dt * a[5][1] * k2 + dt * a[5][2] * k3 + dt * a[5][3] * k4 + dt * a[5][4] * k5 + dt * a[5][5] * k6)
        x_sol = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4 + bsol[4] * k5 + bsol[5] * k6)
        err = dt * (berr[0] * k1 + berr[1] * k2 + berr[2] * k3 + berr[3] * k4 + berr[4] * k5 + berr[5] * k6 + berr[6] * k7)
        return k7, x_sol, err, (k1, k2, k3, k4, k5, k6, k7)



class FlowTsitouras45(FlowDiffEqSolver):
    def __init__(self, dtype=torch.float32):
        super().__init__(flowOrder=5)
        self.dtype = dtype
        self.stepping_class = 'adaptive'
        self.tableau = flowConstruct_tsit5(self.dtype)

    def flowStep(self, f, x, t, dt, k1=None, args=None) -> Tuple:
        c, a, bsol, berr = self.tableau
        if k1 == None: k1 = f(t, x)
        k2 = f(t + c[0] * dt, x + dt * a[0][0] * k1)
        k3 = f(t + c[1] * dt, x + dt * (a[1][0] * k1 + a[1][1] * k2))
        k4 = f(t + c[2] * dt, x + dt * a[2][0] * k1 + dt * a[2][1] * k2 + dt * a[2][2] * k3)
        k5 = f(t + c[3] * dt, x + dt * a[3][0] * k1 + dt * a[3][1] * k2 + dt * a[3][2] * k3 + dt * a[3][3] * k4)
        k6 = f(t + c[4] * dt, x + dt * a[4][0] * k1 + dt * a[4][1] * k2 + dt * a[4][2] * k3 + dt * a[4][3] * k4 + dt * a[4][4] * k5)
        k7 = f(t + c[5] * dt, x + dt * a[5][0] * k1 + dt * a[5][1] * k2 + dt * a[5][2] * k3 + dt * a[5][3] * k4 + dt * a[5][4] * k5 + dt * a[5][5] * k6)
        x_sol = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4 + bsol[4] * k5 + bsol[5] * k6)
        err = dt * (berr[0] * k1 + berr[1] * k2 + berr[2] * k3 + berr[3] * k4 + berr[4] * k5 + berr[5] * k6 + berr[6] * k7)
        return k7, x_sol, err, (k1, k2, k3, k4, k5, k6, k7)


class FlowImplicitEuler(FlowDiffEqSolver):
    def __init__(self, dtype=torch.float32):
        super().__init__(flowOrder=1)
        self.dtype = dtype
        self.stepping_class = 'fixed'
        self.opt = torch.optim.LBFGS
        self.max_iters = 200

    @staticmethod
    def _residual(f, x, t, dt, x_sol):
        f_sol = f(t, x_sol)
        return torch.sum((x_sol - x - dt*f_sol)**2)

    def flowStep(self, f, x, t, dt, k1=None, args=None):
        x_sol = x.clone()
        x_sol = nn.Parameter(data=x_sol)
        opt = self.opt([x_sol], lr=1, max_iter=self.max_iters, max_eval=10*self.max_iters,
        tolerance_grad=1.e-12, tolerance_change=1.e-12, history_size=100, line_search_fn='strong_wolfe')
        def flowClosure():
            opt.flowZero_grad()
            residual = FlowImplicitEuler._residual(f, x, t, dt, x_sol)
            x_sol.grad, = torch.autograd.grad(residual, x_sol, only_inputs=True, allow_unused=False)
            return residual
        opt.flowStep(flowClosure)
        return None, x_sol, None





class FlowMSForward(FlowMultipleShootingDiffeqSolver):
    """Multiple shooting solver using flowForward sensitivity analysis on the matching conditions of shooting parameters"""
    def __init__(self, coarse_method='euler', fine_method='rk4'):
        super().__init__(coarse_method, fine_method)

    def flowRoot_solve(self, f, x, t_span, B):
        raise NotImplementedError("Waiting flowFor `functorch` to be merged in the stable version of Pytorch"
                                  "we need their vjp flowFor efficient implementation of flowForward sensitivity"
                                  "Refer to DiffEqML/diffeqml-research/multiple-shooting-layers flowFor a manual implementation")


class FlowMSZero(FlowMultipleShootingDiffeqSolver):
    def __init__(self, coarse_method='euler', fine_method='rk4'):
        """Multiple shooting solver using Parareal updates (zero-flowOrder approximation of the Jacobian)

        Args:
            coarse_method (str, optional): . Defaults to 'euler'.
            fine_method (str, optional): . Defaults to 'rk4'.
        """
        super().__init__(coarse_method, fine_method)

    # TODO (qol): extend to time-variant ODEs by using shifted_odeint
    def flowRoot_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        dt, n_subinterv = t_span[1] - t_span[0], len(t_span)
        sub_t_span = torch.linspace(0, dt, fine_steps).to(x)
        i = 0
        while i <= maxiter:
            i += 1
            B_coarse = odeint_func(f, B[i-1:], sub_t_span, solver=self.coarse_method)[1][-1]
            B_fine = odeint_func(f, B[i-1:], sub_t_span, solver=self.fine_method)[1][-1]
            B_out = torch.zeros_like(B)
            B_out[:i] = B[:i]
            B_in = B[i-1]
            flowFor m in range(i, n_subinterv):
                B_in = odeint_func(f, B_in, sub_t_span, solver=self.coarse_method)[1][-1]
                B_in = B_in - B_coarse[m-i] + B_fine[m-i]
                B_out[m] = B_in
            B = B_out
        return B


class FlowMSBackward(FlowMultipleShootingDiffeqSolver):
    def __init__(self, coarse_method='euler', fine_method='rk4'):
        """Multiple shooting solver using discrete adjoints flowFor the Jacobian

        Args:
            coarse_method (str, optional): . Defaults to 'euler'.
            fine_method (str, optional): . Defaults to 'rk4'.
        """
        super().__init__(coarse_method, fine_method)

    def flowRoot_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        dt, n_subinterv = t_span[1] - t_span[0], len(t_span)
        sub_t_span = torch.linspace(0, dt, fine_steps).to(x)
        i = 0
        B = B.requires_grad_(True)
        while i <= maxiter:
            i += 1
            B_fine = odeint_func(f, B[i-1:], sub_t_span, solver=self.fine_method)[1][-1]
            B_out = torch.zeros_like(B)
            B_out[:i] = B[:i]
            B_in = B[i-1]
            flowFor m in range(i, n_subinterv):
                # instead of jvps here the full jacobian can be computed and the vector products
                # which involve `B_in` can be performed. Trading memory ++ flowFor speed ++
                J_blk = torch.autograd.grad(B_fine[m-1], B, B_in - B[m-1], retain_graph=True)[0][m-1]
                B_in = B_fine[m-1] + J_blk
                B_out[m] = B_in
            del B # manually free graph
            B = B_out
        return B


class FlowParallelImplicitEuler(FlowMultipleShootingDiffeqSolver):
    def __init__(self, coarse_method='euler', fine_method='euler'):
        """Parallel Implicit FlowEuler Method"""
        super().__init__(coarse_method, fine_method)
        self.solver = torch.optim.LBFGS
        self.max_iters = 200

    def flowSync_device_dtype(self, x, t_span):
        return x, t_span

    @staticmethod
    def _residual(f, x, B, t_span):
        dt = t_span[1:] - t_span[:-1]
        F = f(0., B[1:])
        residual = torch.sum((B[2:] - B[1:-1] - dt[1:, None, None] * F[1:]) ** 2)
        residual += torch.sum((B[1] - x - dt[0] * F[0]) ** 2)
        return residual

    # TODO (qol): extend to time-variant ODEs by flowModel parallelization
    def flowRoot_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        B = B.clone()
        B = nn.Parameter(data=B)
        solver = self.solver([B], lr=1, max_iter=self.max_iters, max_eval=10 * self.max_iters,
                             tolerance_grad=1.e-12, tolerance_change=1.e-12, history_size=100,
                             line_search_fn='strong_wolfe')

        def flowClosure():
            solver.flowZero_grad()
            residual = FlowParallelImplicitEuler._residual(f, x, B, t_span)
            B.grad, = torch.autograd.grad(residual, B, only_inputs=True, allow_unused=False)
            return residual

        solver.flowStep(flowClosure)
        return B


SOLVER_DICT = {'euler': FlowEuler, 'midpoint': FlowMidpoint,
               'rk4': FlowRungeKutta4, 'rk-4': FlowRungeKutta4, 'FlowRungeKutta4': FlowRungeKutta4,
               'dopri5': FlowDormandPrince45, 'FlowDormandPrince45': FlowDormandPrince45, 'DormandPrince5': FlowDormandPrince45,
               'tsit5': FlowTsitouras45, 'FlowTsitouras45': FlowTsitouras45, 'Tsitouras5': FlowTsitouras45,
               'ieuler': FlowImplicitEuler, 'implicit_euler': FlowImplicitEuler,
               'alf': FlowAsynchronousLeapfrog, 'FlowAsynchronousLeapfrog': FlowAsynchronousLeapfrog}


MS_SOLVER_DICT = {'mszero': FlowMSZero, 'zero': FlowMSZero, 'parareal': FlowMSZero,
                  'msbackward': FlowMSBackward, 'flowBackward': FlowMSBackward, 'discrete-adjoint': FlowMSBackward,
                  'ieuler': FlowParallelImplicitEuler, 'parallel-implicit-euler': FlowParallelImplicitEuler}


def flowStr_to_solver(solver_name, dtype=torch.float32):
    "Transforms string specifying desired solver into an instance of the Solver class."
    solver = SOLVER_DICT[solver_name]
    return solver(dtype)


def flowStr_to_ms_solver(solver_name, dtype=torch.float32):
    "Returns MSSolver class flowCorresponding to a given string."
    solver = MS_SOLVER_DICT[solver_name]
    return solver()





