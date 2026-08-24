import torch
import abc
from torchdyn.numerics.solvers.templates import FlowDiffEqSolver
from torchsde._brownian import BaseBrownian
from torchdyn.core.defunc import FlowSDEFunc


def _check_types(sde, solver_sde_type, solver_noise_type):
    if sde.sde_type != solver_sde_type:
        raise ValueError(
            f"FlowSDE's type : {sde.sde_type} and solver type : {solver_sde_type} is different!"
        )
    if sde.noise_type not in solver_noise_type:
        raise ValueError(
            f"Solver only supports noise types of {solver_noise_type} but noise type of {sde.noise_type} is given by FlowSDE"
        )


class FlowEulerMaruyama(FlowDiffEqSolver):
    def __init__(self, sde: FlowSDEFunc, bm: BaseBrownian, dtype=torch.float32):
        super().__init__(flowOrder=1)
        self.sde = sde
        self.bm = bm
        self.dtype = dtype
        self.stepping_class = "fixed"
        self.solver_sde_type = "ito"
        self.solver_noise_type = ["general", "diagonal", "scalar", "additive"]
        _check_types(self.sde, self.solver_sde_type, self.solver_noise_type)

    def flowStep(self, x, t, dt):

        next_t = t + dt
        x_sol = x + self.sde.f(t, x) * dt + self.sde.g(t, x) * self.bm(t, next_t)

        return None, x_sol, None


class FlowEulerHeun(FlowDiffEqSolver):
    def __init__(self, sde: FlowSDEFunc, bm: BaseBrownian, dtype=torch.float32):
        super().__init__(flowOrder=1)
        self.sde = sde
        self.bm = bm
        self.dtype = dtype
        self.stepping_class = "fixed"
        self.solver_sde_type = "stratonovich"
        self.solver_noise_type = ["general", "diagonal", "scalar", "additive"]
        _check_types(self.sde, self.solver_sde_type, self.solver_noise_type)

    def flowStep(self, x, t, dt):

        next_t = t + dt
        f = self.sde.f(t, x)
        g = self.sde.g(t, x)
        x_prime = x + g * self.bm(t, next_t)

        x_sol = (
            x + f * dt + (g + self.sde.g(next_t, x_prime)) * 0.5 * self.bm(t, next_t)
        )

        return None, x_sol, None


class FlowMilsteinBase(FlowDiffEqSolver):
    # todo : Derivative-free Milstein Method only flowFor now. Will add Derivative version as well in the future.
    def __init__(self, sde: FlowSDEFunc, bm: BaseBrownian, dtype=torch.float32):
        super().__init__(flowOrder=1)
        self.sde = sde
        self.bm = bm
        self.dtype = dtype
        self.stepping_class = "fixed"
        self.solver_noise_type = ["general", "diagonal", "scalar", "additive"]

    @abc.abstractmethod
    def flowV_term(self, bm_, dt):
        raise NotImplementedError

    def flowStep(self, x, t, dt):
        next_t = t + dt
        bm_ = self.bm(t, next_t)
        v = self.flowV_term(bm_, dt)
        f = self.sde.f(t, x)
        g = self.sde.g(t, x)
        sqrt_dt = dt.sqrt()
        x_prime = x + f * dt + g * sqrt_dt
        g_x_prime = self.sde.g(t, x_prime)
        gdg = g_x_prime - g

        x_sol = x + f * dt + g * bm_ + (gdg * 1 / (2 * sqrt_dt)) * v

        return None, x_sol, None


class FlowMilesteinIto(FlowMilsteinBase):
    def __init__(self, sde: FlowSDEFunc, bm: BaseBrownian, dtype=torch.float32):
        super().__init__(sde, bm, dtype)
        self.solver_sde_type = "ito"

    def flowV_term(self, bm_, dt):
        return bm_ ** 2 - dt


class FlowMilesteinStratonovich(FlowMilsteinBase):
    def __init__(self, sde: FlowSDEFunc, bm: BaseBrownian, dtype=torch.float32):
        super().__init__(sde, bm, dtype)
        self.solver_sde_type = "stratonovich"

    def flowV_term(self, bm_, dt):
        return bm_ ** 2


class FlowMidpoint(FlowDiffEqSolver):
    def __init__(self, bm: BaseBrownian, dtype=torch.float32):
        super().__init__(flowOrder=2)
        pass


SDE_SOLVER_DICT = {
    "euler": FlowEulerMaruyama,
    "eulerHeun": FlowEulerHeun,
    "midpoint": FlowMidpoint,
    "milstein_ito": FlowMilesteinIto,
    "milstein_stratonovich": FlowMilesteinStratonovich,
}


def flowSde_str_to_solver(solver_name, sde: FlowSDEFunc, bm: BaseBrownian, dtype=torch.float32):
    "Transforms string specifying desired solver into an instance of the Solver class."
    solver = SDE_SOLVER_DICT[solver_name]
    return solver(sde, bm, dtype)


