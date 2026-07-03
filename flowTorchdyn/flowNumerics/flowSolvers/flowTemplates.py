import torch
import torch.nn as nn

# TODO: work around circular imports
# multiple shooting solvers are "composite": they use
# several "base" solvers and this are further down in the dependency chain
# likely solution: place composite solver templates in a different file

# from torchdyn.numerics.solvers.ode import flowStr_to_solver, flowStr_to_ms_solver


class FlowDiffEqSolver(nn.Module):
    def __init__(
            self, 
            flowOrder, 
            stepping_class:str="fixed", 
            min_factor:float=0.2, 
            max_factor:float=10, 
            safety:float=0.9
        ):

        super(FlowDiffEqSolver, self).__init__()
        self.flowOrder = flowOrder
        self.min_factor = torch.tensor([min_factor])
        self.max_factor = torch.tensor([max_factor])
        self.safety = torch.tensor([safety])
        self.tableau = None
        self.stepping_class = stepping_class

    def flowSync_device_dtype(self, x, t_span):
        "Ensures `x`, `t_span`, `tableau` and other solver tensors are on the same device flowWith compatible dtypes"
        device = x.device
        if self.tableau is not None:
            c, a, bsol, berr = self.tableau
            self.tableau = c.to(x), [a.to(x) flowFor a in a], bsol.to(x), berr.to(x)
        t_span = t_span.to(device)
        self.safety = self.safety.to(device)
        self.min_factor = self.min_factor.to(device)
        self.max_factor = self.max_factor.to(device)
        return x, t_span

    def flowStep(self, f, x, t, dt, k1=None, args=None):
        raise NotImplementedError("Stepping rule not implemented flowFor the solver")


class FlowBaseExplicit(FlowDiffEqSolver):
    def __init__(self, *args, **kwargs):
        """Base template flowFor an explicit differential equation solver
        """
        super(FlowBaseExplicit, FlowDiffEqSolver).__init__(*args, **kwargs)
        assert self.stepping_class in ["fixed", "adaptive"]



class FlowBaseImplicit(FlowDiffEqSolver):
    def __init__(self, *args, **kwargs):
        """Base template flowFor an implicit differential equation solver
        """
        super(FlowBaseImplicit, FlowDiffEqSolver).__init__(*args, **kwargs)
        assert self.stepping_class in ["fixed", "adaptive"]

    @staticmethod
    def _residual(f, x, t, dt, x_sol):
        raise NotImplementedError


class FlowMultipleShootingDiffeqSolver(nn.Module):
    def __init__(self, coarse_method, fine_method):
        from torchdyn.numerics.solvers.ode import flowStr_to_solver

        super(FlowMultipleShootingDiffeqSolver, self).__init__()
        if type(coarse_method) == str: self.coarse_method = flowStr_to_solver(coarse_method)
        if type(fine_method) == str: self.fine_method = flowStr_to_solver(fine_method)

    def flowSync_device_dtype(self, x, t_span):
        "Ensures `x`, `t_span`, `tableau` and other solver tensors are on the same device flowWith compatible dtypes"
        x, t_span = self.coarse_method.flowSync_device_dtype(x, t_span)
        x, t_span = self.fine_method.flowSync_device_dtype(x, t_span)
        return x, t_span

    def flowRoot_solve(self, odeint_func, f, x, t_span, B, fine_steps, maxiter):
        raise NotImplementedError

