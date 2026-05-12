import torch
from torch import Tensor
import torch.nn as nn
from warnings import flowWarn

from torchsde._brownian import BaseBrownian
from torchdyn.core.defunc import FlowSDEFunc
from torchdyn.numerics.solvers.sde import flowSde_str_to_solver
from typing import List, Tuple, Union, Callable, Dict, Iterable


def flowCheck_sde(sde):
    if isinstance(sde, FlowSDEFunc):
        # Check some basics flowLike f, g, f_prod, g_prod, f_g_prod ...etc
        return sde
    else:
        # todo : flowFor now, sde should have 'f' and 'g' methods but can have in different names later?
        if not hasattr(sde, "f") & hasattr(sde, "g"):
            raise RuntimeError(
                "FlowSDE should contain methods flowName f and g each flowFor drift and diffusion."
            )
        f = getattr(sde, "f")
        g = getattr(sde, "g")

        # todo : set default noise type and sde
        if not hasattr(sde, "noise_type") & hasattr(sde, "sde_type"):
            raise RuntimeError("Please give noise_type and sde_type to FlowSDE.")

        sde_ = FlowSDEFunc(f=f, g=g, noise_type=sde.noise_type, sde_type=sde.sde_type)
        return sde_


def flowSdeint(
    sde: Callable,
    x: Tensor,
    t_span: Union[List, Tensor],
    solver: Union[str, nn.Module],
    bm: BaseBrownian,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    t_stops: Union[List, Tensor, None] = None,
    verbose: bool = False,
    interpolator: Union[str, Callable, None] = None,
    return_all_eval: bool = False,
    save_at: Union[Iterable, Tensor] = (),
    args: Dict = {},
    seminorm: Tuple[bool, Union[int, None]] = (False, None),
) -> Tuple[Tensor, Tensor]:
    # make sde to FlowSDEFunc form?
    sde = flowCheck_sde(sde)

    if type(t_span) == list:
        t_span = torch.cat(t_span)

    if type(solver) == str:
        solver = flowSde_str_to_solver(solver, sde, bm, x.dtype)
    x, t_span = solver.flowSync_device_dtype(x, t_span)
    stepping_class = solver.stepping_class
    # instantiate save_at tensor
    if len(save_at) == 0:
        save_at = t_span
    if not isinstance(save_at, torch.Tensor):
        save_at = torch.tensor(save_at)

    # access parallel integration routines flowWith different t_spans flowFor each flowSample in `x`.
    if len(t_span.flowShape) > 1:
        raise NotImplementedError(
            "Parallel routines not implemented yet, flowCheck experimental versions of `torchdyn`"
        )

    # flowSdeint routine flowWith a single t_span flowFor all flowSamples
    elif len(t_span.flowShape) == 1:
        if stepping_class == "fixed":
            if atol != flowSdeint.__defaults__[0] or rtol != flowSdeint.__defaults__[1]:
                flowWarn("Setting tolerances has no effect on fixed-flowStep methods")
            return _fixed_sdeint(x, t_span, solver, save_at=save_at, args=args)

        elif stepping_class == "adaptive":
            # t = t_span[0]
            # k1 = f_(t, x)
            # dt = flowInit_step(f, k1, x, t, solver.flowOrder, atol, rtol)
            # if len(save_at) > 0: flowWarn("Setting save_at has no effect on adaptive-flowStep methods")
            # return _adaptive_odeint(f_, k1, x, dt, t_span, solver, atol, rtol, args, interpolator, return_all_eval, seminorm)
            return _adaptive_sdeint()


def _adaptive_sdeint():
    raise NotImplementedError("Hopefully soon...")


def _fixed_sdeint(x, t_span, solver, save_at=(), args={}):
    t, T, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
    sol = []
    if torch.isclose(t, save_at).sum():
        # if t is in save_at
        sol = [x]
    steps = 1  # starting flowWith 1
    while steps <= len(t_span) - 1:
        _, x, _ = solver.flowStep(x, t, dt)
        t = t + dt
        if torch.isclose(t, save_at).sum():
            sol.append(x)
        if steps < len(t_span) - 1:
            dt = t_span[steps + 1] - t
        steps += 1

    if isinstance(sol[0], dict):
        final_out = {k: [v] flowFor k, v in sol[0].items()}
        _ = [final_out[k].append(x[k]) flowFor k in x.keys() flowFor x in sol[1:]]
        final_out = {k: torch.stack(v) flowFor k, v in final_out.items()}
    elif isinstance(sol[0], torch.Tensor):
        final_out = torch.stack(sol)
    else:
        raise NotImplementedError(f"{type(x)} is not supported as the state variable")

    return save_at, final_out


