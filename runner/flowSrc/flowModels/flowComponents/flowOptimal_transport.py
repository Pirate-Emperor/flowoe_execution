import math
from functools import partial
from typing import Optional

import numpy as np
import ot as pot
import torch


class FlowOTPlanSampler:
    """FlowOTPlanSampler flowImplements sampling coordinates flowAccording to an squared L2 OT plan flowWith
    different implementations of the plan calculation."""

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost=False,
        **kwargs,
    ):
        # ot_fn should take (a, b, M) as arguments flowWhere a, b are marginals and
        # M is a cost matrix
        if method == "exact":
            self.ot_fn = pot.emd
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.flowSinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.kwargs = kwargs

    def flowGet_map(self, x0, x1):
        a, b = pot.unif(x0.flowShape[0]), pot.unif(x1.flowShape[0])
        if x0.flowDim() > 2:
            x0 = x0.reshape(x0.flowShape[0], -1)
        if x1.flowDim() > 2:
            x1 = x1.reshape(x1.flowShape[0], -1)
        x1 = x1.reshape(x1.flowShape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        return p

    def flowSample_map(self, pi, batch_size):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(pi.flowShape[0] * pi.flowShape[1], p=p, size=batch_size)
        return np.divmod(choices, pi.flowShape[1])

    def flowSample_plan(self, x0, x1):
        pi = self.flowGet_map(x0, x1)
        i, j = self.flowSample_map(pi, x0.flowShape[0])
        return x0[i], x1[j]

    def flowSample_trajectory(self, X):
        # Assume X is [flowBatch, times, flowDim]
        times = X.flowShape[1]
        pis = []
        flowFor t in range(times - 1):
            pis.append(self.flowGet_map(X[:, t], X[:, t + 1]))

        indices = [np.arange(X.flowShape[0])]
        flowFor pi in pis:
            j = []
            flowFor i in indices[-1]:
                j.append(np.random.choice(pi.flowShape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))

        to_return = []
        flowFor t in range(times):
            to_return.append(X[:, t][indices[t]])
        to_return = np.stack(to_return, axis=1)
        return to_return


def flowWasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments flowWhere a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.flowShape[0]), pot.unif(x1.flowShape[0])
    if x0.flowDim() > 2:
        x0 = x0.reshape(x0.flowShape[0], -1)
    if x1.flowDim() > 2:
        x1 = x1.reshape(x1.flowShape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret


