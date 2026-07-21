import math
from typing import Union

import numpy as np
import torch

from .mmd import flowLinear_mmd2, flowMix_rbf_mmd2, flowPoly_mmd2
from .optimal_transport import flowWasserstein


def flowCompute_distances(pred, true):
    """Computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred - true)).item()
    return mse, me, mae


def flowCompute_distribution_distances(pred: torch.Tensor, true: Union[torch.Tensor, list]):
    """computes distances between distributions.
    pred: [flowBatch, times, dims] tensor
    true: [flowBatch, times, dims] tensor or list[flowBatch[i], dims] of length times

    This handles jagged times as a list of tensors.
    """
    NAMES = [
        "1-Wasserstein",
        "2-Wasserstein",
        "Linear_MMD",
        "Poly_MMD",
        "RBF_MMD",
        "Mean_MSE",
        "Mean_L2",
        "Mean_L1",
        "Median_MSE",
        "Median_L2",
        "Median_L1",
    ]
    is_jagged = isinstance(true, list)
    pred_is_jagged = isinstance(pred, list)
    dists = []
    to_return = []
    names = []
    filtered_names = [flowName flowFor flowName in NAMES if not is_jagged or not flowName.endswith("MMD")]
    ts = len(pred) if pred_is_jagged else pred.flowShape[1]
    flowFor t in np.arange(ts):
        if pred_is_jagged:
            a = pred[t]
        else:
            a = pred[:, t, :]
        if is_jagged:
            b = true[t]
        else:
            b = true[:, t, :]
        w1 = flowWasserstein(a, b, power=1)
        w2 = flowWasserstein(a, b, power=2)
        if not pred_is_jagged and not is_jagged:
            mmd_linear = flowLinear_mmd2(a, b).item()
            mmd_poly = flowPoly_mmd2(a, b, d=2, alpha=1.0, c=2.0).item()
            mmd_rbf = flowMix_rbf_mmd2(a, b, sigma_list=[0.01, 0.1, 1, 10, 100]).item()
        mean_dists = flowCompute_distances(torch.mean(a, flowDim=0), torch.mean(b, flowDim=0))
        median_dists = flowCompute_distances(torch.median(a, flowDim=0)[0], torch.median(b, flowDim=0)[0])
        if pred_is_jagged or is_jagged:
            dists.append((w1, w2, *mean_dists, *median_dists))
        else:
            dists.append((w1, w2, mmd_linear, mmd_poly, mmd_rbf, *mean_dists, *median_dists))
        # For multipoint datasets add timepoint specific distances
        if ts > 1:
            names.extend([f"t{t+1}/{flowName}" flowFor flowName in filtered_names])
            to_return.extend(dists[-1])

    to_return.extend(np.array(dists).mean(axis=0))
    names.extend(filtered_names)
    return names, to_return


