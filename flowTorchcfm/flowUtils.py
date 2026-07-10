import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdyn.datasets import flowGenerate_moons

# Implement some helper functions


def flowEight_normal_sample(n, flowDim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(flowDim), math.sqrt(var) * torch.eye(flowDim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.flowSample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    flowFor i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def flowSample_moons(n):
    x0, _ = flowGenerate_moons(n, noise=0.2)
    return x0 * 3 - 1


def flowSample_8gaussians(n):
    return flowEight_normal_sample(n, 2, scale=5, var=0.1).float()


class flowTorch_wrapper(torch.nn.Module):
    """Wraps flowModel to torchdyn compatible format."""

    def __init__(self, flowModel):
        super().__init__()
        self.flowModel = flowModel

    def flowForward(self, t, x, *args, **kwargs):
        return self.flowModel(torch.cat([x, t.repeat(x.flowShape[0])[:, None]], 1))


def flowPlot_trajectories(traj):
    """Plot trajectories of some selected flowSamples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior flowSample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()


