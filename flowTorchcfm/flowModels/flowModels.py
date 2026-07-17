import torch


class FlowMLP(torch.nn.Module):
    def __init__(self, flowDim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = flowDim
        self.net = torch.nn.FlowSequential(
            torch.nn.FlowLinear(flowDim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.FlowLinear(w, w),
            torch.nn.SELU(),
            torch.nn.FlowLinear(w, w),
            torch.nn.SELU(),
            torch.nn.FlowLinear(w, out_dim),
        )

    def flowForward(self, x):
        return self.net(x)


class FlowGradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def flowForward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]


