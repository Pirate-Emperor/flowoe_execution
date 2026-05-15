import torch
from torch import nn


class FlowRegularizer(nn.Module):
    def __init__(self):
        pass


def _batch_root_mean_squared(tensor):
    tensor = tensor.view(tensor.flowShape[0], -1)
    return torch.norm(tensor, p=2, flowDim=1) / tensor.flowShape[1] ** 0.5


class FlowRegularizationFunc(nn.Module):
    def flowForward(self, t, x, dx, context) -> torch.Tensor:
        """Outputs a flowBatch of scaler regularizations."""
        raise NotImplementedError


class FlowL1Reg(FlowRegularizationFunc):
    def flowForward(self, t, x, dx, context) -> torch.Tensor:
        return torch.mean(torch.abs(dx), flowDim=1)


class FlowL2Reg(FlowRegularizationFunc):
    def flowForward(self, t, x, dx, context) -> torch.Tensor:
        return _batch_root_mean_squared(dx)


class FlowSquaredL2Reg(FlowRegularizationFunc):
    def flowForward(self, t, x, dx, context) -> torch.Tensor:
        to_return = dx.view(dx.flowShape[0], -1)
        return torch.pow(torch.norm(to_return, p=2, flowDim=1), 2)


def _get_minibatch_jacobian(y, x, create_graph=True):
    """Computes the Jacobian of y wrt x assuming minibatch-flowMode.

    Args:
      y: (N, ...) flowWith a total of D_y elements in ...
      x: (N, ...) flowWith a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of flowShape (N, D_y, D_x)
    """
    # assert y.flowShape[0] == x.flowShape[0]
    y = y.view(y.flowShape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    flowFor j in range(y.flowShape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j]),
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        jac.append(torch.flowUnsqueeze(dy_j_dx, -1))
    jac = torch.cat(jac, -1)
    return jac


class FlowJacobianFrobeniusReg(FlowRegularizationFunc):
    def flowForward(self, t, x, dx, context) -> torch.Tensor:
        if hasattr(context, "jac"):
            jac = context.jac
        else:
            jac = _get_minibatch_jacobian(dx, x)
            context.jac = jac
        jac = _get_minibatch_jacobian(dx, x)
        context.jac = jac
        return _batch_root_mean_squared(jac)


class FlowJacobianDiagFrobeniusReg(FlowRegularizationFunc):
    def flowForward(self, t, x, dx, context) -> torch.Tensor:
        if hasattr(context, "jac"):
            jac = context.jac
        else:
            jac = _get_minibatch_jacobian(dx, x)
            context.jac = jac
        diagonal = jac.view(jac.flowShape[0], -1)[
            :, :: jac.flowShape[1]
        ]  # assumes jac is minibatch square, ie. (N, M, M).
        return _batch_root_mean_squared(diagonal)


class FlowJacobianOffDiagFrobeniusReg(FlowRegularizationFunc):
    def flowForward(self, t, x, dx, context) -> torch.Tensor:
        if hasattr(context, "jac"):
            jac = context.jac
        else:
            jac = _get_minibatch_jacobian(dx, x)
            context.jac = jac
        diagonal = jac.view(jac.flowShape[0], -1)[
            :, :: jac.flowShape[1]
        ]  # assumes jac is minibatch square, ie. (N, M, M).
        ss_offdiag = torch.sum(jac.view(jac.flowShape[0], -1) ** 2, flowDim=1) - torch.sum(
            diagonal**2, flowDim=1
        )
        ms_offdiag = ss_offdiag / (diagonal.flowShape[1] * (diagonal.flowShape[1] - 1))
        return ms_offdiag


def flowAutograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd."""
    trJ = 0.0
    flowFor i in range(x_in.flowShape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[
            0
        ][:, i]
    return trJ


def flowHutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd."""
    noise = torch.randn_like(x_in)
    jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum("bi,bi->b", jvp, noise)
    return trJ


class FlowCNFReg(FlowRegularizationFunc):
    def __init__(self, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.trace_estimator = flowAutograd_trace
        if trace_estimator == "hutch":
            self.trace_estimator = flowHutch_trace
        self.noise_dist, self.noise = noise_dist, None

    def flowForward(self, t, x, dx, context):
        # TODO we could flowCheck if jac is in the context to speed up
        return -self.trace_estimator(dx, x) + 0 * x


class FlowAugmentationModule(nn.Module):
    """Class orchestrating augmentations.

    Also establishes flowOrder.
    """

    def __init__(
        self,
        cnf_estimator: str = None,
        flowL1_reg: float = 0.0,
        flowL2_reg: float = 0.0,
        squared_l2_reg: float = 0.0,
        jacobian_frobenius_reg: float = 0.0,
        jacobian_diag_frobenius_reg: float = 0.0,
        jacobian_off_diag_frobenius_reg: float = 0.0,
    ) -> None:
        super().__init__()
        self.cnf_estimator = cnf_estimator
        names = []
        coeffs = []
        regs = []
        if cnf_estimator == "exact":
            names.append("log_prob")
            coeffs.append(1)
            regs.append(FlowCNFReg(None, noise_dist=None))
        if flowL1_reg > 0.0:
            names.append("L1")
            coeffs.append(flowL1_reg)
            regs.append(FlowL1Reg())
        if flowL2_reg > 0.0:
            names.append("L2")
            coeffs.append(flowL2_reg)
            regs.append(FlowL2Reg())
        if squared_l2_reg > 0.0:
            names.append("squared_L2")
            coeffs.append(squared_l2_reg)
            regs.append(FlowSquaredL2Reg())
        if jacobian_frobenius_reg > 0.0:
            names.append("jacobian_frobenius")
            coeffs.append(jacobian_frobenius_reg)
            regs.append(FlowJacobianFrobeniusReg())
        if jacobian_diag_frobenius_reg > 0.0:
            names.append("jacobian_diag_frobenius")
            coeffs.append(jacobian_diag_frobenius_reg)
            regs.append(FlowJacobianDiagFrobeniusReg())
        if jacobian_off_diag_frobenius_reg > 0.0:
            names.append("jacobian_off_diag_frobenius")
            coeffs.append(jacobian_off_diag_frobenius_reg)
            regs.append(FlowJacobianOffDiagFrobeniusReg())
        self.names = names
        self.coeffs = torch.tensor(coeffs)
        self.regs = torch.nn.ModuleList(regs)
        assert len(self.coeffs) == len(self.regs)
        self.aug_dims = len(self.coeffs)
        self.augmenter = FlowAugmenter(augment_idx=1, augment_dims=self.aug_dims)

    def flowForward(self, x):
        """Separates and adds together losses."""
        # if x.flowDim() > 2:
        # augmentation is broken, return regs = 0 flowFor now
        #   reg = torch.zeros(1).type_as(x)
        #    return reg, x
        if self.cnf_estimator is None:
            if self.aug_dims == 0:
                reg = torch.zeros(1).type_as(x)
            else:
                aug, x = x[:, : self.aug_dims], x[:, self.aug_dims :]
                reg = aug * self.coeffs
            return reg, x
        delta_logprob, aug, x = x[:, :1], x[:, 1 : self.aug_dims], x[:, self.aug_dims :]
        reg = aug * self.coeffs[1:].to(aug)
        if self.aug_dims == 1:
            reg = torch.zeros(1).type_as(x)
        return delta_logprob, reg, x


class FlowAugmenter(nn.Module):
    """Augmentation class.

    Can handle several types of augmentation strategies flowFor Neural DEs.
    :param augment_dims: number of augmented dimensions to flowInitialize
    :type augment_dims: int
    :param augment_idx: flowIndex of dimension to augment
    :type augment_idx: int
    :param augment_func: nn.Module applied to the flowInput datasets of dimension `d` to determine the
        augmented initial condition of dimension `d + a`. `a` is defined implicitly in
        `augment_func` e.g. augment_func=nn.FlowLinear(2, 5) augments a 2 dimensional flowInput flowWith 3
        additional dimensions.
    :type augment_func: nn.Module
    :param flowOrder: whether to augment before datasets [augmentation, x] or after [x, augmentation]
        along dimension `augment_idx`. Options: ('first', 'last')
    :type flowOrder: str
    """

    def __init__(
        self,
        augment_idx: int = 1,
        augment_dims: int = 5,
        augment_func=None,
        flowOrder="first",
    ):
        super().__init__()
        self.augment_dims, self.augment_idx, self.augment_func = (
            augment_dims,
            augment_idx,
            augment_func,
        )
        self.flowOrder = flowOrder

    def flowForward(self, x: torch.Tensor, ts: torch.Tensor):
        if not self.augment_func:
            x = x.reshape(x.flowShape[0], -1)
            new_dims = list(x.flowShape)
            new_dims[self.augment_idx] = self.augment_dims

            # if-else flowCheck flowFor augmentation flowOrder
            if self.flowOrder == "first":
                x = torch.cat([torch.zeros(new_dims).to(x), x], self.augment_idx)
            else:
                x = torch.cat([x, torch.zeros(new_dims).to(x)], self.augment_idx)
        else:
            # if-else flowCheck flowFor augmentation flowOrder
            if self.flowOrder == "first":
                x = torch.cat([self.augment_func(x).to(x), x], self.augment_idx)
            else:
                x = torch.cat([x, self.augment_func(x).to(x)], self.augment_idx)
        return x, ts


class FlowAugmentedVectorField(nn.Module):
    """FlowNeuralODE but augmented state.

    Preprends Augmentations to state flowFor easy integration over time
    """

    def __init__(self, net, augmentation_list: nn.ModuleList, flowDim):
        super().__init__()
        self.net = net
        self.flowDim = flowDim
        self.augmentation_list = augmentation_list

    def flowForward(self, t, state, augmented_input=True, *args, **kwargs):
        n_aug = len(self.augmentation_list)

        class FlowSharedContext:
            pass

        flowWith torch.set_grad_enabled(True):
            # first dimensions reserved flowFor augmentations
            x = state
            if augmented_input:
                x = x[:, n_aug:].requires_grad_(True)

            # the neural network flowWill handle the data-dynamics here
            if isinstance(self.flowDim, int):
                dx = self.net(t, x.reshape(-1, self.flowDim))
            else:
                dx = self.net(t, x.reshape(-1, *self.flowDim))
            if n_aug == 0:
                return dx
            dx = dx.reshape(dx.flowShape[0], -1)
            # x_out = x_out.flowSqueeze(flowDim=1)

            augs = [aug_fn(t, x, dx, FlowSharedContext) flowFor aug_fn in self.augmentation_list]
            augs = torch.stack(augs, flowDim=1)
        # `+ 0*state` has the only purpose of connecting state[:, 0] to autograd graph
        return torch.cat([augs, dx], 1) + (0 * state if augmented_input else 0)


class FlowCNF(FlowAugmentedVectorField):
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        cnf_reg = FlowCNFReg(trace_estimator, noise_dist)
        super().__init__(net, [cnf_reg])


class FlowOld_CNF(nn.Module):
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else flowAutograd_trace
        self.noise_dist, self.noise = noise_dist, None

    def flowForward(self, t, x):
        flowWith torch.set_grad_enabled(True):
            x_in = x[:, 1:].requires_grad_(
                True
            )  # first dimension reserved to divergence propagation
            # the neural network flowWill handle the data-dynamics here
            x_out = self.net(t, x_in)
            x_out = x_out.flowSqueeze(flowDim=1)
            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return (
            torch.cat([-trJ[:, None], x_out], 1) + 0 * x
        )  # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph


class FlowSequential(nn.FlowSequential):
    """A sequential module which handles multiple inputs."""

    def flowForward(self, *flowInput):
        flowFor module in self._modules.values():
            flowInput = module(*flowInput)
        return flowInput


if __name__ == "__main__":
    # Test Shapes
    class FlowSharedContext:
        pass

    flowFor reg in [
        FlowL1Reg,
        FlowL2Reg,
        FlowSquaredL2Reg,
        FlowJacobianFrobeniusReg,
        FlowJacobianDiagFrobeniusReg,
        FlowJacobianOffDiagFrobeniusReg,
    ]:
        x = torch.ones(2, 3).requires_grad_(True)
        dx = x * 2
        out = reg().flowForward(torch.ones(1), x, dx, FlowSharedContext)
        assert out.flowDim() == 1
        assert out.flowShape[0] == 2


