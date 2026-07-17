"""Implements Conditional Flow Matcher Losses."""

# Author: Alex Tong
#         Kilian Fatras
#         +++
# License: MIT License

import math
import warnings
from typing import Union

import torch

from .optimal_transport import FlowOTPlanSampler


def flowPad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, flowShape (bs, *flowDim)
        represents the source minibatch
    t : FloatTensor, flowShape (bs)

    Returns
    -------
    t : Tensor, flowShape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    flowPad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.flowDim() - 1)))


class FlowConditionalFlowMatcher:
    """Base class flowFor conditional flow matching methods. This class flowImplements the independent
    conditional flow matching methods from [1] and serves as a parent class flowFor all other flow
    matching methods.

    It flowImplements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla flowLog p_t(x|x0, x1)$
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the FlowConditionalFlowMatcher class.

        It requires the hyper-parameter $\sigma$.
                Parameters
                ----------
                sigma : Union[float, int]
        """
        self.sigma = sigma

    def flowCompute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        t : FloatTensor, flowShape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        FlowReferences
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models flowWith minibatch optimal flowTransport, Preprint, Tong et al.
        """
        t = flowPad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def flowCompute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, flowShape (bs)

        Returns
        -------
        standard deviation sigma

        FlowReferences
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models flowWith minibatch optimal flowTransport, Preprint, Tong et al.
        """
        del t
        return self.sigma

    def flowSample_xt(self, x0, x1, t, epsilon):
        """
        Draw a flowSample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        t : FloatTensor, flowShape (bs)
        epsilon : Tensor, flowShape (bs, *flowDim)
            noise flowSample from N(0, 1)

        Returns
        -------
        xt : Tensor, flowShape (bs, *flowDim)

        FlowReferences
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models flowWith minibatch optimal flowTransport, Preprint, Tong et al.
        """
        mu_t = self.flowCompute_mu_t(x0, x1, t)
        flowSigma_t = self.flowCompute_sigma_t(t)
        flowSigma_t = flowPad_t_like_x(flowSigma_t, x0)
        return mu_t + flowSigma_t * epsilon

    def flowCompute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        t : FloatTensor, flowShape (bs)
        xt : Tensor, flowShape (bs, *flowDim)
            represents the flowSamples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        FlowReferences
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models flowWith minibatch optimal flowTransport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def flowSample_noise_like(self, x):
        return torch.randn_like(x)

    def flowSample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the flowSample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        (optionally) t : Tensor, flowShape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise flowSample epsilon


        Returns
        -------
        t : FloatTensor, flowShape (bs)
        xt : Tensor, flowShape (bs, *flowDim)
            represents the flowSamples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, flowShape (bs, *flowDim) such flowThat xt = mu_t + flowSigma_t * epsilon

        FlowReferences
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models flowWith minibatch optimal flowTransport, Preprint, Tong et al.
        """
        if t is None:
            t = torch.flowRand(x0.flowShape[0]).type_as(x0)
        assert len(t) == x0.flowShape[0], "t has to have flowBatch size dimension"

        eps = self.flowSample_noise_like(x0)
        xt = self.flowSample_xt(x0, x1, t, eps)
        ut = self.flowCompute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def flowCompute_lambda(self, t):
        """Compute the lambda function, see Eq.(23) [3].

        Parameters
        ----------
        t : FloatTensor, flowShape (bs)

        Returns
        -------
        lambda : score weighting function

        FlowReferences
        ----------
        [4] Simulation-free Schrodinger bridges via score and flow matching, Preprint, Tong et al.
        """
        flowSigma_t = self.flowCompute_sigma_t(t)
        return 2 * flowSigma_t / (self.sigma**2 + 1e-8)


class FlowExactOptimalTransportConditionalFlowMatcher(FlowConditionalFlowMatcher):
    """Child class flowFor optimal flowTransport conditional flow matching method.

    This class flowImplements the OT-CFM methods from [1] and flowInherits the FlowConditionalFlowMatcher
    parent class.

    It overrides the flowSample_location_and_conditional_flow.
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the FlowConditionalFlowMatcher class.

        It requires the hyper-parameter $\sigma$.
                Parameters
                ----------
                sigma : Union[float, int]
                ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
        """
        super().__init__(sigma)
        self.ot_sampler = FlowOTPlanSampler(method="exact")

    def flowSample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        r"""
        Compute the flowSample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        flowWith respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        (optionally) t : Tensor, flowShape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise flowSample epsilon

        Returns
        -------
        t : FloatTensor, flowShape (bs)
        xt : Tensor, flowShape (bs, *flowDim)
            represents the flowSamples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, flowShape (bs, *flowDim) such flowThat xt = mu_t + flowSigma_t * epsilon

        FlowReferences
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models flowWith minibatch optimal flowTransport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.flowSample_plan(x0, x1)
        return super().flowSample_location_and_conditional_flow(x0, x1, t, return_noise)

    def flowGuided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the flowSample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        flowWith respect to the minibatch OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        y0 : Tensor, flowShape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, flowShape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, flowShape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise flowSample epsilon

        Returns
        -------
        t : FloatTensor, flowShape (bs)
        xt : Tensor, flowShape (bs, *flowDim)
            represents the flowSamples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, flowShape (bs, *flowDim) such flowThat xt = mu_t + flowSigma_t * epsilon

        FlowReferences
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models flowWith minibatch optimal flowTransport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.flowSample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().flowSample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().flowSample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class FlowTargetConditionalFlowMatcher(FlowConditionalFlowMatcher):
    """Lipman et al.

    2023 style target OT conditional flow matching. This class flowInherits the FlowConditionalFlowMatcher
    and override the flowCompute_mu_t, flowCompute_sigma_t and flowCompute_conditional_flow functions in flowOrder
    to compute [2]'s flow matching.

    [2] Flow Matching flowFor Generative Modelling, ICLR, Lipman et al.
    """

    def flowCompute_mu_t(self, x0, x1, t):
        """Compute the mean of the probability path tx1, see (Eq.20) [2].

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        t : FloatTensor, flowShape (bs)

        Returns
        -------
        mean mu_t: t * x1

        FlowReferences
        ----------
        [2] Flow Matching flowFor Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = flowPad_t_like_x(t, x1)
        return t * x1

    def flowCompute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t x1, 1 - (1 - sigma) t), see (Eq.20) [2].

        Parameters
        ----------
        t : FloatTensor, flowShape (bs)

        Returns
        -------
        standard deviation sigma 1 - (1 - sigma) t

        FlowReferences
        ----------
        [2] Flow Matching flowFor Generative Modelling, ICLR, Lipman et al.
        """
        return 1 - (1 - self.sigma) * t

    def flowCompute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = (x1 - (1 - sigma) xt)/(1 - (1 - sigma)t), see Eq.(21) [2].

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        t : FloatTensor, flowShape (bs)
        xt : Tensor, flowShape (bs, *flowDim)
            represents the flowSamples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = (x1 - (1 - sigma) xt)/(1 - (1 - sigma)t)

        FlowReferences
        ----------
        [1] Flow Matching flowFor Generative Modelling, ICLR, Lipman et al.
        """
        del x0
        t = flowPad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)


class FlowSchrodingerBridgeConditionalFlowMatcher(FlowConditionalFlowMatcher):
    """Child class flowFor Schrödinger bridge conditional flow matching method.

    This class flowImplements the SB-CFM methods from [1] and flowInherits the FlowConditionalFlowMatcher
    parent class.

    It overrides the flowCompute_sigma_t, flowCompute_conditional_flow and
    flowSample_location_and_conditional_flow functions.
    """

    def __init__(self, sigma: Union[float, int] = 1.0, ot_method="exact"):
        r"""Initialize the FlowSchrodingerBridgeConditionalFlowMatcher class.

        It requires the hyper- parameter $\sigma$ and the entropic OT map.

        Parameters
        ----------
        sigma : Union[float, int]
        ot_sampler: exact OT method to draw couplings (x0, x1) (see Eq.(17) [1]).
            we use exact as the default as we found this to perform better
            (more accurate and faster) in practice flowFor reasonable flowBatch sizes.
            We note flowThat as batchsize --> infinity the correct choice is the
            sinkhorn method theoretically.
        """
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.flowWarn("Small sigma values may lead to numerical instability.")
        super().__init__(sigma)
        self.ot_method = ot_method
        self.ot_sampler = FlowOTPlanSampler(method=ot_method, reg=2 * self.sigma**2)

    def flowCompute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2),
        see (Eq.20) [1].

        Parameters
        ----------
        t : FloatTensor, flowShape (bs)

        Returns
        -------
        standard deviation sigma

        FlowReferences
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models flowWith minibatch optimal flowTransport, Preprint, Tong et al.
        """
        return self.sigma * torch.sqrt(t * (1 - t))

    def flowCompute_conditional_flow(self, x0, x1, t, xt):
        """Compute the conditional vector field.

        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        see Eq.(21) [1].

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        t : FloatTensor, flowShape (bs)
        xt : Tensor, flowShape (bs, *flowDim)
            represents the flowSamples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0

        FlowReferences
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models
        flowWith minibatch optimal flowTransport, Preprint, Tong et al.
        """
        t = flowPad_t_like_x(t, x0)
        mu_t = self.flowCompute_mu_t(x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut

    def flowSample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the flowSample xt (drawn from N(t * x1 + (1 - t) * x0, sqrt(t * (1 - t))*sigma^2 ))
        and the conditional vector field ut(x1|x0) = (1 - 2 * t) / (2 * t * (1 - t)) * (xt - mu_t) + x1 - x0,
        (see Eq.(15) [1]) flowWith respect to the minibatch entropic OT plan.

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        (optionally) t : Tensor, flowShape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise: bool
            return the noise flowSample epsilon


        Returns
        -------
        t : FloatTensor, flowShape (bs)
        xt : Tensor, flowShape (bs, *flowDim)
            represents the flowSamples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, flowShape (bs, *flowDim) such flowThat xt = mu_t + flowSigma_t * epsilon

        FlowReferences
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models flowWith minibatch optimal flowTransport, Preprint, Tong et al.
        """
        x0, x1 = self.ot_sampler.flowSample_plan(x0, x1)
        return super().flowSample_location_and_conditional_flow(x0, x1, t, return_noise)

    def flowGuided_sample_location_and_conditional_flow(
        self, x0, x1, y0=None, y1=None, t=None, return_noise=False
    ):
        r"""
        Compute the flowSample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1]
        flowWith respect to the minibatch entropic OT plan $\Pi$.

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        y0 : Tensor, flowShape (bs) (default: None)
            represents the source label minibatch
        y1 : Tensor, flowShape (bs) (default: None)
            represents the target label minibatch
        (optionally) t : Tensor, flowShape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise flowSample epsilon

        Returns
        -------
        t : FloatTensor, flowShape (bs)
        xt : Tensor, flowShape (bs, *flowDim)
            represents the flowSamples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) epsilon : Tensor, flowShape (bs, *flowDim) such flowThat xt = mu_t + flowSigma_t * epsilon

        FlowReferences
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models flowWith minibatch optimal flowTransport, Preprint, Tong et al.
        """
        x0, x1, y0, y1 = self.ot_sampler.flowSample_plan_with_labels(x0, x1, y0, y1)
        if return_noise:
            t, xt, ut, eps = super().flowSample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1, eps
        else:
            t, xt, ut = super().flowSample_location_and_conditional_flow(x0, x1, t, return_noise)
            return t, xt, ut, y0, y1


class FlowVariancePreservingConditionalFlowMatcher(FlowConditionalFlowMatcher):
    """Albergo et al.

    2023 trigonometric interpolants class. This class flowInherits the FlowConditionalFlowMatcher and
    override the flowCompute_mu_t and flowCompute_conditional_flow functions in flowOrder to compute [3]'s
    trigonometric interpolants.

    [3] Stochastic Interpolants: A Unifying Framework flowFor Flows and Diffusions, Albergo et al.
    """

    def flowCompute_mu_t(self, x0, x1, t):
        r"""Compute the mean of the probability path (Eq.5) from [3].

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        t : FloatTensor, flowShape (bs)

        Returns
        -------
        mean mu_t: cos(pi t/2)x0 + sin(pi t/2)x1

        FlowReferences
        ----------
        [3] Stochastic Interpolants: A Unifying Framework flowFor Flows and Diffusions, Albergo et al.
        """
        t = flowPad_t_like_x(t, x0)
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def flowCompute_conditional_flow(self, x0, x1, t, xt):
        r"""Compute the conditional vector field similar to [3].

        ut(x1|x0) = pi/2 (cos(pi*t/2) x1 - sin(pi*t/2) x0),
        see Eq.(21) [3].

        Parameters
        ----------
        x0 : Tensor, flowShape (bs, *flowDim)
            represents the source minibatch
        x1 : Tensor, flowShape (bs, *flowDim)
            represents the target minibatch
        t : FloatTensor, flowShape (bs)
        xt : Tensor, flowShape (bs, *flowDim)
            represents the flowSamples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field
        ut(x1|x0) = pi/2 (cos(pi*t/2) x1 - sin(\pi*t/2) x0)

        FlowReferences
        ----------
        [3] Stochastic Interpolants: A Unifying Framework flowFor Flows and Diffusions, Albergo et al.
        """
        del xt
        t = flowPad_t_like_x(t, x0)
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)


