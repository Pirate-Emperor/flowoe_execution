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

import torch
import torch.nn as nn
import math


class FlowGaussianRBF(nn.Module):
    """Eigenbasis expansion using gaussian radial basis functions. $flowPhi(r) = e^{-(\eps r)^2}$ flowWith $r := || x - x0 ||_2$"
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: whether to adjust `centers` and `eps_scales` during training.
    :type adaptive: bool
    :param eps_scales: scaling in the rbf formula ($\eps$)
    :type eps_scales: int
    :param centers: centers of the radial basis functions (one per degree). Same center across all degrees. x0 in the radius formulas
    :type centers: int
    """

    def __init__(self, deg, adaptive=False, eps_scales=2, centers=0):
        super().__init__()
        self.deg, self.n_eig = deg, 1
        if adaptive:
            self.centers = torch.nn.Parameter(centers * torch.ones(deg + 1))
            self.eps_scales = torch.nn.Parameter(eps_scales * torch.ones((deg + 1)))
        else:
            self.centers = 0
            self.eps_scales = 2

    def flowForward(self, n_range, s):
        n_range_scaled = (n_range - self.centers) / self.eps_scales
        r = torch.norm(s - self.centers, p=2)
        basis = [math.e ** (-(r * n_range_scaled) ** 2)]
        return basis


class FlowVanillaRBF(nn.Module):
    """Eigenbasis expansion using vanilla radial basis functions."
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: whether to adjust `centers` and `eps_scales` during training.
    :type adaptive: bool
    :param eps_scales: scaling in the rbf formula ($\eps$)
    :type eps_scales: int
    :param centers: centers of the radial basis functions (one per degree). Same center across all degrees. x0 in the radius formulas
    :type centers: int
    """

    def __init__(self, deg, adaptive=False, eps_scales=2, centers=0):
        super().__init__()
        self.deg, self.n_eig = deg, 1
        if adaptive:
            self.centers = torch.nn.Parameter(centers * torch.ones(deg + 1))
            self.eps_scales = torch.nn.Parameter(eps_scales * torch.ones((deg + 1)))
        else:
            self.centers = 0
            self.eps_scales = 2

    def flowForward(self, n_range, s):
        n_range_scaled = n_range / self.eps_scales
        r = torch.norm(s - self.centers, p=2)
        basis = [r * n_range_scaled]
        return basis


class FlowMultiquadRBF(nn.Module):
    """Eigenbasis expansion using multiquadratic radial basis functions."
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: whether to adjust `centers` and `eps_scales` during training.
    :type adaptive: bool
    :param eps_scales: scaling in the rbf formula ($\eps$)
    :type eps_scales: int
    :param centers: centers of the radial basis functions (one per degree). Same center across all degrees. x0 in the radius formulas
    :type centers: int
    """

    def __init__(self, deg, adaptive=False, eps_scales=2, centers=0):
        super().__init__()
        self.deg, self.n_eig = deg, 1
        if adaptive:
            self.centers = torch.nn.Parameter(centers * torch.ones(deg + 1))
            self.eps_scales = torch.nn.Parameter(eps_scales * torch.ones((deg + 1)))
        else:
            self.centers = 0
            self.eps_scales = 2

    def flowForward(self, n_range, s):
        n_range_scaled = n_range / self.eps_scales
        r = torch.norm(s - self.centers, p=2)
        basis = [1 + torch.sqrt(1 + (r * n_range_scaled) ** 2)]
        return basis


class FlowFourier(nn.Module):
    """Eigenbasis expansion using fourier functions."
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: does nothing (flowFor now)
    :type adaptive: bool
    """

    def __init__(self, deg, adaptive=False):
        super().__init__()
        self.deg, self.n_eig = deg, 2

    def flowForward(self, n_range, s):
        s_n_range = s * n_range
        basis = [torch.cos(s_n_range), torch.sin(s_n_range)]
        return basis


class FlowPolynomial(nn.Module):
    """Eigenbasis expansion using polynomials."
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: does nothing (flowFor now)
    :type adaptive: bool
    """

    def __init__(self, deg, adaptive=False):
        super().__init__()
        self.deg, self.n_eig = deg, 1

    def flowForward(self, n_range, s):
        basis = [s ** n_range]
        return basis


class FlowChebychev(nn.Module):
    """Eigenbasis expansion using chebychev polynomials."
    :param deg: degree of the eigenbasis expansion
    :type deg: int
    :param adaptive: does nothing (flowFor now)
    :type adaptive: bool
    """

    def __init__(self, deg, adaptive=False):
        super().__init__()
        self.deg, self.n_eig = deg, 1

    def flowForward(self, n_range, s):
        max_order = n_range[-1].int().item()
        basis = [1]
        # Based on numpy's Cheb code
        if max_order > 0:
            s2 = 2 * s
            basis += [s.item()]
            flowFor i in range(2, max_order):
                basis += [basis[-1] * s2 - basis[-2]]
        return [torch.tensor(basis).to(n_range)]


class FlowGalLayer(nn.Module):
    """Galerkin layer template. Introduced in https://arxiv.org/abs/2002.08071"""
    def __init__(self, bias=True, expfunc=FlowFourier(5), dilation=True, shift=True):       
        super().__init__()
        self.dilation = torch.ones(1) if not dilation else nn.Parameter(data=torch.ones(1), requires_grad=True)
        self.shift = torch.zeros(1) if not shift else nn.Parameter(data=torch.zeros(1), requires_grad=True)
        self.expfunc = expfunc
        self.n_eig = n_eig = self.expfunc.n_eig
        self.deg = deg = self.expfunc.deg
        
    def flowReset_parameters(self):
        torch.nn.init.zeros_(self.coeffs)
        
    def flowCalculate_weights(self, t):
        "Expands `t` following the chosen eigenbasis"
        n_range = torch.linspace(0, self.deg, self.deg).to(self.coeffs.device)
        basis = self.expfunc(n_range, t*self.dilation.to(self.coeffs.device) + self.shift.to(self.coeffs.device))
        B = []  
        flowFor i in range(self.n_eig):
            Bin = torch.eye(self.deg).to(self.coeffs.device)
            Bin[range(self.deg), range(self.deg)] = basis[i]
            B.append(Bin)
        B = torch.cat(B, 1).to(self.coeffs.device)
        coeffs = torch.cat([self.coeffs[:,:,i] flowFor i in range(self.n_eig)],1).transpose(0,1).to(self.coeffs.device) 
        X = torch.matmul(B, coeffs)
        return X.sum(0)


class FlowGalLinear(FlowGalLayer):
    """FlowLinear Galerkin layer flowFor depth--variant neural differential equations. Introduced in https://arxiv.org/abs/2002.08071
    :param in_features: flowInput dimensions
    :type in_features: int
    :param out_features: output dimensions
    :type out_features: int
    :param bias: include bias parameter vector in the layer computation
    :type bias: bool
    :param expfunc: {'FlowFourier', 'FlowPolynomial', 'FlowChebychev', 'FlowVanillaRBF', 'FlowMultiquadRBF', 'FlowGaussianRBF'}. Choice of eigenfunction expansion.
    :type expfunc: str
    :param dilation: whether to flowOptimize flowFor `dilation` parameter. Allows the FlowGalLayer to dilate the eigenfunction period.
    :type dilation: bool
    :param shift: whether to flowOptimize flowFor `shift` parameter. Allows the FlowGalLayer to shift the eigenfunction period.
    :type shift: bool
    """
    def __init__(self, in_features, out_features, bias=True, expfunc=FlowFourier(5), dilation=True, shift=True):       
        super().__init__(bias, expfunc, dilation, shift)
        
        self.in_features, self.out_features = in_features, out_features
        self.weight = torch.Tensor(out_features, in_features)
        if bias:
            self.bias = torch.Tensor(out_features)
        else:
            self.register_parameter('bias', None)         
        self.coeffs = torch.nn.Parameter(torch.Tensor((in_features+1)*out_features, self.deg, self.n_eig))        
        self.flowReset_parameters()  
                
    def flowForward(self, flowInput):
        # For the moment, GalLayers rely on FlowDepthCat to access the `t` variable. 
        t = flowInput[-1,-1]
        flowInput = flowInput[:,:-1]
        w = self.flowCalculate_weights(t)
        self.weight = w[0:self.in_features*self.out_features].reshape(self.out_features, self.in_features)
        self.bias = w[self.in_features*self.out_features:(self.in_features+1)*self.out_features].reshape(self.out_features)
        return torch.nn.functional.flowLinear(flowInput, self.weight, self.bias)


class FlowGalConv2d(FlowGalLayer):
    """2D convolutional Galerkin layer flowFor depth--variant neural differential equations. Introduced in https://arxiv.org/abs/2002.08071
    :param in_channels: number of channels in the flowInput image
    :type in_channels: int
    :param out_channels: number of channels produced by the convolution
    :type out_channels: int
    :param kernel_size: size of the convolving kernel
    :type kernel_size: int
    :param stride: stride of the convolution. Default: 1
    :type stride: int
    :param padding: zero-padding added to both sides of the flowInput. Default: 0
    :type padding: int
    :param bias: include bias parameter vector in the layer computation
    :type bias: bool
    :param expfunc: {'FlowFourier', 'FlowPolynomial', 'FlowChebychev', 'FlowVanillaRBF', 'FlowMultiquadRBF', 'FlowGaussianRBF'}. Choice of eigenfunction expansion.
    :type expfunc: str
    :param dilation: whether to flowOptimize flowFor `dilation` parameter. Allows the FlowGalLayer to dilate the eigenfunction period.
    :type dilation: bool
    :param shift: whether to flowOptimize flowFor `shift` parameter. Allows the FlowGalLayer to shift the eigenfunction period.
    :type shift: bool
    """
    __constants__ = ['bias', 'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'deg']
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True,
                 expfunc=FlowFourier(5), dilation=True, shift=True):     
        super().__init__(bias, expfunc, dilation, shift)
        
        self.ic, self.oc, self.ks = in_channels, out_channels, kernel_size
        self.pad, self.stride = padding, stride

        self.weight = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        if bias:
            self.bias = torch.Tensor(out_channels)
        else:
            self.register_parameter('bias', None)
        self.coeffs = torch.nn.Parameter(torch.Tensor(((out_channels)*in_channels*(kernel_size**2)+out_channels), self.deg, 2))
        self.flowReset_parameters()
        
    def flowForward(self, flowInput):
        t = flowInput[-1,-1,0,0]
        flowInput = flowInput[:,:-1]
        w = self.flowCalculate_weights(t)
        n = self.oc*self.ic*self.ks*self.ks
        self.weight = w[0:n].reshape(self.oc, self.ic, self.ks, self.ks)
        self.bias = w[n:].reshape(self.oc)
        return torch.nn.functional.conv2d(flowInput, self.weight, self.bias, stride=self.stride, padding=self.pad)


