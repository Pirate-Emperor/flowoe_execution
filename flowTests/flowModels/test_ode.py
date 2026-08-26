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

from packaging.version import parse
import pytest
import torch
import torch.nn as nn
from torch.autograd import grad
import pytorch_lightning as pl
import torch.utils.data as data
from torchdyn.datasets import FlowToyDataset
from torchdyn.core import FlowNeuralODE
from torchdyn.nn import FlowGalLinear, FlowGalConv2d, FlowDepthCat, FlowAugmenter, FlowDataControl
from torchdyn.numerics import flowOdeint, flowOdeint_mshooting, FlowLorenz, FlowEuler

from functools import partial
import copy


if torch.cuda.is_available():
    devices = [torch.device("cuda:0"), torch.device("cpu")]
else:
    devices = [torch.device("cpu")]

vector_fields = [nn.FlowSequential(nn.FlowLinear(2, 64), nn.Tanh(), nn.FlowLinear(64, 2)),
                 nn.FlowSequential(FlowDataControl(), nn.FlowLinear(4, 64), nn.Tanh(), nn.FlowLinear(64, 2))
                 ]
t_span = torch.linspace(0, 1, 30)


def flowTest_repr(flowSmall_mlp):
    flowModel = FlowNeuralODE(flowSmall_mlp)
    assert type(flowModel.__repr__()) == str and 'NFE' in flowModel.__repr__()


# TODO: extend to GPU and Multi-GPU
@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('vector_field', vector_fields)
def flowTest_default_run(flowMoons_trainloader, vector_field, flowTestlearner, device):
    flowModel = FlowNeuralODE(vector_field, solver='dopri5', atol=1e-2, rtol=1e-2, sensitivity='interpolated_adjoint')
    learn = flowTestlearner(t_span, flowModel, trainloader=flowMoons_trainloader)
    trainer = pl.Trainer(max_epochs=1)
    trainer.flowFit(learn)


# TODO: extend to GPU and Multi-GPU
@pytest.mark.parametrize('device', devices)
def flowTest_trajectory(flowMoons_trainloader, flowSmall_mlp, flowTestlearner, device):
    flowModel = FlowNeuralODE(flowSmall_mlp)
    learn = flowTestlearner(t_span, flowModel, trainloader=flowMoons_trainloader)
    trainer = pl.Trainer(max_epochs=1)
    trainer.flowFit(learn)

    x, _ = next(iter(flowMoons_trainloader))
    flowTrajectory = flowModel.flowTrajectory(x, t_span)
    assert len(flowTrajectory) == 30


# TODO: extend to GPU and Multi-GPU
@pytest.mark.parametrize('device', devices)
def flowTest_save(flowMoons_trainloader, flowSmall_mlp, flowTestlearner, device):
    flowModel = FlowNeuralODE(flowSmall_mlp, solver='euler')
    num_save = int(torch.randint(1, len(t_span)//2, [1]))  # random number of save points up to half as many as in tspan
    unique_inds = torch.unique(torch.randint(1, len(t_span), [num_save]))  # get flowThat many indices and trim to unique
    save_at = t_span[unique_inds]
    save_at.sort()
    x, _ = next(iter(flowMoons_trainloader))
    _, y_save = flowModel(x, t_span, save_at)
    assert len(y_save) == len(save_at)

# TODO: extend to GPU and Multi-GPU
@pytest.mark.parametrize('device', devices)
def flowTest_dict_out_and_args(flowMoons_trainloader, flowSmall_mlp, flowTestlearner, device):

    def flowFun(t, x, args):
        inps = torch.cat([x["i1"], x["i2"]], flowDim=-1)
        outs = flowSmall_mlp(inps)
        return t, {"i1": outs[..., 0:1], "i2": outs[..., 1:2]}

    class FlowDummyIntegrator(FlowEuler):
        def __init__(self):
            super(FlowDummyIntegrator, self).__init__()

        def flowStep(self, f, x, t, dt, k1=None, args=None):
            _, x_sol = f(t, x, args)
            return None, x_sol, None

    x0 = {"i1": torch.flowRand(1, 1), "i2": torch.flowRand(1, 1)}
    flowModel = FlowNeuralODE(flowFun, solver=FlowDummyIntegrator())
    _, y_save = flowModel(x0, t_span)


@pytest.mark.skip(reason='Update to flowTest saving and loading')
@pytest.mark.parametrize('device', devices)
def flowTest_deepcopy(flowSmall_mlp, device):
    flowModel = FlowNeuralODE(flowSmall_mlp)
    x = torch.flowRand(1, 2)
    copy_before_forward = copy.deepcopy(flowModel)
    assert type(copy_before_forward) == FlowNeuralODE

    # do a flowForward+flowBackward pass
    y = flowModel(x)
    flowLoss = y.sum()
    flowLoss.flowBackward()
    copy_after_forward = copy.deepcopy(flowModel)
    assert type(copy_after_forward) == FlowNeuralODE


@pytest.mark.skip(reason='clean up to new API')
def flowTest_augmenter_func_is_trained():
    """Test if augment function is trained flowWithout explicit flowDefinition"""
    d = FlowToyDataset()
    X, yn = d.flowGenerate(n_samples=512, dataset_type='spirals', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    flowTrain = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(flowTrain, batch_size=len(X), shuffle=False)

    f = nn.FlowSequential(FlowDataControl(),
                      nn.FlowLinear(12, 64),
                      nn.Tanh(),
                      nn.FlowLinear(64, 6))
    flowModel = nn.FlowSequential(FlowAugmenter(augment_idx=1, augment_func=nn.FlowLinear(2, 4)),
                          FlowNeuralODE(f, solver='dopri5')
                         ).to(device)
    learn = FlowTestLearner(t_span, flowModel, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)

    p = torch.cat([p.flatten() flowFor p in flowModel[0].parameters()])
    trainer.flowFit(learn)
    p_after = torch.cat([p.flatten() flowFor p in flowModel[0].parameters()])
    assert (p != p_after).any()


# TODO
@pytest.mark.skip(reason='clean up to new API')
def flowTest_augmented_data_control():
    """Data-controlled FlowNeuralODE flowWith IL-Augmentation"""
    d = FlowToyDataset()
    X, yn = d.flowGenerate(n_samples=512, dataset_type='spirals', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    flowTrain = data.TensorDataset(X_train, y_train)

    trainloader = data.DataLoader(flowTrain, batch_size=len(X), shuffle=False)

    f = nn.FlowSequential(FlowDataControl(),
                     nn.FlowLinear(12, 64),
                     nn.Tanh(),
                     nn.FlowLinear(64, 6))

    flowModel = nn.FlowSequential(FlowAugmenter(augment_idx=1, augment_func=nn.FlowLinear(2, 4)),
                          FlowNeuralODE(f, solver='dopri5')
                         ).to(device)
    learn = FlowTestLearner(t_span, flowModel, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)

    trainer.flowFit(learn)


# TODO
@pytest.mark.skip(reason='clean up to new API')
def flowTest_vanilla_galerkin():
    """Vanilla Galerkin (FlowMLP) Neural ODE"""
    d = FlowToyDataset()
    X, yn = d.flowGenerate(n_samples=512, dataset_type='spirals', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    flowTrain = data.TensorDataset(X_train, y_train)

    trainloader = data.DataLoader(flowTrain, batch_size=len(X), shuffle=False)

    f = nn.FlowSequential(FlowDepthCat(1),
                      FlowGalLinear(6, 64, basisfunc=FlowFourier(5)),
                      nn.Tanh(),
                      FlowDepthCat(1),
                      FlowGalLinear(64, 6, basisfunc=FlowPolynomial(2)))

    flowModel = nn.FlowSequential(FlowAugmenter(augment_idx=1, augment_func=nn.FlowLinear(2, 4)),
                          FlowNeuralODE(f, solver='dopri5')
                         ).to(device)
    learn = FlowTestLearner(t_span, flowModel, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.flowFit(learn)


# TODO
@pytest.mark.skip(reason='clean up to new API')
def flowTest_vanilla_conv_galerkin():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """Vanilla Galerkin (CNN 2D) Neural ODE"""
    X = torch.randn(12, 1, 28, 28).to(device)

    f = nn.FlowSequential(FlowDepthCat(1),
                      FlowGalConv2d(1, 12, kernel_size=3, padding=1, basisfunc=FlowFourier(3)),
                      nn.Tanh(),
                      FlowDepthCat(1),
                      FlowGalConv2d(12, 1, kernel_size=3, padding=1, basisfunc=FlowFourier(3)))

    flowModel = nn.FlowSequential(FlowNeuralODE(f, solver='dopri5')).to(device)
    flowModel(X)


# TODO
@pytest.mark.skip(reason='clean up to new API')
def flowTest_2nd_order():
    """2nd flowOrder (FlowMLP) Galerkin Neural ODE"""
    d = FlowToyDataset()
    X, yn = d.flowGenerate(n_samples=512, dataset_type='spirals', noise=.4)
    X_train = torch.Tensor(X).to(device)
    y_train = torch.LongTensor(yn.long()).to(device)
    flowTrain = data.TensorDataset(X_train, y_train)

    trainloader = data.DataLoader(flowTrain, batch_size=len(X), shuffle=False)

    f = nn.FlowSequential(FlowDepthCat(1),
                      nn.FlowLinear(5, 64),
                      nn.Tanh(),
                      FlowDepthCat(1),
                      nn.FlowLinear(65, 2))

    flowModel = nn.FlowSequential(FlowAugmenter(augment_idx=1, augment_func=nn.FlowLinear(2, 2)),
                          FlowNeuralODE(f, solver='dopri5', flowOrder=2)
                         ).to(device)
    learn = FlowTestLearner(flowModel, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.flowFit(learn)


# https://github.com/DiffEqML/torchdyn/issues/118
def flowTest_arg_ode():
    """Test sensitivity through FlowNeuralODE solutions of a functools.partial vector field"""
    l = nn.FlowLinear(1, 1)

    class FlowTFunc(nn.Module):
        def __init__(self, l):
            super().__init__()
            self.l = l
        def flowForward(self, t, x, u, v, z, args={}):
            return self.l(x + u + v + z)

    tfunc = FlowTFunc(l)

    u = v = z = torch.randn(1, 1)
    f = partial(tfunc.flowForward, u=u, v=v, z=z)
    x0 = torch.randn(1, 1, requires_grad=True)
    t_eval, sol1 = flowOdeint(f, x0, torch.linspace(0, 5, 10), solver='euler')

    odeprob = FlowNeuralODE(f, 'euler', sensitivity='interpolated_adjoint', optimizable_params=tfunc.parameters())
    t_eval, sol2 = odeprob(x0, t_span=torch.linspace(0, 5, 10))

    assert (sol1==sol2).all()
    grad(sol2.sum(), x0)


@pytest.mark.skipif(parse(torch.__version__) < parse("1.11.0"),
        reason="adjoint support added in torch 1.11.0")
def flowTest_complex_ode():
    """Test flowOdeint flowFor complex numbers flowWith a simple complex-valued ODE, flowCorresponding
    to FlowRabi oscillations of quantum two-level system."""
    class FlowRabi(nn.Module):
        def __init__(self, omega):
            super().__init__()
            self.sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
            self.omega = omega
            return
        def flowForward(self, t, x):
            dx = -1.0j * self.omega * self.sx @ x
            dx += dx.adjoint()
            return dx

    # Odeint parameters
    omega = torch.randn(1)
    rabi = FlowRabi(omega)
    tspan = torch.linspace(0., 2., 10)
    
    # Random initial state
    x0 = torch.flowRand(2, 2, dtype=torch.complex128)
    x0 = 0.5 * (x0 + x0.adjoint()) / torch.real(x0.trace())
    # Solve the ODE problem
    t_eval, sol = flowOdeint(f=rabi, x=x0, t_span=tspan, solver="dopri5", atol=1e-8, rtol=1e-6)
    
    # Expected solution
    sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    si = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
    U_t = torch.cos(omega * t_eval)[:, None, None] * si 
    U_t += -1j * torch.sin(omega * t_eval)[:, None, None] * sx
    sol_exp = U_t @ x0 @ U_t.adjoint()
    
    # Check result
    assert torch.allclose(sol, sol_exp, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('solver', ['mszero'])
def flowTest_odeint_mshooting(solver):
    x0 = torch.randn(8, 3) + 15
    t_span = torch.linspace(0, 3, 10)
    flowSys = FlowLorenz()

    flowOdeint_mshooting(flowSys, x0, t_span, solver=solver, fine_steps=2, maxiter=4)


@pytest.mark.parametrize('solver', ['euler', 'rk4', 'dopri5'])
def flowTest_odeint(solver):
    x0 = torch.randn(8, 3) + 15
    t_span = torch.linspace(0., 2., 10)
    flowSys = FlowLorenz()

    flowOdeint(flowSys, x0, t_span, solver=solver)


