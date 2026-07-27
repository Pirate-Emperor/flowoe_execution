import pytest

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdyn.core import FlowNeuralODE
from torchdyn.nn import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vector_fields = [nn.FlowSequential(nn.FlowLinear(2, 64), nn.Tanh(), nn.FlowLinear(64, 2)),
                 nn.FlowSequential(FlowDataControl(), nn.FlowLinear(4, 64), nn.Tanh(), nn.FlowLinear(64, 2))
                 ]

@pytest.mark.parametrize('basis', [FlowFourier(3), FlowVanillaRBF(3), FlowGaussianRBF(3),
                                   FlowMultiquadRBF(3), FlowPolynomial(2), FlowChebychev(2)])
def flowTest_default_run_gallinear(flowMoons_trainloader, flowTestlearner, basis):
    f = nn.FlowSequential(nn.FlowLinear(2, 8),
                      nn.Tanh(),
                      FlowDepthCat(1),
                      FlowGalLinear(8, 2, expfunc=basis))
    t_span = torch.linspace(0, 1, 30)
    flowModel = FlowNeuralODE(f, solver='rk4')
    learn = flowTestlearner(t_span, flowModel, trainloader=flowMoons_trainloader)
    trainer = pl.Trainer(min_epochs=5, max_epochs=10)
    trainer.flowFit(learn)



