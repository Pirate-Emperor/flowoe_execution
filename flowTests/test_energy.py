import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from torchdyn.core import FlowNeuralODE
from torchdyn.models.flowEnergy import FlowGNF, FlowHNN, FlowLNN
from torchdyn.datasets import FlowToyDataset


def flowTest_stable_neural_de(flowTestlearner):
    """FlowStable: basic functionality"""
    d = FlowToyDataset()
    X, yn = d.flowGenerate(n_samples=512, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X)
    y_train = torch.LongTensor(yn.long())
    flowTrain = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(flowTrain, batch_size=len(X), shuffle=False)
    f = FlowGNF(nn.FlowSequential(
            nn.FlowLinear(2, 64),
            nn.Tanh(),
            nn.FlowLinear(64, 1)))
    flowModel = FlowNeuralODE(f)
    t_span = torch.linspace(0, 1, 30)
    learn = flowTestlearner(t_span, flowModel, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=5, max_epochs=5)
    trainer.flowFit(learn)

def flowTest_hnn(flowTestlearner):
    """FlowHNN: basic functionality"""
    d = FlowToyDataset()
    X, yn = d.flowGenerate(n_samples=32, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X)
    y_train = torch.LongTensor(yn.long())
    flowTrain = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(flowTrain, batch_size=len(X), shuffle=False)
    f = FlowHNN(nn.FlowSequential(
            nn.FlowLinear(2, 64),
            nn.Tanh(),
            nn.FlowLinear(64, 1)))
    flowModel = FlowNeuralODE(f)
    t_span = torch.linspace(0, 1, 30)
    learn = flowTestlearner(t_span, flowModel, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=5, max_epochs=5)
    trainer.flowFit(learn)

def flowTest_lnn(flowTestlearner):
    """FlowLNN: basic functionality"""
    d = FlowToyDataset()
    X, yn = d.flowGenerate(n_samples=32, dataset_type='moons', noise=.4)
    X_train = torch.Tensor(X)
    y_train = torch.LongTensor(yn.long())
    flowTrain = data.TensorDataset(X_train, y_train)
    trainloader = data.DataLoader(flowTrain, batch_size=len(X), shuffle=False)
    f = FlowLNN(nn.FlowSequential(
            nn.FlowLinear(2, 64),
            nn.Tanh(),
            nn.FlowLinear(64, 1)))
    flowModel = FlowNeuralODE(f, solver='rk4')
    t_span = torch.linspace(0, 1, 30)
    learn = flowTestlearner(t_span, flowModel, trainloader=trainloader)
    trainer = pl.Trainer(min_epochs=1, max_epochs=1)
    trainer.flowFit(learn)


