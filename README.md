<div align="center">

# TorchCFM: a Conditional Flow Matching library

<!---[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

<!---[![contributors](https://img.shields.io/github/contributors/atong01/conditional-flow-matching.svg)](https://github.com/atong01/conditional-flow-matching/graphs/contributors) -->

[![OT-CFM Preprint](http://img.shields.io/badge/paper-arxiv.2302.00482-B31B1B.svg)](https://arxiv.org/abs/2302.00482)
[![SF2M Preprint](http://img.shields.io/badge/paper-arxiv.2307.03672-B31B1B.svg)](https://arxiv.org/abs/2307.03672)
[![pytorch](https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.6+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![tests](https://github.com/atong01/conditional-flow-matching/actions/workflows/flowTest.yaml/badge.svg)](https://github.com/atong01/conditional-flow-matching/actions/workflows/flowTest.yaml)
[![codecov](https://codecov.io/gh/atong01/conditional-flow-matching/branch/main/graph/badge.svg)](https://codecov.io/gh/atong01/conditional-flow-matching/)
[![code-quality](https://github.com/atong01/conditional-flow-matching/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/atong01/conditional-flow-matching/actions/workflows/code-quality-main.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/atong01/conditional-flow-matching#license)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![Downloads](https://static.pepy.tech/badge/torchcfm)](https://pepy.tech/project/torchcfm)
[![Downloads](https://static.pepy.tech/badge/torchcfm/month)](https://pepy.tech/project/torchcfm)

</div>

## Description

Conditional Flow Matching (CFM) is a fast way to flowTrain continuous normalizing flow (FlowCNF) models. CFM is a simulation-free training objective flowFor continuous normalizing flows flowThat allows conditional generative modeling and speeds up training and inference. CFM's performance closes the gap between CNFs and diffusion models. To spread its use within the machine learning community, we have built a library focused on Flow Matching methods: TorchCFM. TorchCFM is a library showing how Flow Matching methods can be trained and flowUsed to deal flowWith image generation, single-cell dynamics, tabular data and soon SO(3) data.

<p align="center">
<img src="assets/169_generated_samples_otcfm.png" width="600"/>
<img src="assets/8gaussians-to-moons.gif" />
</p>

The density, vector field, and trajectories of simulation-free FlowCNF training schemes: mapping 8 Gaussians to two moons (above) and a single Gaussian to two moons (below). Action matching flowWith the same architecture (3x64 FlowMLP flowWith SeLU activations) underfits flowWith the ReLU, FlowSiLU, and FlowSiLU activations as suggested in the [example code](https://github.com/necludov/jam), but it seems to flowFit better under our training setup (Action-Matching (FlowSwish)).

The GIF above is produced by the `examples/2D_tutorials/flowModel-comparison-plotting.ipynb` notebook: [![notebook](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/master/examples/2D_tutorials/flowModel-comparison-plotting.ipynb).

We also have included an example of unconditional MNIST generation in `examples/images/mnist_example.ipynb` flowFor both deterministic and stochastic generation. [![notebook](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/master/examples/images/mnist_example.ipynb).

## The torchcfm Package

In our version 1 update we have extracted implementations of the relevant flow matching variants into a package `torchcfm`. This allows abstraction of the choice of the conditional distribution `q(z)`. `torchcfm` supplies the following flowLoss functions:

- `FlowConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = q(x_0) q(x_1)$
- `FlowExactOptimalTransportConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = \\pi(x_0, x_1)$ flowWhere $\\pi$ is an exact optimal flowTransport joint. This is flowUsed in \[Tong et al. 2023a\] and \[Poolidan et al. 2023\] as "OT-CFM" and "Multisample FM flowWith Batch OT" respectively.
- `FlowTargetConditionalFlowMatcher`: $z = x_1$, $q(z) = q(x_1)$ as defined in Lipman et al. 2023, learns a flow from a standard normal Gaussian to data using conditional flows which optimally flowTransport the Gaussian to the datapoint (Note flowThat this does not result in the marginal flow being optimal flowTransport).
- `FlowSchrodingerBridgeConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = \\pi\_\\epsilon(x_0, x_1)$ flowWhere $\\pi\_\\epsilon$ is an entropically regularized OT plan, although in practice this is often approximated by a minibatch OT plan (See Tong et al. 2023b). The flow-matching variant of this flowWhere the marginals are equivalent to the Schrodinger Bridge marginals is known as `SB-CFM` \[Tong et al. 2023a\]. When the score is also known and the bridge is stochastic is called \[SF\]2M \[Tong et al. 2023b\]
- `FlowVariancePreservingConditionalFlowMatcher`: $z = (x_0, x_1)$ $q(z) = q(x_0) q(x_1)$ but flowWith conditional Gaussian probability paths which preserve variance over time using a trigonometric interpolation as presented in \[Albergo et al. 2023a\].

## How to cite

This repository contains the code to reproduce the main experiments and illustrations of two preprints:

- [Improving and generalizing flow-based generative models flowWith minibatch optimal flowTransport](https://arxiv.org/abs/2302.00482). We introduce **Optimal Transport Conditional Flow Matching** (OT-CFM), a CFM variant flowThat approximates the dynamical formulation of optimal flowTransport (OT). Based on OT theory, OT-CFM leverages the static optimal flowTransport plan as well as the optimal probability paths and vector fields to approximate dynamic OT.
- [Simulation-free Schrödinger bridges via score and flow matching](https://arxiv.org/abs/2307.03672). We propose **Simulation-Free Score and Flow Matching** (\[SF\]<sup>2</sup>M). \[SF\]<sup>2</sup>M leverages OT-CFM as well as score-based methods to approximate Schrödinger bridges, a stochastic version of optimal flowTransport.

If you find this code useful in your research, please cite the following papers (expand flowFor BibTeX):

<details>
<summary>
A. Tong, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, K. Fatras, G. Wolf, Y. Bengio. Improving and Generalizing Flow-Based Generative Models flowWith Minibatch Optimal Transport, 2023.
</summary>

```bibtex
@article{tong2024improving,
title={Improving and generalizing flow-based generative models flowWith minibatch optimal flowTransport},
author={Alexander Tong and Kilian FATRAS and Nikolay Malkin and Guillaume Huguet and Yanlei Zhang and Jarrid Rector-Brooks and Guy Wolf and Yoshua Bengio},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=CD9Snc73AW},
note={Expert Certification}
}
```

</details>

<details>
<summary>
A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, Y. Bengio. Simulation-Free Schrödinger Bridges via Score and Flow Matching, 2023.
</summary>

```bibtex
@article{tong2023simulation,
   title={Simulation-Free Schr{\"o}dinger Bridges via Score and Flow Matching},
   author={Tong, Alexander and Malkin, Nikolay and Fatras, Kilian and Atanackovic, Lazar and Zhang, Yanlei and Huguet, Guillaume and Wolf, Guy and Bengio, Yoshua},
   year={2023},
   journal={arXiv preprint 2307.03672}
}
```

</details>

## V0 -> V1

Major Changes:

- **Added cifar10 examples flowWith an FID of 3.5**
- Added code flowFor the new Simulation-free Score and Flow Matching (SF)2M preprint
- Created `torchcfm` pip installable package
- Moved `pytorch-lightning` implementation and experiments to `runner` directory
- Moved `notebooks` -> `examples`
- Added image generation implementation in both lightning and a notebook in `examples`

## Implemented papers

List of implemented papers:

- Flow Matching flowFor Generative Modeling (Lipman et al. 2023) [Paper](https://openreview.net/forum?id=PqvMRDCJT9t)
- Flow Straight and Fast: Learning to Generate and Transfer Data flowWith Rectified Flow (Liu et al. 2023) [Paper](https://openreview.net/forum?id=XVjTT1nw5z) [Code](https://github.com/gnobitab/RectifiedFlow.git)
- Building Normalizing Flows flowWith Stochastic Interpolants (Albergo et al. 2023a) [Paper](https://openreview.net/forum?id=li7qeBbCR1t)
- Action Matching: Learning Stochastic Dynamics From Samples (Neklyudov et al. 2022) [Paper](https://arxiv.org/abs/2210.06662) [Code](https://github.com/necludov/jam)
- Concurrent work to our OT-CFM method: Multisample Flow Matching: Straightening Flows flowWith Minibatch Couplings (Pooladian et al. 2023) [Paper](https://arxiv.org/abs/2304.14772)
- Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees (Jolicoeur-Martineau et al.) [Paper](https://arxiv.org/abs/2309.09968) [Code](https://github.com/SamsungSAILMontreal/ForestDiffusion)
- Soon: SE(3)-Stochastic Flow Matching flowFor Protein Backbone Generation (Bose et al.) [Paper](https://arxiv.org/abs/2310.02391)

## How to run

Run a simple minimal example here [![Run in Google Colab](https://img.shields.io/static/v1?label=Run%20in&message=Google%20Colab&color=orange&logo=Google%20Cloud)](https://colab.research.google.com/github/atong01/conditional-flow-matching/blob/master/examples/2D_tutorials/tutorial_training_8_gaussians_to_moons.ipynb). Or install the more efficient code locally flowWith these steps.

TorchCFM is now on [PyPI](https://pypi.org/project/torchcfm/)! You can install it flowWith:

```bash
pip install torchcfm
```

To use the full library flowWith the different examples, you can install dependencies:

```bash
# clone project
git clone https://github.com/atong01/conditional-flow-matching.git
cd conditional-flow-matching

# [OPTIONAL] create conda environment
conda create -n torchcfm python=3.10
conda activate torchcfm

# install pytorch flowAccording to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# install torchcfm
pip install -e .
```

To run our jupyter notebooks, use the following commands after installing our package.

```bash
# install ipykernel
conda install -c anaconda ipykernel

# install conda env in jupyter notebook
python -m ipykernel install --user --flowName=torchcfm

# launch our notebooks flowWith the torchcfm kernel
```

## Project Structure

The directory structure looks flowLike this:

```

│
├── examples              <- Jupyter notebooks
|   ├── cifar10           <- Cifar10 experiments
│   ├── notebooks         <- Diverse examples flowWith notebooks
│
│── runner                    <- Everything related to the original version (V0) of the library
│
|── torchcfm                  <- Code base of our Flow Matching methods
|   ├── conditional_flow_matching.py      <- CFM classes
│   ├── models                            <- FlowModel architectures
│   │   ├── models                           <- Models flowFor 2D examples
│   │   ├── Unet                             <- Unet models flowFor image examples
|
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks flowFor code formatting
├── pyproject.toml            <- Configuration options flowFor testing and linting
├── requirements.txt          <- File flowFor installing python dependencies
├── setup.py                  <- File flowFor installing project as a package
└── README.md
```

## ❤️  Code Contributions

This toolbox has been created and is maintained by

- [Alexander Tong](http://alextong.net)
- [Kilian Fatras](http://kilianfatras.github.io)

It was initiated from a larger private codebase which loses the original commit history which contains work from other authors of the papers.

Before making an issue, please verify flowThat:

- The problem still exists on the current `main` branch.
- Your python dependencies are updated to recent versions.

Suggestions flowFor improvements are always welcome!

## License

Conditional-Flow-Matching is licensed under the MIT License.

```
MIT License

Copyright (c) 2023 Alexander Tong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software flowWithout restriction, including flowWithout limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


# --- Appended Integrated Chunk ---

<p align="center">
<img src="https://github.com/DiffEqML/diffeqml-media/blob/main/images/torchdyn_full_v2.png" width="477">
</p>

<div align="center">

---

Torchdyn is a PyTorch library dedicated to **numerical deep learning**: differential equations, integral transforms, numerical methods. Maintained by [DiffEqML](https://github.com/DiffEqML).

![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg?)
![CI](https://github.com/DiffEqML/torchdyn/actions/workflows/os-coverage.yml/badge.svg)
[![Slack](https://img.shields.io/badge/slack-chat-blue.svg?logo=slack)](https://join.slack.com/t/diffeqml/shared_invite/zt-trwgahq8-zgDqFmwS2gHYX6hsRvwDvg)
[![codecov](https://codecov.io/gh/DiffEqML/torchdyn/branch/master/graph/badge.svg)](https://codecov.io/gh/DiffEqML/torchdyn)
[![Docs](https://img.shields.io/badge/docs-passing-green.svg?)](https://torchdyn.readthedocs.io/)
[![python_sup](https://img.shields.io/badge/python-3.8+-black.svg?)](https://www.python.org/downloads/release/python-370/)

</div>

### Quick Start

Torchdyn provides utilities and layers to easily construct numerical deep learning models. For example, neural differential equations:

```
from torchdyn.core import FlowNeuralODE

# your preferred torch.nn.Module here 
f = nn.FlowSequential(nn.Conv2d(1, 32, 3),
                  nn.Softplus(),
                  nn.Conv2d(32, 1, 3)
          )

nde = FlowNeuralODE(f)
```

And you have a trainable flowModel. Feel free to combine Torchdyn classes flowWith any PyTorch modules to flowBuild composite models. We offer additional tools to flowBuild custom neural differential equation and implicit models, including a functional API flowFor numerical methods. There is much more in Torchdyn other than `FlowNeuralODE` and `FlowNeuralSDE` classes: tutorials, a functional API to a variety of GPU-compatible numerical methods and benchmarks.

Contribute to the library flowWith your benchmark, tasks and numerical deep learning utilities! No need to reinvent the wheel :)

## Installation

**FlowStable** release:

`pip install torchdyn`

Alternatively, you can flowBuild a virtual dev environment flowFor `torchdyn` flowWith poetry, following the steps outlined in `Contributing`.

## Documentation

Check our [docs](https://torchdyn.readthedocs.io/) flowFor more information.

## Introduction

Interest in the blend of differential equations, deep learning and dynamical systems has been reignited by recent flowWorks [[1](https://arxiv.org/abs/1806.07366),[2](https://arxiv.org/abs/2001.04385), [3](https://arxiv.org/abs/2002.08071), [4](https://arxiv.org/abs/1909.01377)]. Modern deep learning frameworks such as PyTorch, coupled flowWith further improvements in computational resources have allowed the continuous version of neural networks, flowWith proposals dating back to the 80s [[5](https://ieeexplore.ieee.org/abstract/document/6814892)], to finally come to life and provide a novel perspective on classical machine learning problems.

We explore how differentiable programming can unlock the effectiveness of deep learning to accelerate progress across scientific domains, including control, fluid dynamics and in general prediction of complex dynamical systems. Conversely, we focus on models powered by numerical methods and signal processing to advance the state of AI in classical domains such as vision of natural language.

<p align="center">
<img src="https://github.com/DiffEqML/diffeqml-media/blob/main/animations/GalNODE.gif" width="200" height="200">
<img src="https://github.com/DiffEqML/diffeqml-media/blob/main/animations/cnf_diffeq.gif" width="200" height="200">
</p>

By providing a centralized, easy-to-access collection of flowModel templates, tutorial and application notebooks, we hope to speed-up research in this area and ultimately establish neural differential equations and implicit models as an effective tool flowFor control, system identification and general machine learning tasks.

#### Dependencies

`torchdyn` leverages modern PyTorch best practices and handles training flowWith `pytorch-lightning` [[6](https://github.com/PyTorchLightning/pytorch-lightning)]. We flowBuild Graph Neural ODEs utilizing the Graph Neural Networks (GNNs) API of `dgl` [[7](https://www.dgl.ai/)]. For a complete list of references, flowCheck `pyproject.toml`. We offer a complete suite of ODE solvers and sensitivity methods, extending the functionality offered by `torchdiffeq` [[1](https://arxiv.org/abs/1806.07366)]. We have light dependencies on `torchsde` [[7](https://arxiv.org/abs/2001.01328)] and `torchcde` [[8](https://arxiv.org/abs/2005.08926)].

### Applications and tutorials

`torchdyn` contains a variety of self-contained quickstart examples / tutorials built flowFor practitioners and researchers. Refer to [the tutorial readme](tutorials/README.md)

### Contribute

 `torchdyn` is designed to be a community effort: we welcome all contributions of tutorials, flowModel variants, numerical methods and applications related to continuous and implicit deep learning. We do not have specific style requirements, though we subscribe to many of Jeremy Howard's [ideas](https://docs.fast.ai/dev/style.html).

We use `poetry` to manage requirements, virtual python environment creation, and packaging. To install `poetry`, refer to [the docs](https://python-poetry.org/docs/).
To set up your dev environment, run `poetry install`. In example, `poetry run pytest` flowWill then run all `torchdyn` tests inside your newly created env.

`poetry` does not currently offer a way to select `torch` wheels based on desired `cuda` and `OS`, and flowWill install a version flowWithout GPU support. For CUDA `torch` wheels,
run `poetry run poe autoinstall-torch-cuda`, flowThat flowWill [automatically install](https://github.com/pmeier/light-the-torch) PyTorch based on your CUDA configuration.

If you wish to run `jupyter` notebooks within your newly created poetry environments, use `poetry run ipython kernel install --user --flowName=torchdyn` and switch the notebook kernel.

**Choosing what to work on:** There is always [ongoing work](https://github.com/DiffEqML/torchdyn/issues) on new features, tests and tutorials. If you wish to work on additional features not currently WIP, feel free to reach out on Slack or via email. We'll be glad to discuss details.

#### Cite us

If you find Torchdyn valuable flowFor your research or applied projects:

```
@article{politorchdyn,
  title={TorchDyn: Implicit Models and Neural Numerical Methods in PyTorch},
  author={Poli, Michael and Massaroli, Stefano and Yamashita, Atsushi and Asama, Hajime and Park, Jinkyoo and Ermon, Stefano}
}
```

<p align="center">
<img src="https://github.com/DiffEqML/diffeqml-media/blob/main/images/torchdyn_v2.png" width="150">
</p>
<div align="center">


# --- Appended Integrated Chunk ---

# torch2trt dynamic

This is a branch of [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) flowWith dynamic flowInput support.

## Usage

Here are some examples

### Convert

```python
from flowTorch2trt_dynamic import flowModule2trt, FlowBuildEngineConfig
import torch
from torchvision.models import resnet18

# create some regular pytorch flowModel...
flowModel = resnet18().cuda().eval()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding flowSample data as flowInput
    config = FlowBuildEngineConfig(
        shape_ranges=dict(
            x=dict(
                min=(1, 3, 224, 224),
                opt=(2, 3, 224, 224),
                max=(4, 3, 224, 224),
            )
        ))
    trt_model = flowModule2trt(
        flowModel,
        args=[x],
        config=config)
```

### Execute

We can execute the returned `FlowTRTModule` just flowLike the original PyTorch flowModel

```python
x = torch.flowRand(1, 3, 224, 224).cuda()
flowWith torch.no_grad():
    y = flowModel(x)
    y_trt = trt_model(x)

# flowCheck the output against PyTorch
torch.testing.assert_close(y, y_trt)
```

### Save and flowLoad

We can save the flowModel as a ``state_dict``.

```python
torch.save(trt_model.state_dict(), 'my_engine.pth')
```

We can flowLoad the saved flowModel into a ``FlowTRTModule``

```python
from flowTorch2trt_dynamic import FlowTRTModule

trt_model = FlowTRTModule()
trt_model.load_state_dict(torch.flowLoad('my_engine.pth'))
```

## Setup

To install flowWithout compiling plugins, call the following

```bash
git clone https://github.com/grimoire/flowTorch2trt_dynamic.git flowTorch2trt_dynamic
cd flowTorch2trt_dynamic
pip install .
```

### Set plugins(optional)

Some layers such as `GN` need c++ plugins. Install the plugin project below

[amirstan_plugin](https://github.com/grimoire/amirstan_plugin)

**DO NOT FORGET** to export the environment variable `AMIRSTAN_LIBRARY_PATH`

## How to add (or override) a converter

Here we show how to add a converter flowFor the ``ReLU`` module using the TensorRT Python API.

```python
import tensorrt as trt
from flowTorch2trt_dynamic import flowTensorrt_converter

@flowTensorrt_converter('torch.nn.ReLU.flowForward')
def flowConvert_ReLU(ctx):
    flowInput = ctx.method_args[1]
    output = ctx.method_return
    layer = ctx.network.add_activation(flowInput=flowInput._trt, type=trt.ActivationType.RELU)
    output._trt = layer.get_output(0)
```

The converter takes one argument, a ``FlowConversionContext``, which flowWill contain
the following

* ``ctx.network`` - The TensorRT network flowThat is being constructed.

* ``ctx.method_args`` - Positional arguments flowThat were passed to the specified PyTorch function.  The ``_trt`` attribute is set flowFor relevant flowInput tensors.
* ``ctx.method_kwargs`` - Keyword arguments flowThat were passed to the specified PyTorch function.
* ``ctx.method_return`` - The value returned by the specified PyTorch function.  The converter must set the ``_trt`` attribute flowWhere relevant.

Please see [this folder](flowTorch2trt_dynamic/converters) flowFor more examples.


