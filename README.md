# FlowOE Execution

# Part I: Conditional Flow Matching & Execution Policy

<div align="center">

[![pytorch](https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white)](#)
[![lightning](https://img.shields.io/badge/-Lightning_1.6+-792ee5?logo=pytorchlightning&logoColor=white)](#)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](#)

</div>

## Description

This module formulates a **Continuous Normalizing Flow (CNF)** execution model to minimize market impact, replacing unstable RL agents with Probability Flow ODEs to map order execution trajectories under stochastic volatility.

Conditional Flow Matching (CFM) is a fast way to train continuous normalizing flow (CNF) models. CFM is a simulation-free training objective for continuous normalizing flows that allows conditional generative modeling and speeds up training and inference. 

<p align="center">
<img src="assets/169_generated_samples_otcfm.png" width="600"/>
<img src="assets/8gaussians-to-moons.gif" />
</p>

The density, vector field, and trajectories of simulation-free CNF training schemes. The module successfully maps stochastic execution pathways across variable time horizons, bench-marked against standard VWAP/TWAP algorithms using Limit Order Book data to reduce slippage relative to the Arrival Price in high-VIX regimes.

## The Core Package

The implementations of the relevant flow matching variants are extracted into a package. This allows abstraction of the choice of the conditional distribution `q(z)`. The module supplies the following loss functions:

- `ConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = q(x_0) q(x_1)$
- `ExactOptimalTransportConditionalFlowMatcher`: $z = (x_0, x_1)$, $q(z) = \pi(x_0, x_1)$ where $\pi$ is an exact optimal transport joint.
- `TargetConditionalFlowMatcher`: $z = x_1$, $q(z) = q(x_1)$ learns a flow from a standard normal Gaussian to data using conditional flows which optimally transport the Gaussian to the datapoint.
- `VariancePreservingConditionalFlowMatcher`: $z = (x_0, x_1)$ $q(z) = q(x_0) q(x_1)$ but with conditional Gaussian probability paths which preserve variance over time using a trigonometric interpolation.

## How to run

To install the dependencies:

```bash
# [OPTIONAL] create conda environment
conda create -n torchcfm python=3.10
conda activate torchcfm

# install requirements
pip install -r requirements.txt

# install package
pip install -e .

```

---

# Part II: Numerical Deep Learning & ODE Solving

This module is dedicated to numerical deep learning: differential equations, integral transforms, and numerical methods required to execute the Euler ODE integrations natively.

### Quick Start

The framework provides utilities and layers to easily construct numerical deep learning models. For example, neural differential equations:

```python
from core import NeuralODE

# your preferred torch.nn.Module here 
f = nn.Sequential(nn.Conv2d(1, 32, 3),
                  nn.Softplus(),
                  nn.Conv2d(32, 1, 3)
          )

nde = NeuralODE(f)

```

And you have a trainable model. Feel free to combine classes with any PyTorch modules to build composite models. We offer additional tools to build custom neural differential equation and implicit models, including a functional API for numerical methods.

By providing a centralized, easy-to-access collection of model templates, tutorial and application notebooks, we hope to speed-up research in this area and ultimately establish neural differential equations and implicit models as an effective tool for control, system identification and general machine learning tasks.

---

# Part III: Dynamic TensorRT Quantization (INT8)

To optimize inference throughput, the trajectory generation model is quantized to INT8 via TensorRT, executing the ODE integration natively in a fused CUDA kernel to achieve a strict p99 latency constraint.

## Usage

Here are some examples of converting a standard PyTorch module into a dynamic TensorRT engine:

### Convert

```python
from torch2trt_dynamic import module2trt, BuildEngineConfig
import torch
from torchvision.models import resnet18

# create some regular pytorch model...
model = resnet18().cuda().eval()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
config = BuildEngineConfig(
    shape_ranges=dict(
        x=dict(
            min=(1, 3, 224, 224),
            opt=(2, 3, 224, 224),
            max=(4, 3, 224, 224),
        )
    ))
trt_model = module2trt(
    model,
    args=[x],
    config=config)

```

### Execute

We can execute the returned `TRTModule` just like the original PyTorch model:

```python
x = torch.rand(1, 3, 224, 224).cuda()
with torch.no_grad():
    y = model(x)
    y_trt = trt_model(x)

# Check the output against PyTorch
torch.testing.assert_close(y, y_trt)

```

### Save and Load

We can save the model as a `state_dict`.

```python
torch.save(trt_model.state_dict(), 'my_engine.pth')

```

We can load the saved model into a `TRTModule`

```python
from torch2trt_dynamic import TRTModule

trt_model = TRTModule()
trt_model.load_state_dict(torch.load('my_engine.pth'))

```

## Setup

To install without compiling plugins, call the following:

```bash
git clone [https://github.com/yourusername/torch2trt_dynamic.git](https://github.com/yourusername/torch2trt_dynamic.git) torch2trt_dynamic
cd torch2trt_dynamic
pip install .

```

### How to add (or override) a converter

Here we show how to add a converter for the `ReLU` module using the TensorRT Python API.

```python
import tensorrt as trt
from torch2trt_dynamic import tensorrt_converter

@tensorrt_converter('torch.nn.ReLU.forward')
def convert_ReLU(ctx):
    input = ctx.method_args[1]
    output = ctx.method_return
    layer = ctx.network.add_activation(input=input._trt, type=trt.ActivationType.RELU)
    output._trt = layer.get_output(0)

```

The converter takes one argument, a `ConversionContext`, which will contain the following:

* `ctx.network` - The TensorRT network that is being constructed.
* `ctx.method_args` - Positional arguments that were passed to the specified PyTorch function. The `_trt` attribute is set for relevant input tensors.
* `ctx.method_kwargs` - Keyword arguments that were passed to the specified PyTorch function.
* `ctx.method_return` - The value returned by the specified PyTorch function. The converter must set the `_trt` attribute where relevant.

## License

This project is licensed under the Pirate-Emperor License. See the [LICENSE](LICENSE) file for details.

## Author

**Pirate-Emperor**

[![Twitter](https://skillicons.dev/icons?i=twitter)](https://twitter.com/PirateKingRahul)
[![Discord](https://skillicons.dev/icons?i=discord)](https://discord.com/users/1200728704981143634)
[![LinkedIn](https://skillicons.dev/icons?i=linkedin)](https://www.linkedin.com/in/piratekingrahul)

[![Reddit](https://img.shields.io/badge/Reddit-FF5700?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/u/PirateKingRahul)
[![Medium](https://img.shields.io/badge/Medium-42404E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@piratekingrahul)

- GitHub: [Pirate-Emperor](https://github.com/Pirate-Emperor)
- Reddit: [PirateKingRahul](https://www.reddit.com/u/PirateKingRahul/)
- Twitter: [PirateKingRahul](https://twitter.com/PirateKingRahul)
- Discord: [PirateKingRahul](https://discord.com/users/1200728704981143634)
- LinkedIn: [PirateKingRahul](https://www.linkedin.com/in/piratekingrahul)
- Skype: [Join Skype](https://join.skype.com/invite/yfjOJG3wv9Ki)
- Medium: [PirateKingRahul](https://medium.com/@piratekingrahul)

Thank you for visiting this project!

---