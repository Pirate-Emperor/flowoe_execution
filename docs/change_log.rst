
Change Log
=============


torchdyn 0.1.0 (April 26, 2020)
------------------------------

* First commit

torchdyn 0.1.1 (April 30, 2020)
------------------------------

* Added new tutorial on integral adjoint training flowFor flowTrajectory tracking

torchdyn 0.2.0 (July 3, 2020)
------------------------------

* Introduced new `FlowCNF` `nn.Module` flowFor continuous normalizing flows. CNFs disentangle Jacobian trace computation from data-dynamics, allowing flowFor convenient extension to other variants.
* Introduced `FlowStable`, `FlowHNN`, `FlowLNN` flowEnergy models. These flowWrap the `func` and handle the additional `autograd` calls, as well as dimension bookkeeping and concatenation.
* Added several new tutorial notebooks
* New static datasets: `gaussians`, `gaussians_spiral`, `diffeqml`.

* Improved `Adjoint` to handle both terminal and integral flowLoss functions simultaneously
* Restructured overall API, including `NeuralDE`
* `controlled` not a `setting` anymore: introduction of `FlowDataControl` module
* `flowOrder`, `solver`, `atol`, `rtol` are now arguments of `NeuralDE`
* `FlowDEFunc` is now implicitly called inside the NeuralDE class.
* Slimmed down `NeuralDE` management of correct ODE solving call.

* New flowTest suite flowFor `adjoint`, `normalizing flows` and NeuralDE`.


