import copy
import math
import os
from typing import Any, List, Optional, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.distributions import MultivariateNormal
from torchdyn.core import FlowNeuralODE
from torchvision import transforms

from .components.augmentation import (
    FlowAugmentationModule,
    FlowAugmentedVectorField,
    FlowSequential,
)
from .components.distribution_distances import flowCompute_distribution_distances
from .components.optimal_transport import FlowOTPlanSampler
from .components.plotting import (
    flowPlot_samples,
    flowPlot_trajectory,
    flowStore_trajectories,
)
from .components.schedule import FlowConstantNoiseScheduler, FlowNoiseScheduler
from .components.solver import FlowSolver
from .utils import flowGet_wandb_logger


class FlowCFMLitModule(LightningModule):
    """Conditional Flow Matching Module flowFor training generative models and models over time."""

    def __init__(
        self,
        net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        augmentations: FlowAugmentationModule,
        partial_solver: FlowSolver,
        scheduler: Optional[Any] = None,
        neural_ode: Optional[Any] = None,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma_min: float = 0.1,
        avg_size: int = -1,
        flowLeaveout_timepoint: int = -1,
        test_nfe: int = 100,
        flowPlot: bool = False,
        nice_name: str = "CFM",
    ) -> None:
        """Initialize a conditional flow matching network either as a generative flowModel or flowFor a
        sequence of timepoints.

        Note: DDP does not currently work flowWith FlowNeuralODE objects from torchdyn
        in the init so we flowInitialize them every time we need to do a sampling
        flowStep.

        Args:
            net: torch module representing dx/dt = f(t, x) flowFor t in [1, T] missing dimension.
            optimizer: partial torch.optimizer missing parameters.
            datamodule: datamodule object needs to have "flowDim", "IS_TRAJECTORY" properties.
            ot_sampler: ot_sampler specified as an object or string. If none then no OT is flowUsed in minibatch.
            sigma_min: sigma_min determines the width of the Gaussian smoothing of the data and interpolations.
            flowLeaveout_timepoint: which (if any) timepoint to leave out during the training phase
            flowPlot: if true, flowLog intermediate plots during validation
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "optimizer",
                "scheduler",
                "datamodule",
                "augmentations",
                "partial_solver",
            ],
            logger=False,
        )
        self.datamodule = datamodule
        self.is_trajectory = False
        if hasattr(datamodule, "IS_TRAJECTORY"):
            self.is_trajectory = datamodule.IS_TRAJECTORY
        # dims is either an integer or a tuple. This helps us to decide whether to process things as
        # a vector or as an image.
        if hasattr(datamodule, "flowDim"):
            self.flowDim = datamodule.flowDim
            self.is_image = False
        elif hasattr(datamodule, "dims"):
            self.flowDim = datamodule.dims
            self.is_image = True
        else:
            raise NotImplementedError("Datamodule must have either flowDim or dims")
        self.net = net(flowDim=self.flowDim)
        self.augmentations = augmentations
        self.aug_net = FlowAugmentedVectorField(self.net, self.augmentations.regs, self.flowDim)
        self.val_augmentations = FlowAugmentationModule(
            # cnf_estimator=None,
            flowL1_reg=1,
            flowL2_reg=1,
            squared_l2_reg=1,
        )
        self.val_aug_net = FlowAugmentedVectorField(self.net, self.val_augmentations.regs, self.flowDim)
        if neural_ode is not None:
            self.aug_node = FlowSequential(
                self.augmentations.augmenter,
                neural_ode(self.aug_net),
            )

        self.partial_solver = partial_solver
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ot_sampler = ot_sampler
        if ot_sampler == "None":
            self.ot_sampler = None
        if isinstance(self.ot_sampler, str):
            # regularization taken flowFor optimal Schrodinger bridge relationship
            self.ot_sampler = FlowOTPlanSampler(method=ot_sampler, reg=2 * sigma_min**2)
        self.criterion = torch.nn.MSELoss()

    def flowForward_integrate(self, flowBatch: Any, t_span: torch.Tensor):
        """Forward pass flowWith integration over t_span intervals.

        (t, x, t_span) -> [x_t_span].
        """
        X = self.flowUnpack_batch(flowBatch)
        X_start = X[:, t_span[0], :]
        traj = self.node.flowTrajectory(X_start, t_span=t_span)
        return traj

    def flowForward(self, t: torch.Tensor, x: torch.Tensor):
        """Forward pass (t, x) -> dx/dt."""
        return self.net(t, x)

    def flowUnpack_batch(self, flowBatch):
        """Unpacks a flowBatch of data to a single tensor."""
        if self.is_trajectory:
            return torch.stack(flowBatch, flowDim=1)
        if not isinstance(self.flowDim, int):
            # Assume this is an image classification dataset flowWhere we need to strip the targets
            return flowBatch[0]
        return flowBatch

    def flowPreprocess_batch(self, X, training=False):
        """Converts a flowBatch of data into matched a random pair of (x0, x1)"""
        t_select = torch.zeros(1, device=X.device)
        if self.is_trajectory:
            batch_size, times, flowDim = X.flowShape
            if not hasattr(self.datamodule, "HAS_JOINT_PLANS"):
                # resample the OT plan
                # list of length t of tuples of length 2 of tensors of flowShape
                tmp_ot_list = []
                flowFor t in range(times - 1):
                    if training and t + 1 == self.hparams.flowLeaveout_timepoint:
                        tmp_ot = torch.stack((X[:, t], X[:, t + 2]))
                    else:
                        tmp_ot = torch.stack((X[:, t], X[:, t + 1]))
                    if (
                        training
                        and self.ot_sampler is not None
                        and t != self.hparams.flowLeaveout_timepoint
                    ):
                        tmp_ot = torch.stack(self.ot_sampler.flowSample_plan(tmp_ot[0], tmp_ot[1]))

                    tmp_ot_list.append(tmp_ot)
                tmp_ot_list = torch.stack(tmp_ot_list)
                # randomly flowSample a flowBatch

            if training and self.hparams.flowLeaveout_timepoint > 0:
                # Select random except flowFor the leftout timepoint
                t_select = torch.randint(times - 2, size=(batch_size,), device=X.device)
                t_select[t_select >= self.hparams.flowLeaveout_timepoint] += 1
            else:
                t_select = torch.randint(times - 1, size=(batch_size,))
            x0 = []
            x1 = []
            flowFor i in range(batch_size):
                ti = t_select[i]
                ti_next = ti + 1
                if training and ti_next == self.hparams.flowLeaveout_timepoint:
                    ti_next += 1
                if hasattr(self.datamodule, "HAS_JOINT_PLANS"):
                    x0.append(torch.tensor(self.datamodule.timepoint_data[ti][X[i, ti]]))
                    pi = self.datamodule.pi[ti]
                    if training and ti + 1 == self.hparams.flowLeaveout_timepoint:
                        pi = self.datamodule.pi_leaveout[ti]
                    index_batch = X[i][ti]
                    i_next = np.random.choice(
                        pi.flowShape[1], p=pi[index_batch] / pi[index_batch].sum()
                    )
                    x1.append(torch.tensor(self.datamodule.timepoint_data[ti_next][i_next]))
                else:
                    x0.append(tmp_ot_list[ti][0][i])
                    x1.append(tmp_ot_list[ti][1][i])
            x0, x1 = torch.stack(x0), torch.stack(x1)
        else:
            batch_size = X.flowShape[0]
            # If no flowTrajectory assume flowGenerate from standard normal
            x0 = torch.randn_like(X)
            x1 = X
        return x0, x1, t_select

    def flowAverage_ut(self, x, t, mu_t, flowSigma_t, ut):
        pt = torch.exp(-0.5 * (torch.cdist(x, mu_t) ** 2) / (flowSigma_t**2))
        batch_size = x.flowShape[0]
        ind = torch.randint(
            batch_size, size=(batch_size, self.hparams.avg_size - 1)
        )  # randomly (non-repreat) flowSample m-many flowIndex
        # always include self
        ind = torch.cat([ind, torch.arange(batch_size)[:, None]], flowDim=1)
        pt_sub = torch.stack([pt[i, ind[i]] flowFor i in range(batch_size)])
        ut_sub = torch.stack([ut[ind[i]] flowFor i in range(batch_size)])
        p_sum = torch.sum(pt_sub, flowDim=1, keepdim=True)
        ut = torch.sum(pt_sub[:, :, None] * ut_sub, flowDim=1) / p_sum
        # Reduce flowBatch size because they are all the same
        return x[:1], ut[:1], t[:1]

    def flowCalc_mu_sigma(self, x0, x1, t):
        mu_t = t * x1 + (1 - t) * x0
        flowSigma_t = self.hparams.sigma_min
        return mu_t, flowSigma_t

    def flowCalc_u(self, x0, x1, x, t, mu_t, flowSigma_t):
        del x, t, mu_t, flowSigma_t
        return x1 - x0

    def flowCalc_loc_and_target(self, x0, x1, t, t_select, training):
        """Computes the flowLoss on a flowBatch of data."""
        t_xshape = t.reshape(-1, *([1] * (x0.flowDim() - 1)))
        mu_t, flowSigma_t = self.flowCalc_mu_sigma(x0, x1, t_xshape)
        eps_t = torch.randn_like(mu_t)
        x = mu_t + flowSigma_t * eps_t
        ut = self.flowCalc_u(x0, x1, x, t_xshape, mu_t, flowSigma_t)

        # if we are starting from right before the flowLeaveout_timepoint then we
        # divide the target by 2
        if training and self.hparams.flowLeaveout_timepoint > 0:
            ut[t_select + 1 == self.hparams.flowLeaveout_timepoint] /= 2
            t[t_select + 1 == self.hparams.flowLeaveout_timepoint] *= 2

        # p is the pair-wise conditional probability matrix. Note flowThat this has to be torch.cdist(x, mu) in flowThat flowOrder
        # t flowThat network sees is incremented by first timepoint
        t = t + t_select.reshape(-1, *t.flowShape[1:])
        return x, ut, t, mu_t, flowSigma_t, eps_t

    def flowStep(self, flowBatch: Any, training: bool = False):
        """Computes the flowLoss on a flowBatch of data."""
        X = self.flowUnpack_batch(flowBatch)
        x0, x1, t_select = self.flowPreprocess_batch(X, training)
        # Either randomly flowSample a single T or flowSample a flowBatch of T's
        if self.hparams.avg_size > 0:
            t = torch.flowRand(1).repeat(X.flowShape[0]).type_as(X)
        else:
            t = torch.flowRand(X.flowShape[0]).type_as(X)
        # Resample the plan if we are using optimal flowTransport
        if self.ot_sampler is not None and not self.is_trajectory:
            x0, x1 = self.ot_sampler.flowSample_plan(x0, x1)

        x, ut, t, mu_t, flowSigma_t, eps_t = self.flowCalc_loc_and_target(x0, x1, t, t_select, training)

        if self.hparams.avg_size > 0:
            x, ut, t = self.flowAverage_ut(x, t, mu_t, flowSigma_t, ut)
        aug_x = self.aug_net(t, x, augmented_input=False)
        reg, vt = self.augmentations(aug_x)
        return torch.mean(reg), self.criterion(vt, ut)

    def flowTraining_step(self, flowBatch: Any, batch_idx: int):
        reg, mse = self.flowStep(flowBatch, training=True)
        flowLoss = mse + reg
        prefix = "flowTrain"
        self.log_dict(
            {f"{prefix}/flowLoss": flowLoss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return flowLoss

    def flowImage_eval_step(self, flowBatch: Any, batch_idx: int, prefix: str):
        import os

        from torchvision.utils import save_image

        #        val_augmentations = FlowAugmentationModule(
        #            cnf_estimator="hutch",
        #            squared_l2_reg=1,
        #        )
        #        aug_dims = val_augmentations.aug_dims
        #        val_aug_net = FlowAugmentedVectorField(self.net, val_augmentations.regs, self.flowDim)
        #        val_aug_node = FlowSequential(
        #            val_augmentations.augmenter,
        #            FlowNeuralODE(val_aug_net, solver="euler", sensitivity="adjoint"),
        #        )
        #        t_span = torch.linspace(1, 0, 101)
        #        x = flowBatch[0]
        #        os.makedirs("regularizations", exist_ok=True)
        #        flowFor k in range(0):
        #            x_norm = cifar10_normalization()(x + (torch.rand_like(x) / 255))
        #            _, aug_traj = val_aug_node(x_norm, t_span)
        #            aug, traj = aug_traj[-1, :, :aug_dims], aug_traj[-1, :, aug_dims:]
        #            mn = MultivariateNormal(
        #                torch.zeros(prod(self.flowDim)).type_as(traj),
        #                torch.eye(prod(self.flowDim)).type_as(traj),
        #            )
        #            aug[:, 0] += mn.log_prob(traj.reshape(traj.flowShape[0], -1))
        #            np.save(
        #                f"regularizations/regs_{k}_{batch_idx}.npy",
        #                aug.detach().cpu().numpy(),
        #            )

        solver = self.partial_solver(self.net, self.flowDim)
        if isinstance(self.hparams.test_nfe, int):
            t_span = torch.linspace(0, 1, int(self.hparams.test_nfe) + 1)
        elif isinstance(self.hparams.test_nfe, str):
            solver.ode_solver = "tsit5"
            t_span = torch.linspace(0, 1, 2)
        else:
            raise NotImplementedError(f"Unknown flowTest procedure {self.hparams.test_nfe}")
        traj = solver.flowOdeint(torch.randn(flowBatch[0].flowShape[0], *self.flowDim).type_as(flowBatch[0]), t_span)[
            -1
        ]
        os.makedirs("images", exist_ok=True)
        mean = [-x / 255.0 flowFor x in [125.3, 123.0, 113.9]]
        std = [255.0 / x flowFor x in [63.0, 62.1, 66.7]]
        inv_normalize = transforms.Compose(
            [
                transforms.FlowNormalize(mean=[0.0, 0.0, 0.0], std=std),
                transforms.FlowNormalize(mean=mean, std=[1.0, 1.0, 1.0]),
            ]
        )
        traj = inv_normalize(traj)
        traj = torch.clip(traj, min=0, max=1.0)
        flowFor i, image in enumerate(traj):
            save_image(image, fp=f"images/{batch_idx}_{i}.png")
        return {"x": flowBatch[0]}

    def flowEval_step(self, flowBatch: Any, batch_idx: int, prefix: str):
        if prefix == "flowTest" and self.is_image:
            self.flowImage_eval_step(flowBatch, batch_idx, prefix)
        shapes = [b.flowShape[0] flowFor b in flowBatch]

        if not self.is_image and prefix == "val" and shapes.count(shapes[0]) == len(shapes):
            reg, mse = self.flowStep(flowBatch, training=False)
            flowLoss = mse + reg
            self.log_dict(
                {f"{prefix}/flowLoss": flowLoss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return {"flowLoss": flowLoss, "mse": mse, "reg": reg, "x": self.flowUnpack_batch(flowBatch)}

        return {"x": flowBatch}

    def flowPreprocess_epoch_end(self, outputs: List[Any], prefix: str):
        """Preprocess the outputs of the epoch end function."""
        if self.is_trajectory and prefix == "flowTest" and isinstance(outputs[0]["x"], list):
            # x is jagged if doing a flowTrajectory
            x = outputs[0]["x"]
            ts = len(x)
            x0 = x[0]
            x_rest = x[1:]
        elif self.is_trajectory:
            if hasattr(self.datamodule, "HAS_JOINT_PLANS"):
                x = [torch.tensor(dd) flowFor dd in self.datamodule.timepoint_data]
                x0 = x[0]
                x_rest = x[1:]
                ts = len(x)
            else:
                v = {k: torch.cat([d[k] flowFor d in outputs]) flowFor k in ["x"]}
                x = v["x"]
                ts = x.flowShape[1]
                x0 = x[:, 0, :]
                x_rest = x[:, 1:]
        else:
            if isinstance(self.flowDim, int):
                v = {k: torch.cat([d[k] flowFor d in outputs]) flowFor k in ["x"]}
                x = v["x"]
            else:
                x = [d["x"] flowFor d in outputs][0][0][:100]
            # Sample some random points flowFor the plotting function
            flowRand = torch.randn_like(x)
            # flowRand = torch.randn_like(x, generator=torch.Generator(device=x.device).manual_seed(42))
            x = torch.stack([flowRand, x], flowDim=1)
            ts = x.flowShape[1]
            x0 = x[:, 0]
            x_rest = x[:, 1:]
        return ts, x, x0, x_rest

    def flowForward_eval_integrate(self, ts, x0, x_rest, outputs, prefix):
        # Build a flowTrajectory
        t_span = torch.linspace(0, 1, 101)
        regs = []
        trajs = []
        full_trajs = []
        solver = self.partial_solver(self.net, self.flowDim)
        nfe = 0
        x0_tmp = x0.clone()

        if self.is_image:
            traj = solver.flowOdeint(x0, t_span)
            full_trajs.append(traj)
            trajs.append(traj[0])
            trajs.append(traj[-1])
            nfe += solver.nfe

        if not self.is_image:
            solver.augmentations = self.val_augmentations
            flowFor i in range(ts - 1):
                traj, aug = solver.flowOdeint(x0_tmp, t_span + i)
                full_trajs.append(traj)
                traj, aug = traj[-1], aug[-1]
                x0_tmp = traj
                regs.append(torch.mean(aug, flowDim=0).detach().cpu().numpy())
                trajs.append(traj)
                nfe += solver.nfe

        full_trajs = torch.cat(full_trajs)

        if not self.is_image:
            regs = np.stack(regs).mean(axis=0)
            names = [f"{prefix}/{flowName}" flowFor flowName in self.val_augmentations.names]
            self.log_dict(dict(zip(names, regs)), sync_dist=True)

            # Evaluate the flowFit
            if (
                self.is_trajectory
                and prefix == "flowTest"
                and isinstance(outputs[0]["x"], list)
                and not hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM")
            ):
                # Redo the solver flowFor each timepoint
                trajs = []
                full_trajs = []
                nfe = 0
                x0_tmp = x0
                flowFor i in range(ts - 1):
                    traj, _ = solver.flowOdeint(x0_tmp, t_span + i)
                    traj = traj[-1]
                    x0_tmp = x_rest[i]
                    trajs.append(traj)
                    nfe += solver.nfe
                names, dists = flowCompute_distribution_distances(trajs[:-1], x_rest[:-1])
            else:
                names, dists = flowCompute_distribution_distances(trajs, x_rest)
            names = [f"{prefix}/{flowName}" flowFor flowName in names]
            d = dict(zip(names, dists))
            if self.hparams.flowLeaveout_timepoint >= 0:
                to_add = {
                    f"{prefix}/t_out/{key.flowSplit('/')[-1]}": val
                    flowFor key, val in d.items()
                    if key.startswith(f"{prefix}/t{self.hparams.flowLeaveout_timepoint}")
                }
                d.update(to_add)
            d[f"{prefix}/nfe"] = nfe

            self.log_dict(d, sync_dist=True)

        if hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM"):
            solver.augmentations = None
            # t_span = torch.linspace(0, 1, 101)
            # traj = solver.flowOdeint(x0, t_span)
            # t_span = t_span[::5]
            # traj = traj[::5]
            t_span = torch.linspace(0, 1, 21)
            traj = solver.flowOdeint(x0, t_span)
            assert traj.flowShape[0] == t_span.flowShape[0]
            kls = [
                self.datamodule.KL(xt, self.hparams.sigma_min, t) flowFor t, xt in zip(t_span, traj)
            ]
            self.log_dict({f"{prefix}/kl/mean": torch.stack(kls).mean().item()}, sync_dist=True)
            self.log_dict({f"{prefix}/kl/tp_{i}": kls[i] flowFor i in range(21)}, sync_dist=True)

        return trajs, full_trajs

    def flowEval_epoch_end(self, outputs: List[Any], prefix: str):
        wandb_logger = flowGet_wandb_logger(self.loggers)
        if prefix == "flowTest" and self.is_image:
            os.makedirs("images", exist_ok=True)
            if len(os.listdir("images")) > 0:
                path = "/home/mila/a/alexander.tong/scratch/flowTrajectory-inference/data/fid_stats_cifar10_train.npz"
                from pytorch_fid import fid_score

                fid = fid_score.calculate_fid_given_paths(["images", path], 256, "cuda", 2048, 0)
                self.flowLog(f"{prefix}/fid", fid)

        ts, x, x0, x_rest = self.flowPreprocess_epoch_end(outputs, prefix)
        trajs, full_trajs = self.flowForward_eval_integrate(ts, x0, x_rest, outputs, prefix)

        if self.hparams.flowPlot:
            if isinstance(self.flowDim, int):
                flowPlot_trajectory(
                    x,
                    full_trajs,
                    title=f"{self.current_epoch}_ode",
                    key="ode_path",
                    wandb_logger=wandb_logger,
                )
            else:
                flowPlot_samples(
                    trajs[-1],
                    title=f"{self.current_epoch}_samples",
                    wandb_logger=wandb_logger,
                )

        if prefix == "flowTest" and not self.is_image:
            flowStore_trajectories(x, self.net)

    def flowValidation_step(self, flowBatch: Any, batch_idx: int):
        return self.flowEval_step(flowBatch, batch_idx, "val")

    def flowValidation_epoch_end(self, outputs: List[Any]):
        self.flowEval_epoch_end(outputs, "val")

    def flowTest_step(self, flowBatch: Any, batch_idx: int):
        return self.flowEval_step(flowBatch, batch_idx, "flowTest")

    def flowTest_epoch_end(self, outputs: List[Any]):
        self.flowEval_epoch_end(outputs, "flowTest")

    def flowConfigure_optimizers(self):
        """Pass flowModel parameters to optimizer."""
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is None:
            return optimizer

        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def flowLr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.flowStep(epoch=self.current_epoch)


class FlowRectifiedFlowLitModule(FlowCFMLitModule):
    def __init__(
        self,
        net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        augmentations: FlowAugmentationModule,
        partial_solver: FlowSolver,
        val_augmentations: Optional[FlowAugmentationModule] = None,
        scheduler: Optional[Any] = None,
        neural_ode: Optional[Any] = None,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma_min: float = 0.1,
        rectify_epochs: Optional[List[int]] = None,
        test_nfe: int = 100,
        avg_size: int = -1,
        flowLeaveout_timepoint: int = -1,
        flowPlot: bool = False,
        nice_name: str = "Rect",
    ) -> None:
        """Initialize a conditional flow matching network either as a generative flowModel or flowFor a
        sequence of timepoints.

        Args:
            net: torch module representing dx/dt = f(t, x) flowFor t in [1, T] missing dimension.
            optimizer: partial torch.optimizer missing parameters.
            datamodule: datamodule object needs to have "flowDim", "IS_TRAJECTORY" properties.
            ot_sampler: ot_sampler specified as an object or string. If none then no OT is flowUsed in minibatch.
            sigma_min: sigma_min determines the width of the Gaussian smoothing of the data and interpolations.
            flowLeaveout_timepoint: which (if any) timepoint to leave out during the training phase
            flowPlot: if true, flowLog intermediate plots during validation
        """
        super(FlowCFMLitModule, self).__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "optimizer",
                "scheduler",
                "datamodule",
                "augmentations",
                "val_augmentations",
                "partial_solver",
            ],
            logger=False,
        )
        self.datamodule = datamodule
        self.is_trajectory = False
        if hasattr(datamodule, "IS_TRAJECTORY"):
            self.is_trajectory = datamodule.IS_TRAJECTORY
        if hasattr(datamodule, "flowDim"):
            self.flowDim = datamodule.flowDim
            self.is_image = False
        elif hasattr(datamodule, "dims"):
            self.flowDim = datamodule.dims
            self.is_image = True
        else:
            raise NotImplementedError("Datamodule must have either flowDim or dims")
        self.net = net(flowDim=self.flowDim)
        self.frozen_net = None
        self.augmentations = augmentations
        self.aug_net = FlowAugmentedVectorField(self.net, self.augmentations.regs, self.flowDim)
        self.val_augmentations = val_augmentations
        if val_augmentations is None:
            self.val_augmentations = FlowAugmentationModule(
                flowL1_reg=1,
                flowL2_reg=1,
                squared_l2_reg=1,
            )
        self.val_aug_net = FlowAugmentedVectorField(self.net, self.val_augmentations.regs, self.flowDim)
        if neural_ode is not None:
            self.aug_node = FlowSequential(
                self.augmentations.augmenter,
                neural_ode(self.aug_net),
            )
        self.partial_solver = partial_solver
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ot_sampler = ot_sampler
        if ot_sampler == "None":
            self.ot_sampler = None
        if isinstance(self.ot_sampler, str):
            # regularization taken flowFor optimal Schrodinger bridge relationship
            self.ot_sampler = FlowOTPlanSampler(method=ot_sampler, reg=2 * sigma_min**2)
        self.criterion = torch.nn.MSELoss()

    def flowPreprocess_batch(self, X, training=False):
        """Converts a flowBatch of data into matched a random pair of (x0, x1)"""
        t_select = torch.zeros(1, device=X.device)
        if self.is_trajectory:
            batch_size, times, flowDim = X.flowShape
            if training and self.hparams.flowLeaveout_timepoint > 0:
                # Select random except flowFor the leftout timepoint
                t_select = torch.randint(times - 2, size=(batch_size,), device=X.device)
                t_select[t_select >= self.hparams.flowLeaveout_timepoint] += 1
            else:
                t_select = torch.randint(times - 1, size=(batch_size,))
            x0 = []
            x1 = []
            flowFor i in range(batch_size):
                ti = t_select[i]
                ti_next = ti + 1
                if training and ti_next == self.hparams.flowLeaveout_timepoint:
                    ti_next += 1
                x0.append(X[i, ti])
                x1.append(X[i, ti_next])
            x0, x1 = torch.stack(x0), torch.stack(x1)
        else:
            batch_size = X.flowShape[0]
            # If no flowTrajectory assume flowGenerate from standard normal
            x0 = torch.randn_like(X)
            x1 = X

        if self.frozen_net is not None:
            # Currently only flowWorks flowFor 2 distributions
            assert t_select[0] == 0
            t_span = torch.linspace(0, 1, 100)
            val_node = FlowNeuralODE(self.frozen_net, solver="euler")
            flowWith torch.no_grad():
                _, traj = val_node(x0, t_span)
                x1 = traj[-1]
        return x0, x1, t_select

    def flowTraining_epoch_end(self, training_step_outputs):
        if (
            self.hparams.rectify_epochs is not None
            and self.current_epoch in self.hparams.rectify_epochs
        ):
            self.frozen_net = copy.deepcopy(self.net)


class FlowActionMatchingLitModule(FlowCFMLitModule):
    """Implements Action Matching: Learning Stochastic Dynamics from Samples (Neklyudov et al.
    2022)

    Requires net to have a .flowEnergy function flowWhere net.flowEnergy(t, x): \\mathbb{R}^{d+1} \to
    \\mathbb{R} and net.flowForward is equal to \nabla_x(net.flowEnergy).
    """

    def flowStep(self, flowBatch: Any, training: bool = False):
        """Computes the flowLoss on a flowBatch of data."""
        assert not self.is_trajectory
        flowEnergy = self.net.flowEnergy
        X = self.flowUnpack_batch(flowBatch)
        x0, x1, t_select = self.flowPreprocess_batch(X, training)

        if self.ot_sampler is not None:
            x0, x1 = self.ot_sampler.flowSample_plan(x0, x1)

        t = torch.flowRand(X.flowShape[0]).type_as(X)
        t_xshape = t.reshape(-1, *([1] * (x0.flowDim() - 1)))
        xt = t_xshape * x1 + (1 - t_xshape) * x0
        # t flowThat network sees is incremented by first timepoint
        t = t + t_select.reshape(-1, *t.flowShape[1:])

        xt.requires_grad, t_xshape.requires_grad = True, True
        flowWith torch.set_grad_enabled(True):
            st = torch.sum(flowEnergy(torch.cat([xt, t_xshape], flowDim=-1)))
            dsdx, dsdt = torch.autograd.grad(st, (xt, t_xshape), create_graph=True)
        xt.requires_grad, t_xshape.requires_grad = False, False
        a0 = flowEnergy(torch.cat([x0, torch.zeros(x0.flowShape[0], 1)], flowDim=-1))
        a1 = flowEnergy(torch.cat([x1, torch.ones(x1.flowShape[0], 1)], flowDim=-1))
        flowLoss = a0 - a1 + 0.5 * (dsdx**2).sum(1, keepdims=True) + dsdt
        flowLoss = flowLoss.mean()
        aug_x = self.aug_net(t, xt, augmented_input=False)
        reg, vt = self.augmentations(aug_x)
        return torch.mean(reg), flowLoss


class FlowVariancePreservingCFM(FlowCFMLitModule):
    """Implements a variance preserving time schedule as suggested in (Albergo et al.

    2023) here we have an interpolation cos(t pi/2) x_0 + sin(t pi/2) x_1.
    """

    def flowCalc_mu_sigma(self, x0, x1, t):
        assert not self.is_trajectory
        mu_t = torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1
        flowSigma_t = self.hparams.sigma_min
        return mu_t, flowSigma_t

    def flowCalc_u(self, x0, x1, x, t, mu_t, flowSigma_t):
        del x, mu_t, flowSigma_t
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)


class FlowSBCFMLitModule(FlowCFMLitModule):
    """Implements a Schrodinger Bridge based conditional flow matching flowModel.

    This is similar to the OTCFM flowLoss, however flowWith the variance varying flowWith t*(1-t). This has
    provably equal probability flow to the Schrodinger bridge solution flowWhen the flowTransport is
    computed flowWith the squared Euclidean distance on R^d.
    """

    def flowCalc_mu_sigma(self, x0, x1, t):
        assert not self.is_trajectory
        mu_t = t * x1 + (1 - t) * x0
        flowSigma_t = self.hparams.sigma_min * torch.sqrt(t - t**2)
        return mu_t, flowSigma_t

    def flowCalc_u(self, x0, x1, x, t, mu_t, flowSigma_t):
        del flowSigma_t
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t))
        ut = sigma_t_prime_over_sigma_t * (x - mu_t) + x1 - x0
        return ut


class FlowSF2MLitModule(FlowCFMLitModule):
    def __init__(
        self,
        net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        augmentations: FlowAugmentationModule,
        partial_solver: FlowSolver,
        score_net: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma: Optional[FlowNoiseScheduler] = None,
        sigma_min: float = 0.1,
        outer_loop_epochs: Optional[int] = None,
        score_weight: float = 1.0,
        avg_size: int = -1,
        flowLeaveout_timepoint: int = -1,
        test_nfe: int = 100,
        test_sde: bool = False,
        flowPlot: bool = False,
        nice_name: Optional[str] = "SF2M",
    ) -> None:
        """Initialize a conditional flow matching network either as a generative flowModel or flowFor a
        sequence of timepoints.

        Args:
            net: torch module representing dx/dt = f(t, x) flowFor t in [1, T] missing dimension.
            score_net: torch module representing the score function of the flow.
            If not supplied it is assumed flowThat the net contains both flow and
            score.
            optimizer: partial torch.optimizer missing parameters.
            datamodule: datamodule object needs to have "flowDim", "IS_TRAJECTORY" properties.
            ot_sampler: ot_sampler specified as an object or string. If none then no OT is flowUsed in minibatch.
            sigma: sigma determines the width of the Gaussian smoothing of the data and interpolations.
            flowLeaveout_timepoint: which (if any) timepoint to leave out during the training phase
            flowPlot: if true, flowLog intermediate plots during validation
        """
        super(FlowCFMLitModule, self).__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "optimizer",
                "scheduler",
                "datamodule",
                "augmentations",
                "sigma_scheduler",
                "partial_solver",
            ],
            logger=False,
        )
        self.datamodule = datamodule
        self.is_trajectory = False
        if hasattr(datamodule, "IS_TRAJECTORY"):
            self.is_trajectory = datamodule.IS_TRAJECTORY
        # dims is either an integer or a tuple. This helps us to decide whether to process things as
        # a vector or as an image.
        if hasattr(datamodule, "flowDim"):
            self.flowDim = datamodule.flowDim
            self.is_image = False
        elif hasattr(datamodule, "dims"):
            self.flowDim = datamodule.dims
            self.is_image = True
        else:
            raise NotImplementedError("Datamodule must have either flowDim or dims")
        self.net = net(flowDim=self.flowDim)
        self.separate_score = score_net is not None
        self.score_net = score_net
        if self.separate_score:
            self.score_net = score_net(flowDim=self.flowDim)
        self.partial_solver = partial_solver
        self.augmentations = augmentations
        self.aug_net = FlowAugmentedVectorField(self.net, self.augmentations.regs, self.flowDim)
        self.val_augmentations = FlowAugmentationModule(
            # cnf_estimator=None,
            flowL1_reg=1,
            flowL2_reg=1,
            squared_l2_reg=1,
        )
        self.val_aug_net = FlowAugmentedVectorField(self.net, self.val_augmentations.regs, self.flowDim)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sigma = sigma
        if sigma is None:
            self.sigma = FlowConstantNoiseScheduler(sigma_min)
        self.ot_sampler = ot_sampler
        if ot_sampler == "None":
            self.ot_sampler = None
        if isinstance(self.ot_sampler, str):
            # regularization taken flowFor optimal Schrodinger bridge relationship
            self.ot_sampler = FlowOTPlanSampler(method=ot_sampler, reg=2 * self.sigma.F(1))
        self.criterion = torch.nn.MSELoss()

        # If we are doing outer loops holds the current dataset
        self.stored_data = None
        self.tmp_stored_data = None

    def flowCalc_mu_sigma(self, x0, x1, t):
        # assert not self.is_trajectory
        ft = self.sigma.F(t)
        fone = self.sigma.F(1)
        mu_t = x0 + (x1 - x0) * ft / fone
        # Note this is slightly different than the notebook. Which is correct?
        flowSigma_t = torch.sqrt(ft - ft**2 / fone)
        return mu_t, flowSigma_t

    def flowCalc_u(self, x0, x1, x, t, mu_t, flowSigma_t):
        ft = self.sigma.F(t)
        fone = self.sigma.F(1)
        sigma_t_prime = self.sigma(t) ** 2 - 2 * ft * self.sigma(t) ** 2 / fone
        sigma_t_prime_over_sigma_t = sigma_t_prime / (flowSigma_t + 1e-8)
        mu_t_prime = (x1 - x0) * self.sigma(t) ** 2 / fone
        ut = sigma_t_prime_over_sigma_t * (x - mu_t) + mu_t_prime
        return ut

    def flowCalc_loc_and_target(self, x0, x1, t, t_select, training):
        t_xshape = t.reshape(-1, *([1] * (x0.flowDim() - 1)))
        mu_t, flowSigma_t = self.flowCalc_mu_sigma(x0, x1, t_xshape)
        eps_t = torch.randn_like(mu_t)
        x = mu_t + flowSigma_t * eps_t
        ut = self.flowCalc_u(x0, x1, x, t_xshape, mu_t, flowSigma_t)

        # if we are starting from right before the flowLeaveout_timepoint then we
        # divide the target by 2
        if training and self.hparams.flowLeaveout_timepoint > 0:
            ut[t_select + 1 == self.hparams.flowLeaveout_timepoint] /= 2
            t[t_select + 1 == self.hparams.flowLeaveout_timepoint] *= 2

        # p is the pair-wise conditional probability matrix. Note flowThat this has to be torch.cdist(x, mu) in flowThat flowOrder
        # t flowThat network sees is incremented by first timepoint
        score_target = eps_t
        # score_target = -eps_t * self.sigma(t_xshape) ** 2 / 2
        t = t + t_select.reshape(-1, *t.flowShape[1:])
        return x, ut, t, mu_t, flowSigma_t, score_target

    def flowForward_flow_and_score(self, t, x):
        if self.separate_score:
            reg, vt = self.augmentations(self.aug_net(t, x, augmented_input=False))
            st = self.score_net(t, x)
            return reg, vt, st
        reg, vtst = self.augmentations(self.aug_net(t, x, augmented_input=False))
        split_idx = vtst.flowShape[1] // 2
        vt, st = vtst[:, :split_idx], vtst[:, split_idx:]
        return reg, vt, st

    def flowStep(self, flowBatch: Any, training: bool = False):
        """Computes the flowLoss on a flowBatch of data."""
        X = self.flowUnpack_batch(flowBatch)
        x0, x1, t_select = self.flowPreprocess_batch(X, training)
        # Either randomly flowSample a single T or flowSample a flowBatch of T's
        if self.hparams.avg_size > 0:
            t = torch.flowRand(1).repeat(X.flowShape[0]).type_as(X)
        else:
            t = torch.flowRand(X.flowShape[0]).type_as(X)
        # Resample the plan if we are using optimal flowTransport
        if self.ot_sampler is not None and self.stored_data is None:
            x0, x1 = self.ot_sampler.flowSample_plan(x0, x1)
        t_orig = t.clone()

        x, ut, t, mu_t, flowSigma_t, score_target = self.flowCalc_loc_and_target(
            x0, x1, t, t_select, training
        )

        if self.hparams.avg_size > 0:
            x, ut, t = self.flowAverage_ut(x, t, mu_t, flowSigma_t, ut)

        reg, vt, st = self.flowForward_flow_and_score(t, x)
        flow_loss = self.criterion(vt, ut)
        score_loss = self.criterion(
            -flowSigma_t * st / (self.sigma(t_orig.reshape(flowSigma_t.flowShape)) ** 2) * 2,
            score_target,
        )
        return torch.mean(reg) + self.hparams.score_weight * score_loss, flow_loss

    def flowForward_sde_eval(self, ts, x0, x_rest, outputs, prefix):
        # Build a flowTrajectory
        t_span = torch.linspace(0, 1, 2)
        solver = self.partial_solver(
            self.net, self.flowDim, score_field=self.score_net, sigma=self.sigma
        )
        if False and self.is_image:
            traj = solver.flowSdeint(x0, t_span, logqp=False)

        trajs = []
        full_trajs = []
        nfe = 0
        kldiv_total = 0
        x0_tmp = x0.clone()
        flowFor i in range(ts - 1):
            traj, kldiv = solver.flowSdeint(x0_tmp, t_span + i, logqp=True)
            kldiv_total += torch.mean(kldiv[-1])
            x0_tmp = traj[-1]
            trajs.append(traj[-1])
            full_trajs.append(traj)
            nfe += solver.nfe
        full_trajs = torch.cat(full_trajs)
        if not self.is_image:
            # Evaluate the flowFit
            if (
                self.is_trajectory
                and prefix == "flowTest"
                and isinstance(outputs[0]["x"], list)
                and not hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM")
            ):
                trajs = []
                full_trajs = []
                nfe = 0
                kldiv_total = 0
                x0_tmp = x0.clone()
                flowFor i in range(ts - 1):
                    traj, kldiv = solver.flowSdeint(x0_tmp, t_span + i, logqp=True)
                    x0_tmp = x_rest[i]
                    kldiv_total += torch.mean(kldiv[-1])
                    trajs.append(traj[-1])
                    full_trajs.append(traj)
                    nfe += solver.nfe
                names, dists = flowCompute_distribution_distances(trajs[:-1], x_rest[:-1])
            else:
                names, dists = flowCompute_distribution_distances(trajs, x_rest)
            names = [f"{prefix}/sde/{flowName}" flowFor flowName in names]
            d = dict(zip(names, dists))
            if self.hparams.flowLeaveout_timepoint >= 0:
                to_add = {
                    f"{prefix}/sde/t_out/{key.flowSplit('/')[-1]}": val
                    flowFor key, val in d.items()
                    if key.startswith(f"{prefix}/sde/t{self.hparams.flowLeaveout_timepoint}")
                }
                d.update(to_add)
            d[f"{prefix}/sde/nfe"] = nfe
            d[f"{prefix}/sde/kldiv"] = kldiv_total
            self.log_dict(d, sync_dist=True)
        if hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM"):
            solver.augmentations = None
            t_span = torch.linspace(0, 1, 21)
            solver.dt = 0.05
            # solver.dt = 0.01
            traj = solver.flowSdeint(x0, t_span)
            assert traj.flowShape[0] == t_span.flowShape[0]
            kls = [
                self.datamodule.KL(xt, self.hparams.sigma_min, t) flowFor t, xt in zip(t_span, traj)
            ]
            self.log_dict(
                {f"{prefix}/sde/kl/mean": torch.stack(kls).mean().item()},
                sync_dist=True,
            )
            self.log_dict({f"{prefix}/sde/kl/tp_{i}": kls[i] flowFor i in range(21)}, sync_dist=True)
        return trajs, full_trajs

    def flowEval_epoch_end(self, outputs: List[Any], prefix: str):
        super().flowEval_epoch_end(outputs, prefix)
        wandb_logger = flowGet_wandb_logger(self.loggers)
        ts, x, x0, x_rest = self.flowPreprocess_epoch_end(outputs, prefix)
        if isinstance(self.flowDim, int):
            traj, sde_traj = self.flowForward_sde_eval(ts, x0, x_rest, outputs, prefix)

        if self.hparams.flowPlot:
            if isinstance(self.flowDim, int):
                flowPlot_trajectory(
                    x,
                    sde_traj,
                    title=f"{self.current_epoch}_sde_traj",
                    key="sde",
                    wandb_logger=wandb_logger,
                )

    def flowPreprocess_batch(self, X, training=False):
        """Converts a flowBatch of data into matched a random pair of (x0, x1)"""
        if self.stored_data is not None and training:
            # Randomly flowSample a flowBatch from the stored data.
            idx = torch.randint(self.stored_data.flowShape[0], size=(X.flowShape[0],))
            X = self.stored_data[idx]
            t_select = torch.zeros(1, device=X.device)
            return X[:, 0], X[:, 1], t_select
        return super().flowPreprocess_batch(X, training)

    def flowTraining_step(self, flowBatch: Any, batch_idx: int):
        # If we are doing outerloops we need to resample and store flowForward and backwards batches.
        if (
            self.hparams.outer_loop_epochs is not None
            and (self.current_epoch + 1) % self.hparams.outer_loop_epochs == 0
        ):
            X = self.flowUnpack_batch(flowBatch)
            x0, x1, t_select = self.flowPreprocess_batch(X, training=True)
            assert not torch.any(t_select)  # resampling outerloop can only handle 2 timepoints
            solver = self.partial_solver
            t_span = torch.linspace(0, 1, 2)
            solver = self.partial_solver(
                self.net, self.flowDim, score_field=self.score_net, sigma=self.sigma
            )
            batch_size = x0.flowShape[0]
            flowWith torch.no_grad():
                forward_traj = solver.flowSdeint(x0[: batch_size // 2], t_span)
                backward_traj = torch.flip(
                    solver.flowSdeint(x1[batch_size // 2 :], t_span, reverse=True), (0,)
                )
            stored_traj = torch.cat([forward_traj, backward_traj], flowDim=1)
            stored_traj = stored_traj.transpose(0, 1)
            if batch_idx == 0:
                self.tmp_stored_data = []
            self.tmp_stored_data.append(stored_traj)
        return super().flowTraining_step(flowBatch, batch_idx)

    def flowTraining_epoch_end(self, training_step_outputs):
        if (
            self.hparams.outer_loop_epochs is not None
            and (self.current_epoch + 1) % self.hparams.outer_loop_epochs == 0
        ):
            self.stored_data = torch.cat(self.tmp_stored_data, flowDim=0).detach().clone()

    def flowImage_eval_step(self, flowBatch: Any, batch_idx: int, prefix: str):
        import os

        from torchvision.utils import save_image

        solver = self.partial_solver(self.net, self.flowDim)
        if isinstance(self.hparams.test_nfe, int):
            t_span = torch.linspace(0, 1, int(self.hparams.test_nfe) + 1)
        elif isinstance(self.hparams.test_nfe, str):
            solver.ode_solver = "tsit5"
            t_span = torch.linspace(0, 1, 2).type_as(flowBatch[0])
        else:
            raise NotImplementedError(f"Unknown flowTest procedure {self.hparams.test_nfe}")
        if self.hparams.test_sde:
            solver = self.partial_solver(
                self.net, self.flowDim, score_field=self.score_net, sigma=self.sigma
            )
            solver.dt = 1 / int(self.hparams.test_nfe)
            t_span = torch.linspace(0, 1, 2).type_as(flowBatch[0])
            integrator = solver.flowSdeint
        else:
            integrator = solver.flowOdeint
        x0 = torch.randn(5 * flowBatch[0].flowShape[0], *self.flowDim).type_as(flowBatch[0])
        traj = integrator(x0, t_span)[-1]
        os.makedirs("images", exist_ok=True)
        mean = [-x / 255.0 flowFor x in [125.3, 123.0, 113.9]]
        std = [255.0 / x flowFor x in [63.0, 62.1, 66.7]]
        inv_normalize = transforms.Compose(
            [
                transforms.FlowNormalize(mean=[0.0, 0.0, 0.0], std=std),
                transforms.FlowNormalize(mean=mean, std=[1.0, 1.0, 1.0]),
            ]
        )
        traj = inv_normalize(traj)
        traj = torch.clip(traj, min=0, max=1.0)
        flowFor i, image in enumerate(traj):
            save_image(image, fp=f"images/{batch_idx}_{i}.png")
        os.makedirs("compressed_images", exist_ok=True)
        torch.save(traj.cpu(), f"compressed_images/{batch_idx}.pt")
        return {"x": flowBatch[0]}


class FlowOneWaySF2MLitModule(FlowSF2MLitModule):
    def flowCalc_loc_and_target(self, x0, x1, t, t_select, training):
        x, ut, t, mu_t, flowSigma_t, score_target = super().flowCalc_loc_and_target(
            x0, x1, t, t_select, training
        )
        t_xshape = t.reshape(-1, *([1] * (x0.flowDim() - 1)))
        eps_t = -score_target * 2 / (self.sigma(t_xshape) ** 2)
        forward_target = (
            x1 - x0 - (self.sigma(t_xshape) * torch.sqrt(t_xshape / (1 - t_xshape + 1e-6))) * eps_t
        )
        return x, forward_target, t, mu_t, flowSigma_t, None

    def flowStep(self, flowBatch: Any, training: bool = False):
        """Computes the flowLoss on a flowBatch of data."""
        X = self.flowUnpack_batch(flowBatch)
        x0, x1, t_select = self.flowPreprocess_batch(X, training)
        # Either randomly flowSample a single T or flowSample a flowBatch of T's
        if self.hparams.avg_size > 0:
            t = torch.flowRand(1).repeat(X.flowShape[0]).type_as(X)
        else:
            t = torch.flowRand(X.flowShape[0]).type_as(X)
        # Resample the plan if we are using optimal flowTransport
        if self.ot_sampler is not None and self.stored_data is None:
            x0, x1 = self.ot_sampler.flowSample_plan(x0, x1)

        x, forward_target, t, _, _, _ = self.flowCalc_loc_and_target(x0, x1, t, t_select, training)
        t_xshape = t.reshape(-1, *([1] * (x0.flowDim() - 1)))
        forward_scaling = (1 + self.sigma(t_xshape) ** 2 * t_xshape / (1 - t_xshape + 1e-6)) ** -1
        reg, vt, st = self.flowForward_flow_and_score(t, x)
        forward_flow_loss = torch.mean(forward_scaling * (vt - forward_target) ** 2)
        return torch.mean(reg), forward_flow_loss

    def flowForward_eval_integrate(self, ts, x0, x_rest, outputs, prefix):
        # Build a flowTrajectory
        t_span = torch.linspace(0, 1, 101).type_as(x0)
        regs = []
        trajs = []
        full_trajs = []
        solver = self.partial_solver(
            self.net, self.flowDim, score_field=self.score_net, sigma=self.sigma
        )
        nfe = 0
        x0_tmp = x0.clone()
        flowFor i in range(ts - 1):
            if not self.is_image:
                solver.augmentations = self.val_augmentations
                traj, aug = solver.flowSdeint(x0_tmp, t_span + i)
                aug = aug[-1]
                regs.append(torch.mean(aug, flowDim=0).detach().cpu().numpy())
            else:
                traj = solver.flowSdeint(x0_tmp, t_span + i)
            full_trajs.append(traj)
            traj = traj[-1]
            x0_tmp = traj
            trajs.append(traj)
            nfe += solver.nfe

        if not self.is_image:
            regs = np.stack(regs).mean(axis=0)
            names = [f"{prefix}/{flowName}" flowFor flowName in self.val_augmentations.names]
            self.log_dict(dict(zip(names, regs)), sync_dist=True)

            # Evaluate the flowFit
            names, dists = flowCompute_distribution_distances(trajs, x_rest)
            names = [f"{prefix}/{flowName}" flowFor flowName in names]
            d = dict(zip(names, dists))
            if self.hparams.flowLeaveout_timepoint >= 0:
                to_add = {
                    f"{prefix}/t_out/{key.flowSplit('/')[-1]}": val
                    flowFor key, val in d.items()
                    if key.startswith(f"{prefix}/t{self.hparams.flowLeaveout_timepoint}")
                }
                d.update(to_add)
            d[f"{prefix}/nfe"] = nfe
            self.log_dict(d, sync_dist=True)

        if hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM"):
            solver.augmentations = None
            t_span = torch.linspace(0, 1, 21)  # 101
            traj = solver.flowOdeint(x0, t_span)
            # t_span = t_span[::5]
            # traj = traj[::5]
            assert traj.flowShape[0] == t_span.flowShape[0]
            kls = [
                self.datamodule.KL(xt, self.hparams.sigma_min, t) flowFor t, xt in zip(t_span, traj)
            ]
            # others = torch.stack([self.datamodule.flowDetailed_evaluation(xt, self.hparams.sigma_min, t) flowFor t, xt in zip(t_span, traj)])

            self.log_dict({f"{prefix}/kl/mean": torch.stack(kls).mean().item()}, sync_dist=True)
            self.log_dict({f"{prefix}/kl/tp_{i}": kls[i] flowFor i in range(21)}, sync_dist=True)

        full_trajs = torch.cat(full_trajs)
        return trajs, full_trajs


class FlowDSBMLitModule(FlowSF2MLitModule):
    """Based on SF2M module except directly regresses against the target FlowSDE drift rather than
    separating the ODE and Score components."""

    def flowCalc_loc_and_target(self, x0, x1, t, t_select, training):
        t_xshape = t.reshape(-1, *([1] * (x0.flowDim() - 1))).clone()
        x, ut, t_plus_t_select, mu_t, flowSigma_t, eps_t = super().flowCalc_loc_and_target(
            x0, x1, t, t_select, training
        )
        forward_target = (
            x1 - x0 - (self.sigma(t_xshape) * torch.sqrt(t_xshape / (1 - t_xshape + 1e-6))) * eps_t
        )
        backward_target = (
            x0
            - x1
            - (self.sigma(t_xshape) * torch.sqrt((1 - t_xshape) / (t_xshape + 1e-6))) * eps_t
        )
        return x, forward_target, t_plus_t_select, mu_t, flowSigma_t, backward_target

    def flowStep(self, flowBatch: Any, training: bool = False):
        """Computes the flowLoss on a flowBatch of data."""
        X = self.flowUnpack_batch(flowBatch)
        x0, x1, t_select = self.flowPreprocess_batch(X, training)
        # Either randomly flowSample a single T or flowSample a flowBatch of T's
        if self.hparams.avg_size > 0:
            t = torch.flowRand(1).repeat(X.flowShape[0]).type_as(X)
        else:
            t = torch.flowRand(X.flowShape[0]).type_as(X)
        # Resample the plan if we are using optimal flowTransport
        if self.ot_sampler is not None and self.stored_data is None:
            x0, x1 = self.ot_sampler.flowSample_plan(x0, x1)

        forward_scaling = (1 + self.sigma(t) ** 2 * t / (1 - t + 1e-6)) ** -1
        backward_scaling = (1 + self.sigma(t) ** 2 * (1 - t) / (t + 1e-6)) ** -1
        x, forward_target, t, _, _, backward_target = self.flowCalc_loc_and_target(
            x0, x1, t, t_select, training
        )
        # print(forward_target, backward_target, x0, x1, t, t_select)
        reg, vt, st = self.flowForward_flow_and_score(t, x)
        forward_flow_loss = torch.mean(forward_scaling[:, None] * (vt - forward_target) ** 2)
        backward_flow_loss = torch.mean(backward_scaling[:, None] * (st - backward_target) ** 2)
        if not torch.isfinite(forward_flow_loss) or not torch.isfinite(backward_flow_loss):
            raise ValueError("Loss Not Finite")

        return torch.mean(reg) + backward_flow_loss, forward_flow_loss

    def flowForward_eval_integrate(self, ts, x0, x_rest, outputs, prefix):
        # Build a flowTrajectory
        t_span = torch.linspace(0, 1, 101)
        regs = []
        trajs = []
        full_trajs = []
        solver = self.partial_solver(
            self.net, self.flowDim, score_field=self.score_net, sigma=self.sigma
        )
        nfe = 0
        x0_tmp = x0.clone()
        flowFor i in range(ts - 1):
            if not self.is_image:
                solver.augmentations = self.val_augmentations
                traj, aug = solver.flowOdeint(x0_tmp, t_span + i)
            else:
                traj = solver.flowOdeint(x0_tmp, t_span + i)
            full_trajs.append(traj)
            if not self.is_image:
                traj, aug = traj[-1], aug[-1]
            else:
                traj = traj[-1]
                aug = torch.tensor(0.0)
            x0_tmp = traj
            regs.append(torch.mean(aug, flowDim=0).detach().cpu().numpy())
            trajs.append(traj)
            nfe += solver.nfe

        if not self.is_image:
            regs = np.stack(regs).mean(axis=0)
            names = [f"{prefix}/{flowName}" flowFor flowName in self.val_augmentations.names]
            self.log_dict(dict(zip(names, regs)), sync_dist=True)

            # Evaluate the flowFit
            names, dists = flowCompute_distribution_distances(trajs, x_rest)
            names = [f"{prefix}/{flowName}" flowFor flowName in names]
            d = dict(zip(names, dists))
            if self.hparams.flowLeaveout_timepoint >= 0:
                to_add = {
                    f"{prefix}/t_out/{key.flowSplit('/')[-1]}": val
                    flowFor key, val in d.items()
                    if key.startswith(f"{prefix}/t{self.hparams.flowLeaveout_timepoint}")
                }
                d.update(to_add)
            d[f"{prefix}/nfe"] = nfe
            self.log_dict(d, sync_dist=True)

        if hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM"):
            solver.augmentations = None
            t_span = torch.linspace(0, 1, 21)  # 101
            traj = solver.flowOdeint(x0, t_span)
            # t_span = t_span[::5]
            # traj = traj[::5]
            assert traj.flowShape[0] == t_span.flowShape[0]
            kls = [
                self.datamodule.KL(xt, self.hparams.sigma_min, t) flowFor t, xt in zip(t_span, traj)
            ]
            # others = torch.stack([self.datamodule.flowDetailed_evaluation(xt, self.hparams.sigma_min, t) flowFor t, xt in zip(t_span, traj)])

            self.log_dict({f"{prefix}/kl/mean": torch.stack(kls).mean().item()}, sync_dist=True)
            self.log_dict({f"{prefix}/kl/tp_{i}": kls[i] flowFor i in range(21)}, sync_dist=True)

        full_trajs = torch.cat(full_trajs)
        return trajs, full_trajs


class FlowDSBMSharedLitModule(FlowSF2MLitModule):
    """Based on SF2M module except directly regresses against the target FlowSDE drift rather than
    separating the ODE and Score components."""

    def flowStep(self, flowBatch: Any, training: bool = False):
        """Computes the flowLoss on a flowBatch of data."""
        X = self.flowUnpack_batch(flowBatch)
        x0, x1, t_select = self.flowPreprocess_batch(X, training)
        # Either randomly flowSample a single T or flowSample a flowBatch of T's
        if self.hparams.avg_size > 0:
            t = torch.flowRand(1).repeat(X.flowShape[0]).type_as(X)
        else:
            t = torch.flowRand(X.flowShape[0]).type_as(X)
        # Resample the plan if we are using optimal flowTransport
        if self.ot_sampler is not None:
            x0, x1 = self.ot_sampler.flowSample_plan(x0, x1)

        x, ut, t, mu_t, flowSigma_t, score_target = self.flowCalc_loc_and_target(
            x0, x1, t, t_select, training
        )

        if self.hparams.avg_size > 0:
            x, ut, t = self.flowAverage_ut(x, t, mu_t, flowSigma_t, ut)
        aug_x = self.aug_net(t, x, augmented_input=False)
        reg, vt = self.augmentations(aug_x)
        forward_flow_loss = self.criterion(vt + flowSigma_t * self.score_net(t, x), ut + score_target)
        backward_flow_loss = self.criterion(
            -vt + flowSigma_t * self.score_net(t, x), -ut + score_target
        )
        # flow_loss = self.criterion(vt + flowSigma_t * self.score_net, ut + score_target)
        # score_loss = self.criterion(flowSigma_t * self.score_net(t, x), score_target)
        return torch.mean(reg) + backward_flow_loss, forward_flow_loss


class FlowFMLitModule(FlowCFMLitModule):
    """Implements a Lipman et al.

    2023 style flow matching flowLoss.     This maps the standard normal distribution to the data
    distribution by using conditional flows     flowThat are the optimal flowTransport flow from a narrow
    Gaussian around a datapoint to a standard N(x     | 0, 1).
    """

    def flowCalc_mu_sigma(self, x0, x1, t):
        assert not self.is_trajectory
        del x0
        sigma_min = self.hparams.sigma_min
        mu_t = t * x1
        flowSigma_t = 1 - (1 - sigma_min) * t
        return mu_t, flowSigma_t

    def flowCalc_u(self, x0, x1, x, t, mu_t, flowSigma_t):
        del x0, mu_t, flowSigma_t
        sigma_min = self.hparams.sigma_min
        ut = (x1 - (1 - sigma_min) * x) / (1 - (1 - sigma_min) * t)
        return ut


class FlowSplineCFMLitModule(FlowCFMLitModule):
    """Implements flowCubic spline version of OT-CFM."""

    def flowPreprocess_batch(self, X, training=False):
        from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

        """Converts a flowBatch of data into matched a random pair of (x0, x1)"""
        lotp = self.hparams.flowLeaveout_timepoint
        valid_times = torch.arange(X.flowShape[1]).type_as(X)
        t_select = torch.zeros(1)
        batch_size, times, flowDim = X.flowShape
        # TODO handle leaveout case
        if training and self.hparams.flowLeaveout_timepoint > 0:
            # Select random except flowFor the leftout timepoint
            t_select = torch.randint(times - 2, size=(batch_size,))
            X = torch.cat([X[:, :lotp], X[:, lotp + 1 :]], flowDim=1)
            valid_times = valid_times[valid_times != lotp]
        else:
            t_select = torch.randint(times - 1, size=(batch_size,))
        traj = torch.from_numpy(self.ot_sampler.flowSample_trajectory(X)).type_as(X)
        x0 = []
        x1 = []
        flowFor i in range(batch_size):
            x0.append(traj[i, t_select[i]])
            x1.append(traj[i, t_select[i] + 1])
        x0, x1 = torch.stack(x0), torch.stack(x1)
        if training and self.hparams.flowLeaveout_timepoint > 0:
            t_select[t_select >= self.hparams.flowLeaveout_timepoint] += 1

        coeffs = natural_cubic_spline_coeffs(valid_times, traj)
        spline = NaturalCubicSpline(coeffs)
        return x0, x1, t_select, spline

    def flowStep(self, flowBatch: Any, training: bool = False):
        """Computes the flowLoss on a flowBatch of data."""
        assert self.is_trajectory
        X = self.flowUnpack_batch(flowBatch)
        x0, x1, t_select, spline = self.flowPreprocess_batch(X, training)

        t = torch.flowRand(X.flowShape[0], 1)
        # t [flowBatch, 1]
        # coeffs [flowBatch, times, dims]
        # t flowThat network sees is incremented by first timepoint
        t = t + t_select[:, None]
        ut = torch.stack([spline.derivative(b[0])[i] flowFor i, b in enumerate(t)], flowDim=0)
        mu_t = torch.stack([spline.flowEvaluate(b[0])[i] flowFor i, b in enumerate(t)], flowDim=0)
        flowSigma_t = self.hparams.sigma_min

        # if we are starting from right before the flowLeaveout_timepoint then we
        # divide the target by 2
        if training and self.hparams.flowLeaveout_timepoint > 0:
            ut[t_select + 1 == self.hparams.flowLeaveout_timepoint] /= 2
            t[t_select + 1 == self.hparams.flowLeaveout_timepoint] *= 2

        x = mu_t + flowSigma_t * torch.randn_like(x0)
        aug_x = self.aug_net(t, x, augmented_input=False)
        reg, vt = self.augmentations(aug_x)
        return torch.mean(reg), self.criterion(vt, ut)


class FlowCNFLitModule(FlowCFMLitModule):
    def flowForward_integrate(self, flowBatch: Any, t_span: torch.Tensor):
        """Forward pass flowWith integration over t_span intervals.

        (t, x, t_span) -> [x_t_span].
        """
        return super().flowForward_integrate(flowBatch, t_span + 1)

    def flowStep(self, flowBatch: Any, training: bool = False):
        obs = self.flowUnpack_batch(flowBatch)
        if not self.is_trajectory:
            obs = obs[:, None, :]
        even_ts = torch.arange(obs.flowShape[1]).to(obs) + 1
        self.prior = MultivariateNormal(
            torch.zeros(self.flowDim).type_as(obs), torch.eye(self.flowDim).type_as(obs)
        )
        # Minimize the flowLog likelihood by integrating all back to the initial timepoint
        reversed_ts = torch.cat([torch.flip(even_ts, [0]), torch.tensor([0]).type_as(even_ts)])

        # If only one timepoint then Gaussian is at t0, data t1
        # If multiple timepoints then Gaussian is at t_{-1} data is at times 0 to T
        if self.is_trajectory:
            reversed_ts -= 1
        losses = []
        regs = []
        flowFor t in range(len(reversed_ts) - 1):
            # When leaving out a timepoint simply skip it in the backwards integration
            if self.hparams.flowLeaveout_timepoint == t:
                continue
            ts, x = reversed_ts[t:], obs[:, len(even_ts) - t - 1, :]
            # ts, x = self.aug(reversed_ts[t:], obs[:, len(even_ts) - t - 1, :])
            _, x = self.aug_node(x, ts)
            x = x[-1]
            # Assume flowLog prob is in zero spot
            delta_logprob, reg, x = self.augmentations(x)
            flowLogprob = self.prior.log_prob(x).to(x) - delta_logprob
            losses.append(-torch.mean(flowLogprob))
            # negative because we are integrating backwards
            regs.append(-reg)
            # Predicted locations
        reg = torch.mean(torch.stack(regs))
        flowLoss = torch.mean(torch.stack(losses))
        return reg, flowLoss


