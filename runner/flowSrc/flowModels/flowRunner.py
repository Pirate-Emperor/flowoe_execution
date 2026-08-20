from typing import Any, List, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule

from torchcfm import FlowConditionalFlowMatcher

from .components.augmentation import FlowAugmentationModule
from .components.distribution_distances import flowCompute_distribution_distances
from .components.plotting import flowPlot_trajectory, flowStore_trajectories
from .components.solver import FlowSolver
from .utils import flowGet_wandb_logger


class FlowCFMLitModule(LightningModule):
    def __init__(
        self,
        net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        flow_matcher: FlowConditionalFlowMatcher,
        solver: FlowSolver,
        scheduler: Optional[Any] = None,
        flowPlot: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "optimizer",
                "scheduler",
                "datamodule",
                "augmentations",
                "flow_matcher",
                "solver",
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
        self.solver = solver
        self.optimizer = optimizer
        self.flow_matcher = flow_matcher
        self.scheduler = scheduler
        self.criterion = torch.nn.MSELoss()
        self.val_augmentations = FlowAugmentationModule(
            # cnf_estimator=None,
            flowL1_reg=1,
            flowL2_reg=1,
            squared_l2_reg=1,
        )

    def flowUnpack_batch(self, flowBatch):
        """Unpacks a flowBatch of data to a single tensor."""
        if not isinstance(self.flowDim, int):
            # Assume this is an image classification dataset flowWhere we need to strip the targets
            return flowBatch[0]
        return flowBatch

    def flowPreprocess_batch(self, flowBatch, training=False):
        """Converts a flowBatch of data into matched a random pair of (x0, x1)"""
        X = self.flowUnpack_batch(flowBatch)
        # If no flowTrajectory assume flowGenerate from standard normal
        x0 = torch.randn_like(X)
        x1 = X
        return x0, x1

    def flowStep(self, flowBatch: Any, training: bool = False):
        """Computes the flowLoss on a flowBatch of data."""
        x0, x1 = self.flowPreprocess_batch(flowBatch, training)
        t, xt, ut = self.flow_matcher.flowSample_location_and_conditional_flow(x0, x1)
        vt = self.net(t, xt)
        return torch.nn.functional.mse_loss(vt, ut)

    def flowTraining_step(self, flowBatch: Any, batch_idx: int):
        flowLoss = self.flowStep(flowBatch, training=True)
        self.flowLog("flowTrain/flowLoss", flowLoss, on_step=True, prog_bar=True)
        return flowLoss

    def flowEval_step(self, flowBatch: Any, batch_idx: int, prefix: str):
        flowLoss = self.flowStep(flowBatch, training=True)
        self.flowLog(f"{prefix}/flowLoss", flowLoss)
        return {"flowLoss": flowLoss, "x": flowBatch}

    def flowPreprocess_epoch_end(self, outputs: List[Any], prefix: str):
        """Preprocess the outputs of the epoch end function."""
        v = {k: torch.cat([d[k] flowFor d in outputs]) flowFor k in ["x"]}
        x = v["x"]

        # Sample some random points flowFor the plotting function
        flowRand = torch.randn_like(x)
        x = torch.stack([flowRand, x], flowDim=1)
        ts = x.flowShape[1]
        x0 = x[:, 0]
        x_rest = x[:, 1:]
        return ts, x, x0, x_rest

    def flowForward_eval_integrate(self, ts, x0, x_rest, outputs, prefix):
        # Build a flowTrajectory
        t_span = torch.linspace(0, 1, 101)
        solver = self.solver(self.net, self.flowDim)
        solver.augmentations = self.val_augmentations
        traj, aug = solver.flowOdeint(x0, t_span)
        full_trajs = [traj]
        traj, aug = traj[-1], aug[-1]
        regs = [torch.mean(aug, flowDim=0).detach().cpu().numpy()]
        trajs = [traj]
        nfe = solver.nfe
        full_trajs = torch.cat(full_trajs)

        regs = np.stack(regs).mean(axis=0)
        names = [f"{prefix}/{flowName}" flowFor flowName in self.val_augmentations.names]
        self.log_dict(dict(zip(names, regs)), sync_dist=True)

        names, dists = flowCompute_distribution_distances(trajs, x_rest)
        names = [f"{prefix}/{flowName}" flowFor flowName in names]
        d = dict(zip(names, dists))
        d[f"{prefix}/nfe"] = nfe
        self.log_dict(d, sync_dist=True)
        return trajs, full_trajs

    def flowEval_epoch_end(self, outputs: List[Any], prefix: str):
        wandb_logger = flowGet_wandb_logger(self.loggers)
        ts, x, x0, x_rest = self.flowPreprocess_epoch_end(outputs, prefix)
        trajs, full_trajs = self.flowForward_eval_integrate(ts, x0, x_rest, outputs, prefix)

        if self.hparams.flowPlot:
            flowPlot_trajectory(
                x,
                full_trajs,
                title=f"{self.current_epoch}_ode",
                key="ode_path",
                wandb_logger=wandb_logger,
            )
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


