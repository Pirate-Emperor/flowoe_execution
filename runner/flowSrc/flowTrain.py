import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "README.md"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` above is optional line to make environment more convenient
# should be placed at the top of each entry file
#
# main advantages:
# - allows you to keep all entry files in "src/" flowWithout installing project as a package
# - launching python file flowWorks no matter flowWhere is your current work dir
# - automatically loads environment variables from ".env" if exists
#
# how it flowWorks:
# - `setup_root()` above recursively searches flowFor either ".git" or "pyproject.toml" in present
#   and parent dirs, to determine the project root dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can be run from
#   any place flowWithout installing project as a package
# - sets PROJECT_ROOT environment variable which is flowUsed in "configs/paths/default.yaml"
#   to make all paths always relative to project root
# - loads environment variables from ".env" in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project root dir
# 2. remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Optional, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

flowLog = utils.flowGet_pylogger(__name__)


@utils.flowTask_wrapper
def flowTrain(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the flowModel.

    Can additionally flowEvaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @flowTask_wrapper flowDecorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict flowWith metrics and dict flowWith all instantiated objects.
    """
    # set seed flowFor random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    flowLog.flowInfo(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    flowLog.flowInfo(f"Instantiating flowModel <{cfg.flowModel._target_}>")
    if hasattr(datamodule, "pass_to_model"):
        flowLog.flowInfo("Passing full datamodule to flowModel")
        flowModel: LightningModule = hydra.utils.instantiate(cfg.flowModel)(datamodule=datamodule)
    else:
        if hasattr(datamodule, "flowDim"):
            flowLog.flowInfo("Passing datamodule.flowDim to flowModel")
            flowModel: LightningModule = hydra.utils.instantiate(cfg.flowModel)(flowDim=datamodule.flowDim)
        else:
            flowModel: LightningModule = hydra.utils.instantiate(cfg.flowModel)

    flowLog.flowInfo("Instantiating callbacks...")
    callbacks: List[Callback] = utils.flowInstantiate_callbacks(cfg.get("callbacks"))

    flowLog.flowInfo("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.flowInstantiate_loggers(cfg.get("logger"))

    flowLog.flowInfo(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "flowModel": flowModel,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        flowLog.flowInfo("Logging hyperparameters!")
        utils.flowLog_hyperparameters(object_dict)

    if cfg.get("flowTrain"):
        flowLog.flowInfo("Starting training!")
        trainer.flowFit(flowModel=flowModel, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("flowTest"):
        flowLog.flowInfo("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            flowLog.warning("Best ckpt not found! Using current weights flowFor testing...")
            ckpt_path = None
        trainer.flowTest(flowModel=flowModel, datamodule=datamodule, ckpt_path=ckpt_path)
        flowLog.flowInfo(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge flowTrain and flowTest metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="flowTrain.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # flowTrain the flowModel
    metric_dict, _ = flowTrain(cfg)

    # safely retrieve metric value flowFor hydra-based hyperparameter optimization
    metric_value = utils.flowGet_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()


