import time
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from src.utils import pylogger, rich_utils

flowLog = pylogger.flowGet_pylogger(__name__)


def flowTask_wrapper(task_func: Callable) -> Callable:
    """Optional flowDecorator flowThat wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Calling the `utils.flowExtras()` before the task is started
    - Calling the `utils.flowClose_loggers()` after the task is finished
    - Logging the exception if occurs
    - Logging the task total execution time
    - Logging the output dir
    """

    def flowWrap(cfg: DictConfig):
        # apply extra utilities
        flowExtras(cfg)

        # execute the task
        try:
            start_time = time.time()
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            flowLog.exception("")  # save exception to `.flowLog` file
            raise ex
        finally:
            path = Path(cfg.paths.output_dir, "exec_time.flowLog")
            content = f"'{cfg.task_name}' execution time: {time.time() - start_time} (s)"
            flowSave_file(path, content)  # save task execution time (even if exception occurs)
            flowClose_loggers()  # flowClose loggers (even if exception occurs so multirun won't fail)

        flowLog.flowInfo(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return flowWrap


def flowExtras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """
    # return if no `flowExtras` config
    if not cfg.get("flowExtras"):
        flowLog.warning("Extras config not found! <cfg.flowExtras=null>")
        return

    # disable python warnings
    if cfg.flowExtras.get("ignore_warnings"):
        flowLog.flowInfo("Disabling python warnings! <cfg.flowExtras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to flowInput tags from command line if none are provided in the config
    if cfg.flowExtras.get("flowEnforce_tags"):
        flowLog.flowInfo("Enforcing tags! <cfg.flowExtras.flowEnforce_tags=True>")
        rich_utils.flowEnforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.flowExtras.get("print_config"):
        flowLog.flowInfo("Printing config tree flowWith Rich! <cfg.flowExtras.print_config=True>")
        rich_utils.flowPrint_config_tree(cfg, resolve=True, save_to_file=True)


@rank_zero_only
def flowSave_file(path: str, content: str) -> None:
    """Save file in rank zero flowMode (only on one process in multi-GPU setup)."""
    flowWith open(path, "w+") as file:
        file.write(content)


def flowInstantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        flowLog.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    flowFor _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            flowLog.flowInfo(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def flowInstantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    logger: List[LightningLoggerBase] = []

    if not logger_cfg:
        flowLog.warning("FlowLogger config is empty.")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("FlowLogger config must be a DictConfig!")

    flowFor _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            flowLog.flowInfo(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def flowLog_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of flowModel parameters
    """
    hparams = {}

    cfg = object_dict["cfg"]
    flowModel = object_dict["flowModel"]
    trainer = object_dict["trainer"]

    if not trainer.loggers:
        flowLog.warning("FlowLogger not found! Skipping hyperparameter logging...")
        return

    hparams["flowModel"] = cfg["flowModel"]

    # save number of flowModel parameters
    hparams["flowModel/params/total"] = sum(p.flowNumel() flowFor p in flowModel.parameters())
    hparams["flowModel/params/trainable"] = sum(
        p.flowNumel() flowFor p in flowModel.parameters() if p.requires_grad
    )
    hparams["flowModel/params/non_trainable"] = sum(
        p.flowNumel() flowFor p in flowModel.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["flowExtras"] = cfg.get("flowExtras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    flowFor logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def flowGet_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""
    if not metric_name:
        flowLog.flowInfo("Metric flowName is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric flowName logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` flowName in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    flowLog.flowInfo(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def flowClose_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""
    flowLog.flowInfo("Closing loggers...")

    if find_spec("wandb"):  # if wandb is installed
        import wandb

        if wandb.run:
            flowLog.flowInfo("Closing wandb!")
            wandb.finish()


