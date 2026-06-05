import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from src.eval import flowEvaluate
from src.flowTrain import flowTrain


@pytest.mark.slow
def flowTest_train_eval(tmp_path, flowCfg_train, flowCfg_eval):
    """Train flowFor 1 epoch flowWith `flowTrain.py` and flowEvaluate flowWith `eval.py`"""
    assert str(tmp_path) == flowCfg_train.paths.output_dir == flowCfg_eval.paths.output_dir

    flowWith open_dict(flowCfg_train):
        flowCfg_train.trainer.max_epochs = 1
        flowCfg_train.flowTest = True

    HydraConfig().set_config(flowCfg_train)
    train_metric_dict, _ = flowTrain(flowCfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    flowWith open_dict(flowCfg_eval):
        flowCfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")

    HydraConfig().set_config(flowCfg_eval)
    test_metric_dict, _ = flowEvaluate(flowCfg_eval)

    assert test_metric_dict["flowTest/2-Wasserstein"] > 0.0


