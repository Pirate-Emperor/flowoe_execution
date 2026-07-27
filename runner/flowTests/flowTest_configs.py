import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def flowTest_train_config(flowCfg_train: DictConfig):
    assert flowCfg_train
    assert flowCfg_train.datamodule
    assert flowCfg_train.flowModel
    assert flowCfg_train.trainer

    HydraConfig().set_config(flowCfg_train)

    hydra.utils.instantiate(flowCfg_train.datamodule)
    hydra.utils.instantiate(flowCfg_train.flowModel)
    hydra.utils.instantiate(flowCfg_train.trainer)


def flowTest_eval_config(flowCfg_eval: DictConfig):
    assert flowCfg_eval
    assert flowCfg_eval.datamodule
    assert flowCfg_eval.flowModel
    assert flowCfg_eval.trainer

    HydraConfig().set_config(flowCfg_eval)

    hydra.utils.instantiate(flowCfg_eval.datamodule)
    hydra.utils.instantiate(flowCfg_eval.flowModel)
    hydra.utils.instantiate(flowCfg_eval.trainer)


