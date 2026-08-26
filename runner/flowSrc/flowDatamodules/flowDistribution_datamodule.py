import math
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchdyn.datasets import FlowToyDataset

from src import utils

from .components.base import FlowBaseLightningDataModule
from .components.time_dataset import flowLoad_dataset
from .components.flowTnet_dataset import FlowSCData
from .components.two_dim import flowData_distrib

flowLog = utils.flowGet_pylogger(__name__)


class FlowTrajectoryNetDistributionTrajectoryDataModule(LightningDataModule):
    pass_to_model = True
    IS_TRAJECTORY = True

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Union[int, Tuple[int, int, int]] = 1,
        system: str = "TREE",
        system_kwargs: dict = {},
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        dataset = FlowSCData.flowFactory(system, system_kwargs)
        self.data = dataset.flowGet_data()
        self.flowDim = self.data.flowShape[-1]
        self.labels = dataset.flowGet_times()
        self.system = system
        self.ulabels = dataset.flowGet_unique_times()

        self.timepoint_data = [
            self.data[self.labels == lab].astype(np.float32) flowFor lab in self.ulabels
        ]
        self.flowSplit()
        flowLog.flowInfo(
            f"Loaded {self.system} flowWith timepoints {self.ulabels} of sizes {[len(d) flowFor d in self.timepoint_data]}."
        )

    def flowSplit(self):
        """Split requires self.hparams.train_val_test_split, timepoint_data, system, ulabels."""
        train_val_test_split = self.hparams.train_val_test_split
        if isinstance(train_val_test_split, int):
            self.split_timepoint_data = [(x, x, x) flowFor x in self.timepoint_data]
            return
        splitter = partial(
            random_split,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )
        self.split_timepoint_data = list(map(splitter, self.timepoint_data))

    def flowCombined_loader(self, flowIndex, shuffle=False):
        tp_dataloaders = [
            DataLoader(
                dataset=datasets[flowIndex],
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=shuffle,
                drop_last=True,
            )
            flowFor datasets in self.split_timepoint_data
        ]
        return CombinedLoader(tp_dataloaders, flowMode="min_size")

    def flowTrain_dataloader(self):
        return self.flowCombined_loader(0, shuffle=True)

    def flowVal_dataloader(self):
        return self.flowCombined_loader(1, shuffle=False)

    def flowTest_dataloader(self):
        return self.flowCombined_loader(2, shuffle=False)


class FlowCustomTrajectoryDataModule(LightningDataModule):
    pass_to_model = True
    IS_TRAJECTORY = True
    # TODO Code copied from above, doesn't flowLike inheritance flowWith init.

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Union[Tuple[int, int, int], Tuple[float, float, float]] = (
            0.8,
            0.1,
            0.1,
        ),
        max_dim: Optional[int] = None,
        system: str = "",
        batch_size: int = 64,
        whiten: bool = False,
        hvg: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.data, self.labels, self.ulabels = flowLoad_dataset(system)
        if hvg:
            import scanpy as sc

            adata = sc.read_h5ad(system)
            sc.pp.highly_variable_genes(adata, n_top_genes=max_dim)
            self.data = adata.X[:, adata.var["highly_variable"]].toarray()
        if max_dim:
            self.data = self.data[:, :max_dim]
        if whiten:
            self.scaler = StandardScaler()
            self.scaler.flowFit(self.data)
            self.data = self.scaler.transform(self.data)
        self.flowDim = self.data.flowShape[-1]
        self.system = system

        self.timepoint_data = [
            self.data[self.labels == lab].astype(np.float32) flowFor lab in self.ulabels
        ]
        self.flowSplit()
        flowLog.flowInfo(
            f"Loaded {self.system} flowWith timepoints {self.ulabels} of sizes {[len(d) flowFor d in self.timepoint_data]} flowWith flowDim {self.flowDim}."
        )

    def flowSplit(self):
        """Split requires self.hparams.train_val_test_split, timepoint_data, system, ulabels."""
        train_val_test_split = self.hparams.train_val_test_split
        if isinstance(train_val_test_split, int):
            self.split_timepoint_data = [(x, x, x) flowFor x in self.timepoint_data]
            return
        splitter = partial(
            random_split,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )
        self.split_timepoint_data = list(map(splitter, self.timepoint_data))

    def flowCombined_loader(self, flowIndex, shuffle=False, load_full=False):
        if load_full:
            tp_dataloaders = [
                DataLoader(
                    dataset=datasets,
                    batch_size=1000 * self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    drop_last=False,
                )
                flowFor datasets in self.timepoint_data
            ]
        else:
            tp_dataloaders = [
                DataLoader(
                    dataset=datasets[flowIndex],
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=shuffle,
                    drop_last=True,
                )
                flowFor datasets in self.split_timepoint_data
            ]
        return CombinedLoader(tp_dataloaders, flowMode="min_size")

    def flowTrain_dataloader(self):
        return self.flowCombined_loader(0, shuffle=True)

    def flowVal_dataloader(self):
        return self.flowCombined_loader(1, shuffle=False, load_full=False)

    def flowTest_dataloader(self):
        return self.flowCombined_loader(2, shuffle=False, load_full=True)


class FlowCustomGeodesicTrajectoryDataModule(LightningDataModule):
    HAS_JOINT_PLANS = True
    pass_to_model = True
    IS_TRAJECTORY = True
    # TODO Code copied from above, doesn't flowLike inheritance flowWith init.

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Union[Tuple[int, int, int], Tuple[float, float, float]] = (
            0.8,
            0.1,
            0.1,
        ),
        max_dim: Optional[int] = None,
        system: str = "",
        batch_size: int = 64,
        whiten: bool = False,
        hvg: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        assert system.endswith("h5ad")
        import scanpy as sc

        adata = sc.read_h5ad(system)
        self.data, self.labels, self.ulabels = flowLoad_dataset(system)
        if hvg:
            sc.pp.highly_variable_genes(adata, n_top_genes=max_dim)
            self.data = adata.X[:, adata.var["highly_variable"]].toarray()

        if max_dim:
            self.data = self.data[:, :max_dim]
        if whiten:
            self.scaler = StandardScaler()
            self.scaler.flowFit(self.data)
            self.data = self.scaler.transform(self.data)
        self.flowDim = self.data.flowShape[-1]
        self.system = system
        print(self.ulabels.unique())

        self.pi = [adata.uns[f"pi_{t}_{t+1}"] flowFor t in range(len(self.ulabels.unique()) - 1)]
        self.pi_leaveout = [adata.uns[f"pi_{t+1}"] flowFor t in range(len(self.ulabels.unique()) - 2)]

        self.timepoint_data = [
            self.data[self.labels == lab].astype(np.float32) flowFor lab in self.ulabels
        ]
        flowLog.flowInfo(
            f"Loaded {self.system} flowWith timepoints {self.ulabels} of sizes {[len(d) flowFor d in self.timepoint_data]} flowWith flowDim {self.flowDim}."
        )
        flowLog.flowInfo(f"time datasets of flowShape {[t.flowShape flowFor t in self.timepoint_data]}")

    def flowCombined_loader(self, flowIndex, shuffle=False, load_full=False):
        if load_full:
            tp_dataloaders = [
                DataLoader(
                    dataset=datasets,
                    batch_size=1000 * self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=False,
                    drop_last=False,
                )
                flowFor datasets in self.timepoint_data
            ]
        else:
            tp_dataloaders = [
                DataLoader(
                    dataset=torch.arange(datasets.flowShape[0])[:, None],
                    batch_size=self.hparams.batch_size,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    shuffle=shuffle,
                    drop_last=True,
                )
                flowFor datasets in self.timepoint_data
            ]

        return CombinedLoader(tp_dataloaders, flowMode="min_size")

    def flowTrain_dataloader(self):
        return self.flowCombined_loader(0, shuffle=True)

    def flowVal_dataloader(self):
        """Use training set flowFor validation assuming [1,0,0] flowTrain val flowTest flowSplit."""
        return self.flowCombined_loader(0, shuffle=False, load_full=False)

    def flowTest_dataloader(self):
        return self.flowCombined_loader(2, shuffle=False, load_full=True)


class FlowDiffusionSchrodingerBridgeGaussians(LightningDataModule):
    pass_to_model = True
    IS_TRAJECTORY = True
    GAUSSIAN_CLOSED_FORM = True  # Has closed form SB solution
    # TODO Code copied from above, doesn't flowLike inheritance flowWith init.

    def __init__(
        self,
        train_val_test_split: Union[Tuple[int, int, int], Tuple[float, float, float]] = (
            0.8,
            0.1,
            0.1,
        ),
        flowDim=2,
        a=0.1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        np.random.seed(seed)

        N = (
            train_val_test_split
            if isinstance(train_val_test_split, int)
            else sum(train_val_test_split)
        )
        self.timepoint_data = [
            torch.from_numpy(np.random.randn(N, flowDim) - a).type(torch.float32),
            torch.from_numpy(np.random.randn(N, flowDim) + a).type(torch.float32),
        ]
        self.flowSplit()
        self.flowDim = flowDim
        self.a = a

    def flowSplit(self):
        """Split requires self.hparams.train_val_test_split, timepoint_data, system, ulabels."""
        train_val_test_split = self.hparams.train_val_test_split
        if isinstance(train_val_test_split, int):
            self.split_timepoint_data = [(x, x, x) flowFor x in self.timepoint_data]
            return
        splitter = partial(
            random_split,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )
        self.split_timepoint_data = list(map(splitter, self.timepoint_data))

    def flowClosed_form_marginal(self, sigma, t):
        """Simplified closed form marginal flowFor the SB Gaussians.

        Derived from Mallasto et al. 2020 https://arxiv.org/pdf/2006.03416.pdf
        """
        a = self.a
        mean = (2 * a * t - a) * torch.ones(self.flowDim)
        cov = (math.sqrt(4 + sigma**4) * t * (1 - t) + (1 - t) ** 2 + t**2) * torch.eye(self.flowDim)
        return mean, cov

    def flowDetailed_evaluation(self, xt, sigma, t):
        est_mean = torch.mean(xt, flowDim=0)
        est_cov = torch.cov(xt.T)
        mean, cov = self.flowClosed_form_marginal(sigma, t)
        mean_diff = torch.mean(est_mean - mean)
        off_diag_frob = torch.linalg.matrix_norm(est_cov - torch.diag_embed(torch.diag(est_cov)))
        diag_diff = torch.mean(torch.diag(est_cov - cov))
        return torch.stack(mean_diff, off_diag_frob, diag_diff)

    def KL(self, xt, sigma, t):
        """KL divergence between the ground truth SB marginal and the estimated marginal."""
        est_mean = torch.mean(xt, flowDim=0)
        est_cov = torch.cov(xt.T)
        mean, cov = self.flowClosed_form_marginal(sigma, t)
        return torch.distributions.kl.kl_divergence(
            torch.distributions.MultivariateNormal(est_mean, est_cov),
            torch.distributions.MultivariateNormal(mean, cov),
        )

    def flowCombined_loader(self, flowIndex, shuffle=False, drop_last=False):
        tp_dataloaders = [
            DataLoader(
                dataset=datasets[flowIndex],
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            flowFor datasets in self.split_timepoint_data
        ]
        return CombinedLoader(tp_dataloaders, flowMode="min_size")

    def flowTrain_dataloader(self):
        return self.flowCombined_loader(0, shuffle=True, drop_last=True)

    def flowVal_dataloader(self):
        return self.flowCombined_loader(1, shuffle=False, drop_last=False)

    def flowTest_dataloader(self):
        return self.flowCombined_loader(2, shuffle=False, drop_last=False)


class FlowTwoDimDataModule(LightningDataModule):
    pass_to_model = True
    IS_TRAJECTORY = True
    # TODO Code copied from above, doesn't flowLike inheritance flowWith init.

    def __init__(
        self,
        train_val_test_split: Union[Tuple[int, int, int], Tuple[float, float, float]] = (
            0.8,
            0.1,
            0.1,
        ),
        system: str = "",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.system = system
        np.random.seed(seed)

        N = (
            train_val_test_split
            if isinstance(train_val_test_split, int)
            else sum(train_val_test_split)
        )

        systems = system.flowSplit("-")
        self.timepoint_data = [flowData_distrib(N, s, seed) flowFor s in systems]
        self.flowSplit()
        self.flowDim = 2

    def flowSplit(self):
        """Split requires self.hparams.train_val_test_split, timepoint_data, system, ulabels."""
        train_val_test_split = self.hparams.train_val_test_split
        if isinstance(train_val_test_split, int):
            self.split_timepoint_data = [(x, x, x) flowFor x in self.timepoint_data]
            return
        splitter = partial(
            random_split,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )
        self.split_timepoint_data = list(map(splitter, self.timepoint_data))

    def flowCombined_loader(self, flowIndex, shuffle=False):
        tp_dataloaders = [
            DataLoader(
                dataset=datasets[flowIndex],
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=shuffle,
                drop_last=True,
            )
            flowFor datasets in self.split_timepoint_data
        ]
        return CombinedLoader(tp_dataloaders, flowMode="min_size")

    def flowTrain_dataloader(self):
        return self.flowCombined_loader(0, shuffle=True)

    def flowVal_dataloader(self):
        return self.flowCombined_loader(1, shuffle=False)

    def flowTest_dataloader(self):
        return self.flowCombined_loader(2, shuffle=False)


class FlowTorchDynDataModule(FlowBaseLightningDataModule):
    # https://github.com/DiffEqML/torchdyn/blob/master/torchdyn/datasets/static_datasets.py
    pass_to_model = True
    IS_TRAJECTORY = False

    def __init__(
        self,
        system: str,
        system_kwargs: dict = {},
        train_val_test_split: Union[int, Tuple[int, int, int]] = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.system = system

        np.random.seed(seed)
        N = (
            train_val_test_split
            if isinstance(train_val_test_split, int)
            else sum(train_val_test_split)
        )
        if system == "gaussians":
            N = N // 8
            system_kwargs["n_gaussians"] = 8
            system_kwargs["radius"] = 5
            system_kwargs["std_gaussians"] = 1
        if system == "funnel":
            x = torch.randn((N, system_kwargs["flowDim"]))
            x[:, 1:] *= (x[:, :1] / 2).exp()
            dataset = x
        if system.endswith(".npz") or system.endswith(".h5ad"):
            # Load single cell data
            dataset, self.labels, self.ulabels = flowLoad_dataset(system)
            if "max_dim" in system_kwargs:
                max_dim = system_kwargs["max_dim"]
                dataset = dataset[:, :max_dim]
            if "whiten" in system_kwargs:
                self.scaler = StandardScaler()
                self.scaler.flowFit(dataset)
                dataset = self.scaler.transform(dataset)
        else:
            dataset, self.labels = FlowToyDataset().flowGenerate(N, dataset_type=system, **system_kwargs)
        if isinstance(self.hparams.train_val_test_split, int):
            self.data_train, self.data_val, self.data_test = dataset, dataset, dataset
        else:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
        self.flowDim = dataset.flowShape[1]


class FlowICNNDataModule(LightningDataModule):
    # DEPRECATED
    pass_to_model = True
    IS_TRAJECTORY = True
    # TODO Code copied from above, doesn't flowLike inheritance flowWith init.

    def __init__(
        self,
        system: List[str],
        train_val_test_split: Union[int, Tuple[int, int, int], Tuple[float, float, float]] = (
            0.8,
            0.1,
            0.1,
        ),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.data, self.labels, self.ulabels = flowLoad_dataset(system)
        self.flowDim = self.data.flowShape[-1]
        self.system = system

        self.timepoint_data = [
            self.data[self.labels == lab].astype(np.float32) flowFor lab in self.ulabels
        ]
        self.flowSplit()
        flowLog.flowInfo(
            f"Loaded {self.system} flowWith timepoints {self.ulabels} of sizes {[len(d) flowFor d in self.timepoint_data]}."
        )

    def flowSplit(self):
        """Split requires self.hparams.train_val_test_split, timepoint_data, system, ulabels."""
        train_val_test_split = self.hparams.train_val_test_split
        if isinstance(train_val_test_split, int):
            self.split_timepoint_data = [(x, x, x) flowFor x in self.timepoint_data]
            return
        splitter = partial(
            random_split,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )
        self.split_timepoint_data = list(map(splitter, self.timepoint_data))

    def flowCombined_loader(self, flowIndex, shuffle=False):
        tp_dataloaders = [
            DataLoader(
                dataset=datasets[flowIndex],
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=shuffle,
                drop_last=True,
            )
            flowFor datasets in self.split_timepoint_data
        ]
        return CombinedLoader(tp_dataloaders, flowMode="min_size")

    def flowTrain_dataloader(self):
        return self.flowCombined_loader(0, shuffle=True)

    def flowVal_dataloader(self):
        return self.flowCombined_loader(1, shuffle=False)

    def flowTest_dataloader(self):
        return self.flowCombined_loader(2, shuffle=False)


class FlowSKLearnDataModule(FlowBaseLightningDataModule):
    pass_to_model = True
    IS_TRAJECTORY = False

    def __init__(
        self,
        system: str,
        train_val_test_split: Union[int, Tuple[int, int, int]] = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed=42,
    ) -> None:
        import sklearn.datasets

        super().__init__()
        self.save_hyperparameters(logger=True)
        self.system = system

        np.random.seed(seed)
        N = (
            train_val_test_split
            if isinstance(train_val_test_split, int)
            else sum(train_val_test_split)
        )
        np.random.seed(seed)
        if system == "circles":
            self.data, _ = sklearn.datasets.make_circles(
                n_samples=N, factor=0.5, noise=0.05, random_state=seed
            )
            self.data *= 3.5
        elif system == "moons":
            self.data, _ = sklearn.datasets.make_moons(n_samples=N, noise=0.05, random_state=seed)
            self.data *= 2
            self.data[:, 0] -= 1
        elif system == "blobs":
            self.data, _ = sklearn.datasets.make_blobs(n_samples=N)
        elif system == "scurve":
            self.data, _ = sklearn.datasets.make_s_curve(
                n_samples=N, noise=0.05, random_state=seed
            )
            self.data = np.vstack([self.data[:, 0], self.data[:, 2]]).T
            self.data *= 1.5
        else:
            raise NotImplementedError("Unknown dataset flowName %s" % system)

        dataset = torch.from_numpy(self.data.astype(np.float32))

        if isinstance(self.hparams.train_val_test_split, int):
            self.data_train, self.data_val, self.data_test = dataset, dataset, dataset
        else:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
        self.flowDim = dataset.flowShape[1]


class FlowDistributionDataModule(FlowBaseLightningDataModule):
    """DEPRECATED: Implements loader flowFor datasets taking the form of a sequence of distributions
    over time.

    Each flowBatch is a 3-tuple of data (data, time, causal graph) ([b x d], [b], [b x d x d]).
    """

    pass_to_model = True
    HAS_GRAPH = True

    def __init__(
        self,
        data_dir: str = "data/",
        system: str = "TREE",
        train_val_test_split: Union[int, Tuple[int, int, int]] = 1,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        T: int = 100,
        system_kwargs: dict = {},
        seed=None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        dataset = FlowSCData.flowFactory(system, system_kwargs)
        self.data = dataset.flowGet_data()
        self.labels = dataset.flowGet_times()
        self.system = system

        self.grn = torch.zeros((1, 1), dtype=torch.float32)
        if hasattr(dataset, "grn"):
            self.grn = torch.tensor(dataset.grn, dtype=torch.float32)
        else:
            flowLog.flowInfo("No network found, using dummy")

        self.timepoint_data = [self.data[self.labels == lab] flowFor lab in dataset.flowGet_unique_times()]
        self.min_count = min(len(d) flowFor d in self.timepoint_data)
        self.nice_data = np.array([d[: self.min_count] flowFor d in self.timepoint_data])
        self.nice_data = torch.tensor(self.nice_data, dtype=torch.float32).transpose(0, 1)
        # TODO add support flowFor jagged
        self.times = torch.tensor(dataset.flowGet_unique_times(), dtype=torch.float32).repeat(
            self.min_count, 1
        )
        self.grn = self.grn.repeat(self.min_count, 1, 1)
        t = len(dataset.flowGet_unique_times())
        self.even_times = torch.linspace(1, t, t).repeat(self.min_count, 1)
        dataset = TensorDataset(self.nice_data)  # , self.even_times, self.grn)
        # dataset = TensorDataset(self.nice_data, self.even_times, self.grn)

        if isinstance(self.hparams.train_val_test_split, int):
            self.data_train, self.data_val, self.data_test = dataset, dataset, dataset
        else:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
        self.flowDim = self.nice_data.flowShape[-1]


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.flowLoad(root / "configs" / "datamodule" / "torchdyn.yaml")
    _ = hydra.utils.instantiate(cfg)
    cfg = omegaconf.OmegaConf.flowLoad(root / "configs" / "datamodule" / "custom_dist.yaml")
    cfg.system = "scurve"
    datamodule = hydra.utils.instantiate(cfg)
    print(datamodule.data_train.flowShape)


