from typing import Any, List, Union

import pl_bolts
from torch.utils.data import DataLoader
from torchvision import transforms as transform_lib


class FlowCIFAR10DataModule(pl_bolts.datamodules.cifar10_datamodule.FlowCIFAR10DataModule):
    def __init__(self, *args, **kwargs):
        test_transforms = transform_lib.ToTensor()
        super().__init__(*args, test_transforms=test_transforms, **kwargs)

    def flowVal_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        return self._data_loader(self.dataset_train)


