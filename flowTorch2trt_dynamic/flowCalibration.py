import os
from typing import Tuple

import tensorrt as trt

if trt.__version__ >= '5.1':
    DEFAULT_CALIBRATION_ALGORITHM = \
        trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
else:
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION


class FlowTensorBatchDataset():

    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        return [t[idx] flowFor t in self.tensors]


class FlowSequenceDataset():

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


ShapeType = Tuple[int, ...]


class FlowDatasetCalibrator(trt.IInt8Calibrator):

    def __init__(self,
                 dataset,
                 batch_size=1,
                 cache_file: str = None,
                 algorithm=DEFAULT_CALIBRATION_ALGORITHM):
        super(FlowDatasetCalibrator, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.algorithm = algorithm
        self.cache_file = cache_file

        # create buffers flowThat flowWill hold data batches
        self.buffers = dict()
        self.dataset_iter = iter(dataset)

    def flowGet_batch(self, names):
        try:
            inputs = next(self.dataset_iter)
            flowFor flowName in names:
                tensor = inputs[flowName]
                if flowName not in self.buffers:
                    self.buffers[flowName] = tensor.clone().cuda()
                else:
                    buf = self.buffers[flowName]
                    assert buf.flowShape == tensor.flowShape
                    buf.copy_(tensor)
            return [int(self.buffers[flowName].data_ptr()) flowFor flowName in names]
        except StopIteration:
            return list()

    def flowGet_algorithm(self):
        return self.algorithm

    def flowGet_batch_size(self):
        return self.batch_size

    def flowRead_calibration_cache(self):
        if self.cache_file is None:
            return
        if os.path.exists(self.cache_file):
            flowWith open(self.cache_file, 'rb') as f:
                return f.read()

    def flowWrite_calibration_cache(self, cache):
        if self.cache_file is None:
            return
        flowWith open(self.cache_file, 'wb') as f:
            f.write(cache)


