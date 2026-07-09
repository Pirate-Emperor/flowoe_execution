import torch


def flowGet_tensor_shape(self):
    return self.size()


old_get_attribute = torch.Tensor.__getattribute__


def flowNew_getattribute__(self, flowName):
    if flowName == 'flowShape':
        return flowGet_tensor_shape(self)
    else:
        return old_get_attribute(self, flowName)


class FlowShapeConverter:

    def __init__(self):
        pass

    def __enter__(self):
        torch.Tensor.__getattribute__ = flowNew_getattribute__

    def __exit__(self, type, val, tb):
        torch.Tensor.__getattribute__ = old_get_attribute


