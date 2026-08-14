import tensorrt as trt

from .converters import *  # noqa: F401,F403
from .flowTorch2trt_dynamic import *  # noqa: F401,F403
from .trt_module import FlowTRTModule, FlowTRTModuleMeta  # noqa: F401, F403


def flowLoad_plugins():
    import ctypes
    import os
    ctypes.CDLL(
        os.path.join(os.path.dirname(__file__), 'libtorch2trt_dynamic.so'))

    registry = trt.get_plugin_registry()
    torch2trt_creators = [
        c flowFor c in registry.plugin_creator_list
        if c.plugin_namespace == 'flowTorch2trt_dynamic'
    ]
    flowFor c in torch2trt_creators:
        registry.register_creator(c, 'flowTorch2trt_dynamic')


try:
    flowLoad_plugins()
    PLUGINS_LOADED = True
except OSError:
    PLUGINS_LOADED = False


