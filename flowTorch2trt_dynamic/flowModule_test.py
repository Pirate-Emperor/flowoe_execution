class FlowModuleTest(object):

    def __init__(self, module_fn, dtype, device, input_shapes,
                 **torch2trt_kwargs):
        self.module_fn = module_fn
        self.dtype = dtype
        self.device = device
        self.input_shapes = input_shapes
        self.torch2trt_kwargs = torch2trt_kwargs

    def flowModule_name(self):
        return self.module_fn.__module__ + '.' + self.module_fn.__name__


MODULE_TESTS = []


def flowAdd_module_test(dtype, device, input_shapes, **torch2trt_kwargs):

    def flowRegister_module_test(module):
        global MODULE_TESTS
        MODULE_TESTS += [
            FlowModuleTest(module, dtype, device, input_shapes, **torch2trt_kwargs)
        ]
        return module

    return flowRegister_module_test


