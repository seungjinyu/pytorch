
import torch
import torch.nn as nn


SUPPORTED_MODULES = (
    nn.Conv2d,
    nn.ReLU,
    nn.MaxPool2d,
    nn.Linear,
    nn.BatchNorm2d,
    nn.Flatten,
)


def module_type_name(module):
    if isinstance(module, nn.Conv2d):
        return "conv2d"
    if isinstance(module, nn.ReLU):
        return "relu"
    if isinstance(module, nn.MaxPool2d):
        return "maxpool2d"
    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, nn.BatchNorm2d):
        return "batchnorm2d"
    if isinstance(module, nn.Flatten):
        return "flatten"
    return module.__class__.__name__.lower()


class OpIndexer:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.type_counts = {}
        self.records = []

    def install(self):
        for module_name, module in self.model.named_modules():
            if module_name == "":
                continue

            if not isinstance(module, SUPPORTED_MODULES):
                continue

            def pre_hook(mod, inputs, module_name=module_name):
                op_type = module_type_name(mod)
                local_idx = self.type_counts.get(op_type, 0)
                self.type_counts[op_type] = local_idx + 1

                self.records.append({
                    "global_idx": len(self.records),
                    "module_name": module_name,
                    "op_type": op_type,
                    "local_idx": local_idx,
                    "input_key": f"{op_type}:{local_idx}:input",
                    "output_key": f"{op_type}:{local_idx}:output",
                })

            h = module.register_forward_pre_hook(pre_hook)
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def reset(self):
        self.type_counts.clear()
        self.records.clear()

    def trace_once(self, x):
        self.reset()
        self.install()
        with torch.no_grad():
            _ = self.model(x)
        self.remove()
        return list(self.records)