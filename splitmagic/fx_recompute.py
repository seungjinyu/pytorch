import torch
import torch.nn.functional as F

class FXRecomputeEngine:
    def __init__(self, model):
        self.model = model
        self.modules = dict(model.named_modules())

    def run_path(self, start_node, start_tensor, path):
        """
        path example:
          ["relu2", "pool2"]
        start_node:
          "relu2"
        start_tensor:
          output tensor of relu2
        """
        cur = start_tensor

        for node_name in path[1:]:
            module = self.modules[node_name]

            if isinstance(module, torch.nn.MaxPool2d):
                cur = module(cur)
            else:
                cur = module(cur)

        return cur

    def recompute_maxpool_indices(self, pool_node_name, pool_input):
        module = self.modules[pool_node_name]

        if not isinstance(module, torch.nn.MaxPool2d):
            raise TypeError(f"{pool_node_name} is not MaxPool2d")

        out, indices = F.max_pool2d(
            pool_input,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
            return_indices=True,
        )

        return out, indices