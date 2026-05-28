import torch


class FXRecomputeEngine:
    def __init__(self, model):
        self.model = model
        self.modules = dict(model.named_modules())

    def recompute_path(
        self,
        start_tensor,
        path,
    ):
        """
        path example:
            ["pool1", "conv2", "relu2"]

        start_tensor:
            tensor corresponding to pool1
        """

        cur = start_tensor

        print(f"[RECOMPUTE] start path={' -> '.join(path)}")

        for node_name in path[1:]:

            before_shape = tuple(cur.shape)
            if node_name == "flatten":
                cur = cur.flatten(1)

            elif node_name == "view":
                raise NotImplementedError("view recompute needs shape info")

            elif node_name == "reshape":
                raise NotImplementedError("reshape recompute needs shape info")

            else:
                module = self.modules[node_name]
                cur = module(cur)

            print(
                f"[RECOMPUTE] {node_name} "
                f"{before_shape} -> {tuple(cur.shape)}"
            )

        return cur