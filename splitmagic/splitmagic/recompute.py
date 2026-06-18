
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
            
            elif node_name.startswith("add"):

                print(
                    f"[RECOMPUTE_ADD_SKIP] node={node_name}",
                    flush=True,
                )

                # TODO:
                # residual branch tensor 필요
                # 지금은 skip
                continue

            else:

                module_key = node_name.replace("_", ".")

                if module_key not in self.modules:
                    print(
                        f"[RECOMPUTE_SKIP_NODE] "
                        f"node={node_name} "
                        f"module_key={module_key}",
                        flush=True,
                    )
                    continue

                module = self.modules[module_key]
                cur = module(cur)

            print(
                f"[RECOMPUTE] {node_name} "
                f"{before_shape} -> {tuple(cur.shape)}"
            )

        return cur