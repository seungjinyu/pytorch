import time 
import torch


class FXRecomputeEngine:
    def __init__(self, model, gm=None, node_values=None):
        self.model = model
        self.modules = dict(model.named_modules())

        self.gm = gm 
        self.node_values = node_values or {}

        self.fx_nodes = {}

        if gm is not None :
            self.fx_nodes = {
                n.name: n
                for n in gm.graph.nodes
            }
    def _value_for_node(self, node_name):
        if node_name in self.node_values:
            return self.node_values[node_name]

        raise RuntimeError(f"[RECOMPUTE] missing value for node={node_name}")

    def _compute_node_from_start(self, target_node_name):
        """
        현재 recompute start node에서 target_node_name까지 path를 찾아 재계산.
        """
        if target_node_name in self.node_values:
            return self.node_values[target_node_name]

        if self.current_start is None:
            raise RuntimeError(
                f"[RECOMPUTE] current_start is None; cannot compute {target_node_name}"
            )

        path = self._find_path(self.current_start, target_node_name)

        if path is None:
            raise RuntimeError(
                f"[RECOMPUTE] no path from {self.current_start} to {target_node_name}"
            )

        return self.recompute_path(
            start_tensor=self.node_values[self.current_start],
            path=path,
        )
    
    def _find_path(self, start, target):
        if self.gm is None:
            return None

        graph = {
            n.name: [
                user.name
                for user in n.users
            ]
            for n in self.gm.graph.nodes
        }

        q = [(start, [start])]
        seen = {start}

        while q:
            cur, path = q.pop(0)

            if cur == target:
                return path

            for nxt in graph.get(cur, []):
                if nxt in seen:
                    continue

                seen.add(nxt)
                q.append((nxt, path + [nxt]))

        return None
    def _compute_add(self, add_node_name, cur):
        if add_node_name not in self.fx_nodes:
            raise RuntimeError(f"[RECOMPUTE] missing fx add node={add_node_name}")

        node = self.fx_nodes[add_node_name]
        args = list(node.args)

        if len(args) != 2:
            raise RuntimeError(
                f"[RECOMPUTE] add node expects 2 args: {add_node_name}, args={args}"
            )

        lhs_node = args[0]
        rhs_node = args[1]

        lhs_name = lhs_node.name
        rhs_name = rhs_node.name

        if lhs_name in self.node_values:
            lhs = self.node_values[lhs_name]
        else:
            # lhs = self._compute_node_from_start(lhs_name)
            lhs = self._compute_node_from_any_available(lhs_name)

        if rhs_name in self.node_values:
            rhs = self.node_values[rhs_name]
        else:
            # rhs = self._compute_node_from_start(rhs_name)
            rhs = self._compute_node_from_any_available(rhs_name)

        return lhs + rhs
    
    def recompute_path(
        self,
        start_tensor,
        path,
    ):

        cur = start_tensor
        self.current_start = path[0]
        self.node_values[path[0]] = start_tensor

        # print(f"[RECOMPUTE] start path={' -> '.join(path)}")
        op_profiles = []

        for node_name in path[1:]:

            # # if the value for this node is already cached, use it
            # if node_name in self.node_values:

            #     before_shape = tuple(cur.shape)

            #     cur = self.node_values[node_name]

            #     op_profiles.append(
            #         (
            #             node_name + "[cache]",
            #             before_shape,
            #             tuple(cur.shape),
            #             0.0,
            #         )
            #     )

            #     continue

            op_t0 = time.perf_counter()

            before_shape = tuple(cur.shape)

            if node_name == "flatten":
                cur = cur.flatten(1)

            elif node_name == "view":
                raise NotImplementedError("view recompute needs shape info")

            elif node_name == "reshape":
                raise NotImplementedError("reshape recompute needs shape info")
            
            elif node_name.startswith("add"):
                
                before_shape = tuple(cur.shape)

                cur = self._compute_add(node_name, cur)

                self.node_values[node_name] = cur

                print(
                    f"[RECOMPUTE] {node_name} "
                    f"{before_shape} -> {tuple(cur.shape)}",
                    flush=True,
                )
            elif "relu" in node_name and node_name not in self.modules:
                before_shape = tuple(cur.shape)
                cur = torch.relu(cur)
                self.node_values[node_name] = cur

                print(
                    f"[RECOMPUTE] {node_name} "
                    f"{before_shape} -> {tuple(cur.shape)}",
                    flush=True,
                )
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
                else:

                    module = self.modules[module_key]
                    cur = module(cur)
                    self.node_values[node_name] = cur

            op_t1 = time.perf_counter()
            op_profiles.append(
                (
                    node_name,
                    before_shape,
                    tuple(cur.shape),
                    (op_t1 - op_t0) * 1000,
                )
            )

        total_ms = sum(x[3] for x in op_profiles)

        print(
            f"[RECOMPUTE][PATH_PROFILE] "
            f"path_len={len(path)} "
            f"total_ms={total_ms:.3f}",
            flush=True,
        )

        for node_name, before_shape, after_shape, ms in op_profiles:
            print(
                f"[RECOMPUTE][OP_PROFILE] "
                f"node={node_name} "
                f"shape={before_shape}->{after_shape} "
                f"ms={ms:.3f}",
                flush=True,
            )
        return cur
    def _compute_node_from_any_available(self, target_node_name):
        if target_node_name in self.node_values:
            return self.node_values[target_node_name]

        best_start = None
        best_path = None

        for start_name in list(self.node_values.keys()):
            path = self._find_path(start_name, target_node_name)

            if path is None:
                continue

            if best_path is None or len(path) < len(best_path):
                best_start = start_name
                best_path = path

        if best_path is None:
            raise RuntimeError(
                f"[RECOMPUTE] no available path to {target_node_name}; "
                f"available={list(self.node_values.keys())[:30]}"
            )

        old_start = self.current_start

        try:
            self.current_start = best_start
            out = self.recompute_path(
                start_tensor=self.node_values[best_start],
                path=best_path,
            )
        finally:
            self.current_start = old_start

        self.node_values[target_node_name] = out

        print(
            f"[RECOMPUTE][ANY_AVAILABLE] "
            f"target={target_node_name} "
            f"start={best_start} "
            f"path_len={len(best_path)}",
            flush=True,
        )

        return out