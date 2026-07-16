import time 
import torch


class FXRecomputeEngine:
    def __init__(self, model, gm=None, node_values=None):
        self.model = model
        self.modules = dict(model.named_modules())        

        self.gm = gm 
        self.node_values = node_values or {}

        self.fx_nodes = {}

        self.current_start = None

        self.profile_db ={} # node_name -> avg_ms
        
        # 실제 recompute 과정에서 실행된 node 를 순서대로 기록
        self.executed_recompute_nodes = []

        if gm is not None :
            self.fx_nodes = {
                n.name: n
                for n in gm.graph.nodes
            }


    def _get_module_for_node(self, node_name):
        if node_name in self.modules:
            return self.modules[node_name]

        module_key = node_name.replace("_", ".")

        if module_key in self.modules:
            return self.modules[module_key]

        return None
    
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

        for node_name in path[1:]:

            if node_name == "flatten":
                self.executed_recompute_nodes.append(node_name)

                cur = cur.flatten(1)
                self.node_values[node_name] = cur

            elif node_name == "view":
                raise NotImplementedError(
                    "view recompute needs shape info"
                )

            elif node_name == "reshape":
                raise NotImplementedError(
                    "reshape recompute needs shape info"
                )

            elif node_name.startswith("add"):
                self.executed_recompute_nodes.append(node_name)

                cur = self._compute_add(node_name, cur)
                self.node_values[node_name] = cur

            elif (
                "relu" in node_name
                and node_name not in self.modules
            ):
                self.executed_recompute_nodes.append(node_name)

                cur = torch.relu(cur)
                self.node_values[node_name] = cur

            else:
                module = self._get_module_for_node(node_name)

                if module is None:
                    print(
                        f"[RECOMPUTE_SKIP_NODE] node={node_name}",
                        flush=True,
                    )
                    continue

                self.executed_recompute_nodes.append(node_name)

                cur = module(cur)
                self.node_values[node_name] = cur

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
    
    def estimate_recompute_cost(self, path):
        total_ms = 0.0
        missing = []

        for node_name in path[1:]:
            if node_name in self.profile_db:
                total_ms += self.profile_db[node_name]
            else:
                missing.append(node_name)

        return total_ms, missing
    
    @torch.no_grad()
    def profile_module(self, node_name, input_tensor, repeat = 100 , warmup = 10):

        module = self._get_module_for_node(node_name)

        if module is None:
            raise RuntimeError(f"[PROFILE] no module for node={node_name}")
        was_training = module.training
        # module.eval()

        x = input_tensor.detach()

        # warmup 
        for _ in range(warmup):
            _ = module(x)

        if x.is_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        for _ in range(repeat):
            _ = module(x)
        
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        module.train(was_training)

        avg_ms = (t1 - t0) * 1000 / repeat
        self.profile_db[node_name] = avg_ms
        return avg_ms
    
    @torch.no_grad()
    def profile_path(self, start_tensor, path, repeat=100, warmup=10):
        cur = start_tensor.detach()
        self.current_start = path[0]
        self.node_values[path[0]] = start_tensor

        for node_name in path[1:]:

            if node_name == "flatten":
                self.profile_db[node_name] = 0.0
                cur = cur.flatten(1)
                continue

            if node_name.startswith("add"):
                node = self.fx_nodes[node_name]
                lhs_name = node.args[0].name
                rhs_name = node.args[1].name

                lhs = self.node_values.get(lhs_name)
                rhs = self.node_values.get(rhs_name)

                if lhs is None:
                    lhs = self._compute_node_from_any_available(lhs_name)

                if rhs is None:
                    rhs = self._compute_node_from_any_available(rhs_name)

                for _ in range(warmup):
                    _ = lhs + rhs

                if lhs.is_cuda:
                    torch.cuda.synchronize()

                t0 = time.perf_counter()

                for _ in range(repeat):
                    _ = lhs + rhs

                if lhs.is_cuda:
                    torch.cuda.synchronize()

                t1 = time.perf_counter()

                avg_ms = (t1 - t0) * 1000.0 / repeat
                self.profile_db[node_name] = avg_ms

                cur = lhs + rhs
                self.node_values[node_name] = cur

                print(
                    f"[PROFILE] {node_name} avg_ms={avg_ms:.6f}",
                    flush=True,
                )
                continue

            if "relu" in node_name and self._get_module_for_node(node_name) is None:
                # functional relu
                for _ in range(warmup):
                    _ = torch.relu(cur)

                if cur.is_cuda:
                    torch.cuda.synchronize()

                t0 = time.perf_counter()

                for _ in range(repeat):
                    _ = torch.relu(cur)

                if cur.is_cuda:
                    torch.cuda.synchronize()

                t1 = time.perf_counter()

                avg_ms = (t1 - t0) * 1000.0 / repeat
                self.profile_db[node_name] = avg_ms

                print(f"[PROFILE] {node_name} avg_ms: {avg_ms:.6f}")

                cur = torch.relu(cur)
                continue

            module = self._get_module_for_node(node_name)

            if module is None:
                print(f"[PROFILE_SKIP_NODE] node={node_name}", flush=True)
                continue

            if node_name in self.profile_db:
                cur = module(cur)
                continue

            avg_ms = self.profile_module(
                node_name=node_name,
                input_tensor=cur,
                repeat=repeat,
                warmup=warmup,
            )

            print(f"[PROFILE] {node_name} avg_ms: {avg_ms:.6f}")

            cur = module(cur)

        return dict(self.profile_db)
    
    def save_profile_db_csv(self, path="./recompute_layer_profile.csv"):
        import csv
        with open(path, "w") as f:

            writer = csv.writer(f)
            writer.writerow(["node_name", "avg_ms"])

            for node_name, avg_ms in self.profile_db.items():
                writer.writerow([node_name, avg_ms])
        
        print(f"[PROFILE_SAVE] layer_profile_csv={path}", flush=True)

    def load_profile_db_csv(
        self,
        path="./recompute_layer_profile.csv",
    ):
        import csv

        profile_db = {}

        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                node_name = row["node_name"]
                avg_ms = float(row["avg_ms"])
                profile_db[node_name] = avg_ms

        self.profile_db = profile_db

        print(
            f"[PROFILE_LOAD] "
            f"path={path} "
            f"nodes={len(profile_db)}",
            flush=True,
        )

        return dict(self.profile_db)