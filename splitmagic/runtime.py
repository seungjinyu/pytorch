import torch
import os 

from .replay import ReplayEngine
from .payload import  payload_from_jin_items


ALWAYS_LOCAL_KEYS = {
    "conv2d:0:weight",
    "conv2d:1:weight",
    "addmm:0:mat2",
    "addmm:1:mat2",
    "addmm:2:mat2",
}

def is_always_local_key(key):
    if key.startswith("conv2d:") and key.endswith(":weight"):
        return True
    if key.startswith("addmm:") and key.endswith(":mat2"):
        return True

    if key.startswith("graph:conv:") and key.endswith(":weight"):
        return True
    if key.startswith("graph:addmm:") and key.endswith(":mat2"):
        return True

    return False

# read the dryrun plan and be ready to send it in the payload 
def read_dryrun_plan(path="/tmp/jin_dryrun_plan.tsv"):
    plan = []
    
    if not os.path.exists(path):
        return plan 

    with open(path,"r") as f :
        for line in f :
            line = line.rstrip("\n")

            if not line:
                continue 
            
            parts = line.split("\t")
            if len(parts) == 5 :
                row_id, op, idx, suffix, shape = parts 
            elif len(parts) == 4 :
                op, idx, suffix, shape = parts 
                row_id = len(plan)
            else:
                raise ValueError(f"bad dryrun plan line: {line}")
            # row,op,idx,suffix,shape = line.split("\t")

            plan.append({
                "row_id":int(row_id),
                "op":op,
                "idx":int(idx),
                "suffix":suffix,
                "shape": shape,
            })
    return plan

def keys_from_dryrun_plan(plan):
    keys = set()

    for e in plan:
        op = e["op"]
        idx = e["idx"]
        suffix = e["suffix"]

        # local parameter라 안 보내도 되는 것
        if op == "conv" and suffix == "weight":
            continue
        if op == "addmm" and suffix == "mat2":
            continue

        keys.add(f"graph:{op}:{idx}:{suffix}")

    return keys

def parse_shape_str(s):
    # "[32,512,4,4]" -> (32,512,4,4)
    s = s.strip()
    s = s.strip("[]")
    if not s:
        return tuple()
    return tuple(int(x) for x in s.split(","))

class SplitRuntime:
    def __init__(self, model, role: str):
        self.model = model 
        self.role = role.upper()

        if self.role not in ("A","B"):
            raise ValueError("Role must be either 'A' or 'B'")
        
        if self.role == "B":
            self.replay_engine = ReplayEngine(model)
        else:
            self.replay_engine = None

    def capture_jin_forward_plan(self, x, y, plan):
        if self.role != "A":
            raise RuntimeError("Only role 'A' can capture tensors")

        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self.model.train()

        tensors = {}
        handles = []

        queues = {}

        for e in plan:
            op = e["op"]
            suffix = e["suffix"]
            idx = e["idx"]

            # B local parameter라 payload로 안 보냄
            if op == "conv" and suffix == "weight":
                continue
            if op == "addmm" and suffix == "mat2":
                continue

            key = f"graph:{op}:{idx}:{suffix}"
            queues.setdefault((op, suffix), []).append(key)

        # plan은 backward order, forward hook은 forward order
        # 그래서 각 op/suffix queue를 뒤집음
        for k in queues:
            queues[k] = list(reversed(queues[k]))

        def pop_key(op, suffix):
            q = queues.get((op, suffix), None)
            if not q:
                raise RuntimeError(f"[A][PLAN_KEY_EMPTY] op={op} suffix={suffix}")
            return q.pop(0)

        def save_tensor(key, tensor):
            tensors[key] = tensor.detach().cpu().contiguous()

        def make_hook(module):
            def hook(mod, inputs, output):
                if len(inputs) == 0:
                    return

                inp = inputs[0]

                if isinstance(mod, nn.Conv2d):
                    key = pop_key("conv", "input")
                    save_tensor(key, inp)

                elif isinstance(mod, nn.BatchNorm2d):
                    # BN backward saved input
                    save_tensor(pop_key("bn", "input"), inp)

                    # BN weight
                    if mod.weight is not None:
                        save_tensor(pop_key("bn", "weight"), mod.weight)

                    save_tensor(pop_key("bn", "running_mean"), mod.running_mean)
                    save_tensor(pop_key("bn", "running_var"), mod.running_var)

                    # BN result1/result2 = save_mean / save_invstd
                    dims = (0, 2, 3)
                    mean = inp.detach().mean(dim=dims)
                    var = inp.detach().var(dim=dims, unbiased=False)
                    invstd = torch.rsqrt(var + mod.eps)

                    save_tensor(pop_key("bn", "result1"), mean)
                    save_tensor(pop_key("bn", "result2"), invstd)

                elif isinstance(mod, nn.ReLU):
                    key = pop_key("relu", "result")
                    save_tensor(key, output)

                elif isinstance(mod, nn.Linear):
                    key = pop_key("addmm", "mat1")
                    save_tensor(key, inp)

                elif isinstance(mod, nn.MaxPool2d):
                    save_tensor(pop_key("maxpool2d", "input"), inp)

                    _, indices = F.max_pool2d(
                        inp,
                        kernel_size=mod.kernel_size,
                        stride=mod.stride,
                        padding=mod.padding,
                        dilation=mod.dilation,
                        ceil_mode=mod.ceil_mode,
                        return_indices=True,
                    )

                    save_tensor(pop_key("maxpool2d", "indices"), indices)

            return hook

        for _, m in self.model.named_modules():
            if isinstance(
                m,
                (
                    nn.Conv2d,
                    nn.BatchNorm2d,
                    nn.ReLU,
                    nn.Linear,
                    nn.MaxPool2d,
                ),
            ):
                handles.append(m.register_forward_hook(make_hook(m)))

        try:
            out = self.model(x)
        finally:
            for h in handles:
                h.remove()

        tensors["model.output"] = out.detach().cpu().contiguous()

        # 남은 queue가 있으면 A forward에서 못 채운 saved tensor가 있다는 뜻
        leftovers = {
            f"{op}:{suffix}": len(q)
            for (op, suffix), q in queues.items()
            if len(q) > 0
        }

        if leftovers:
            raise RuntimeError(f"[A][PLAN_KEYS_LEFTOVER] {leftovers}")

        items = []
        for key, tensor in tensors.items():
            items.append({
                "key": key,
                "jin_key": key,
                "graph_key": key,
                "tensor": tensor,
                "node": key,
                "attr": key,
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
                "requires_grad": False,
            })

        payload = payload_from_jin_items(items)

        payload.meta = getattr(payload, "meta", {})
        payload.meta["dryrun_backward_plan"] = plan
        payload.meta["tensor_policy"] = {
            "policy": "forward_only_plan_keys",
            "included_keys": sorted(tensors.keys()),
            "num_payload_tensors": len(tensors),
            "all_keys": sorted(tensors.keys()),
        }

        return payload    
    
    def info(self):
        return f"SplitRuntime(role={self.role}, model={self.model.__class__.__name__})"
    
    def backward_jin(
        self,
        x_dummy,
        y,
        payload,
        loss_fn,
        payload_path=None,
        tensor_policy=None,
    ):
        if self.role != "B":
            raise RuntimeError("backward_jin() is only available for Node B")

        self.model.train()
        self.model.zero_grad(set_to_none=True)

        out_dummy = self.model(x_dummy)

        if "model.output" not in payload.tensors:
            raise KeyError("payload does not contain 'model.output'")

        out_real = payload.tensors["model.output"].detach().to(out_dummy.device)

        out = out_dummy + (out_real - out_dummy).detach()

        loss = loss_fn(out, y)

        print("[B][BACKWARD] start")

        #

        #
        loss.backward()

        grad_dump = {}

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            grad_dump[name] = p.grad.detach().cpu().clone()

        torch.save(grad_dump, "/tmp/node_b_grads.pt")

        print("[B][BACKWARD] done")

        return loss

    def get_jin_key(item):
        if isinstance(item, dict):
            for name in ["key", "jin_key", "name"]:
                if name in item:
                    return item[name]
            raise KeyError(f"No key field in item: {item.keys()}")

        if isinstance(item, (tuple, list)):
            return item[0]

        if hasattr(item, "key"):
            return item.key

        if hasattr(item, "jin_key"):
            return item.jin_key

        raise TypeError(f"Cannot get key from item type: {type(item)}")
