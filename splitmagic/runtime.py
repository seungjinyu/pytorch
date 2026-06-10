import torch
import torch.nn as nn
import torch.nn.functional as F
import os 

from collections import defaultdict

# 사용자가 import 하는 입구
# Node A / Node B 역할 선택 
from .graph import collect_saved_attrs
from .payload import payload_from_saved_attrs
from .replay import ReplayEngine

from .keymap import assign_jin_keys, assign_jin_keys_by_autograd_order
from .payload import payload_from_jin_items 
from .resolver import SavedTensorResolver
from .fx_trace import (
    explain_missing_keys,
    analyze_recompute_for_missing_keys,
    build_available_nodes_from_payload,
    build_available_tensors_from_jin1,
    build_fx_maps,
    build_jin_key_to_fx_node,
    find_nearest_available_start,
    build_path_from_start_to_node,
)
from .recompute import FXRecomputeEngine
from .resolver import read_jin1_payload
from .debug_backward_graph import dump_backward_graph


ALWAYS_LOCAL_KEYS = {
    "conv2d:0:weight",
    "conv2d:1:weight",
    "addmm:0:mat2",
    "addmm:1:mat2",
    "addmm:2:mat2",
}

# Minimal keys required for full-model correctness in LeNet.
#
# Note:
# - conv2d:1:input is actually the input to conv1 in current JIN ordering.
#   It is required for conv1 weight gradient.
# - conv2d:0:input is the checkpoint activation before conv2.
#   It is used for checkpoint recompute from conv2 onward.
# - addmm:*:mat1 are inputs to Linear layers.
#   They are required for exact Linear weight gradients unless recomputed.
REQUIRED_KEYS = {
    "conv2d:0:input",
    "conv2d:1:input", # x 
    "addmm:0:mat1",
    "addmm:1:mat1",
    "addmm:2:mat1",
}

OPTIONAL_KEYS = {
    "relu:0:out",
    "relu:1:out",
    "relu:2:out",
    "relu:3:out",
    "maxpool2d:0:input",
    "maxpool2d:0:indices",
    "maxpool2d:1:input",
    "maxpool2d:1:indices",
}

SEND_MIN_KEYS = {
    "conv2d:1:input",          # model input x
    "maxpool2d:0:indices",
    "maxpool2d:1:indices",
}

RECOMPUTE_KEYS = {
    "relu:0:out",
    "relu:1:out",
    "relu:2:out",
    "relu:3:out",
    "addmm:0:mat1",
    "addmm:1:mat1",
    "addmm:2:mat1",
    "conv2d:0:input",
}

LENET_REQUIRED_KEYS = SEND_MIN_KEYS | RECOMPUTE_KEYS

LENET_OPTIONAL_OR_REPLACEABLE_KEYS = {
    # "relu:0:out",
    # "relu:1:out",
    # "relu:2:out",
    # "relu:3:out",
}

################################ PLAN ###################################
def build_backward_plan(jin_items):
    plan = []

    for item in jin_items:
        gk = item.get("graph_key")
        if gk is None:
            continue

        parts = gk.split(":")
        if len(parts) < 4:
            continue

        op = parts[1]
        idx = parts[2]
        suffix = ":".join(parts[3:])

        if op not in ("conv", "addmm", "maxpool2d", "bn", "relu"):
            continue

        t = item["tensor"]

        plan.append({
            "op": op,
            "idx": idx,
            "suffix": suffix,
            "key": gk,
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
        })

    return plan
################################ PLAN ###################################
def capture_module_named_payload(model):

    import torch.nn as nn
    import torch.nn.functional as F

    records = []
    handles = []
    call_counts = {}
    bn_records_by_name = defaultdict(list)

    relu_mask_records = []

    def add_record(key, tensor):
        t = tensor.detach().cpu().clone()

        print(f"[ADD_RECORD] {key}")
        records.append({
            "key": key,
            "graph_key": key,
            "tensor": t,
            "node": "module",
            "attr": key,
            "shape": tuple(t.shape),
            "dtype": t.dtype,
            "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        })

    def make_call_key(base_key):
        i = call_counts.get(base_key, 0)
        call_counts[base_key] = i + 1
        return f"{base_key}#{i}"

    def make_hook(name, module):
        def hook(mod, inputs, output):
            if len(inputs) == 0:
                return

            inp = inputs[0]
            if isinstance(mod, nn.BatchNorm2d):
                print(f"[BN_HOOK] {name} shape={tuple(inp.shape)}")
                bn_records_by_name[name].append({
                    "input": inp.detach().cpu().clone(),
                    "running_mean": mod.running_mean.detach().cpu(),
                    "running_var": mod.running_var.detach().cpu(),
                    "weight": (
                        mod.weight.detach().cpu()
                        if mod.weight is not None
                        else None
                    ),
                })

            elif isinstance(mod, nn.Conv2d):
                add_record(
                    f"graph:conv:{name}:input",
                    inp,
                )
            elif isinstance(mod, nn.ReLU):
                fwd_idx = call_counts.get("relu_mask", 0)
                call_counts["relu_mask"] = fwd_idx + 1

                mask = (output > 0).to(torch.uint8).detach().cpu().clone()

                print(
                    f"[PY][RELU_MASK_FWD] fwd_idx={fwd_idx} "
                    f"shape={tuple(output.shape)} "
                    f"numel={output.numel()}"
                )

                relu_mask_records.append({
                    "fwd_idx": fwd_idx,
                    "tensor": mask,
                    "shape": tuple(output.shape),
                    "numel": output.numel(),
                })

            elif isinstance(mod, nn.Linear):
                add_record(f"graph:addmm:{name}:mat1", inp)
                add_record(f"graph:addmm:{name}:mat2", mod.weight.t())

            elif isinstance(mod, nn.MaxPool2d):
                add_record(f"graph:maxpool2d:{name}:input", inp)

                _, indices = F.max_pool2d(
                    inp,
                    kernel_size=mod.kernel_size,
                    stride=mod.stride,
                    padding=mod.padding,
                    dilation=mod.dilation,
                    ceil_mode=mod.ceil_mode,
                    return_indices=True,
                )
                add_record(f"graph:maxpool2d:{name}:indices", indices)

        return hook
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.Conv2d, nn.ReLU, nn.Linear, nn.MaxPool2d)):
            handles.append(module.register_forward_hook(make_hook(name, module)))
    def finalize_bn_records(jin_items):
        import torch

        print("[BN_HOOK_KEYS]", sorted(bn_records_by_name.keys()))

        used_names = set()

        for item in jin_items:
            gk = item.get("graph_key", "")

            if not (gk.startswith("graph:bn:") and gk.endswith(":input")):
                continue

            parts = gk.split(":")
            idx = int(parts[2])

            target = item["tensor"].detach().cpu()

            matched_name = None
            matched_record = None

            for bn_name, records in bn_records_by_name.items():
                if bn_name in used_names:
                    continue

                for r in records:
                    cand = r["input"]

                    if tuple(cand.shape) != tuple(target.shape):
                        continue

                    if torch.equal(cand, target):
                        matched_name = bn_name
                        matched_record = r
                        break

                if matched_record is not None:
                    break

            if matched_record is None:
                print(
                    f"[BN_AUTO_MATCH_FAIL] idx={idx} "
                    f"graph_key={gk} shape={tuple(target.shape)}"
                )
                continue

            used_names.add(matched_name)

            r = matched_record

            print(
                f"[BN_AUTO_MATCH] idx={idx} "
                f"name={matched_name} "
                f"shape={tuple(r['input'].shape)}"
            )

            add_record(f"graph:bn:{matched_name}:input", r["input"])
            add_record(f"graph:bn:{matched_name}:running_mean", r["running_mean"])
            add_record(f"graph:bn:{matched_name}:running_var", r["running_var"])

            if r["weight"] is not None:
                add_record(f"graph:bn:{matched_name}:weight", r["weight"])

            print(f"[BN_FINALIZE_OK] idx={idx} name={matched_name}")

    def finalize_relu_masks():
        # forward 순서로 모은 ReLU mask를 backward 순서로 뒤집어서 저장
        for bwd_idx, rec in enumerate(reversed(relu_mask_records)):
            key = f"relu_mask:{bwd_idx}"
            t = rec["tensor"]

            print(
                f"[PY][RELU_MASK_FINAL] "
                f"fwd_idx={rec['fwd_idx']} -> bwd_idx={bwd_idx} "
                f"key={key} "
                f"shape={rec['shape']} "
                f"numel={rec['numel']}"
            )

            add_record(key, t)

    return records, handles, finalize_bn_records, finalize_relu_masks

def build_module_alias_map(jin_items, module_records):
    import torch

    alias_map = {}
    saved_alias_map = {}

    # module_records를 key별로 정리
    module_by_key = {
        r["graph_key"]: r
        for r in module_records
        if "graph_key" in r
    }

    # -------------------------
    # BN: graph:bn:0:* -> graph:bn:<module_name>:*
    # -------------------------
    bn_module_inputs = [
        r for r in module_records
        if r.get("graph_key", "").startswith("graph:bn:")
        and r.get("graph_key", "").endswith(":input")
        and not r.get("graph_key", "").split(":")[2].isdigit()
    ]

    used_bn = set()

    for item in jin_items:
        gk = item.get("graph_key", "")

        if not (
            gk.startswith("graph:bn:")
            and gk.endswith(":input")
            and gk.split(":")[2].isdigit()
        ):
            continue

        target = item["tensor"].detach().cpu()

        matched = None

        for r in bn_module_inputs:
            rk = r["graph_key"]

            if rk in used_bn:
                continue

            cand = r["tensor"]

            if tuple(cand.shape) != tuple(target.shape):
                continue

            if torch.equal(cand, target):
                matched = r
                break

        if matched is None:
            print(f"[ALIAS_BN_FAIL] {gk}")
            continue

        used_bn.add(matched["graph_key"])

        old_prefix = gk.rsplit(":", 1)[0]
        new_prefix = matched["graph_key"].rsplit(":", 1)[0]

        for suffix in ["input", "running_mean", "running_var", "weight","result1", "result2"]:
            old_key = f"{old_prefix}:{suffix}"
            new_key = f"{new_prefix}:{suffix}"

            if new_key in module_by_key:
                alias_map[old_key] = new_key
                # print(f"[ALIAS_BN] {old_key} -> {new_key}")

    # -------------------------
    # Conv: graph:conv:0:input -> graph:conv:<module_name>:input
    # -------------------------
    conv_module_inputs = [
        r for r in module_records
        if r.get("graph_key", "").startswith("graph:conv:")
        and r.get("graph_key", "").endswith(":input")
        and not r.get("graph_key", "").split(":")[2].isdigit()
    ]

    used_conv = set()

    for item in jin_items:
        gk = item.get("graph_key", "")

        if not (
            gk.startswith("graph:conv:")
            and gk.endswith(":input")
            and gk.split(":")[2].isdigit()
        ):
            continue

        target = item["tensor"].detach().cpu()

        matched = None

        for r in conv_module_inputs:
            rk = r["graph_key"]

            if rk in used_conv:
                continue

            cand = r["tensor"]

            if tuple(cand.shape) != tuple(target.shape):
                continue

            if torch.equal(cand, target):
                matched = r
                break

        if matched is None:
            print(f"[ALIAS_CONV_FAIL] {gk}")
            continue

        used_conv.add(matched["graph_key"])

        alias_map[gk] = matched["graph_key"]
        # print(f"[ALIAS_CONV] {gk} -> {matched['graph_key']}")

    return alias_map, saved_alias_map

def capture_batchnorm_inputs(model, x):
    import torch.nn as nn

    records = []
    handles = []

    def make_hook(name):
        def hook(mod, inputs, output):
            if len(inputs) == 0:
                return

            inp = inputs[0]

            records.append({
                "name": name,
                "input": inp.detach().cpu().clone(),
                "input_shape": tuple(inp.shape),
                "running_mean": mod.running_mean.detach().cpu().clone(),
                "running_var": mod.running_var.detach().cpu().clone(),
                "weight": mod.weight.detach().cpu().clone()
                if mod.weight is not None else None,
            })

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    return records

def capture_conv1x1_inputs(model, x):
    records = []
    handles = []

    def make_hook(name, module):
        def hook(mod, inputs, output):
            if len(inputs) == 0:
                return

            inp = inputs[0]

            if not hasattr(inp, "shape"):
                return

            if not isinstance(mod, nn.Conv2d):
                return

            if mod.kernel_size != (1, 1):
                return

            records.append({
                "name": name,
                "input": inp.detach().cpu().clone(),
                "input_shape": tuple(inp.shape),
                "weight_shape": tuple(mod.weight.shape),
            })

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    return records

def capture_batchnorm_saved_by_module(model, x):
    records = []
    handles = []

    def make_hook(name, module):
        def hook(mod, inputs, output):
            if len(inputs) == 0:
                return

            inp = inputs[0]

            records.append({
                "name": name,
                "input": inp.detach().cpu().clone(),
                "input_shape": tuple(inp.shape),
                "running_mean": mod.running_mean.detach().cpu().clone(),
                "running_var": mod.running_var.detach().cpu().clone(),
                "weight": mod.weight.detach().cpu().clone() if mod.weight is not None else None,
            })

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    return records


def shape_sig(shape):
    return "x".join(str(int(s)) for s in shape)

def classify_jin_key_for_policy(key):
    """
    Generic rule-based policy.
    Works for Conv/ReLU/MaxPool/Addmm style models.
    """

    # Node B local parameter
    if is_always_local_key(key):
        return "local"

    # Must send
    if key.startswith("maxpool2d:") and key.endswith(":indices"):
        return "send"
    if key.startswith("maxpool2d:") and key.endswith(":input"):
        return "send"
    # if key.startswith("batchnorm:") and (
    #     key.endswith(":running_mean")
    #     or key.endswith(":running_var")
    #     or key.endswith(":weight")
    # ):
    #     return "local"
    
    if key.startswith("batchnorm:"):
        return "send"

    # model input 쪽 conv input은 우선 send 유지
    # 일반적으로 가장 마지막 conv index가 실제 model input일 가능성이 높음.
    # 정확한 자동 판별은 shape/meta 기반으로 나중에 개선.
    if key.startswith("conv2d:") and key.endswith(":input"):
        return "recompute_or_send"

    # Recomputable candidates
    if key.startswith("relu:") and key.endswith(":out"):
        return "send"

    if key.startswith("addmm:") and key.endswith(":mat1"):
        return "send"

    # default: send
    return "send"

def get_jin_global_idx(item):
    if isinstance(item, dict):
        return item.get("global_idx", None)

    if hasattr(item, "global_idx"):
        return item.global_idx

    return None

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

def parse_jin_index(key):
    """
    Examples:
        conv2d:0:input -> 0
        relu:3:out     -> 3
        addmm:1:mat1   -> 1
    """
    parts = key.split(":")
    if len(parts) < 3:
        return None

    try:
        return int(parts[1])
    except ValueError:
        return None
    
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


def should_include_jin_key(
    key,
    policy="full",
    optional_keys=None,
):
    optional_keys = set(optional_keys or [])

    if policy == "full_with_params":
        return True

    if is_always_local_key(key):
        return False

    if policy == "full":
        return True

    if policy == "minimal":
        return key in REQUIRED_KEYS
    
    if policy == "send_min":
        return key in SEND_MIN_KEYS

    if policy == "custom":
        invalid = optional_keys - OPTIONAL_KEYS
        if invalid:
            raise ValueError(f"Invalid optional keys: {sorted(invalid)}")
        return key in REQUIRED_KEYS or key in optional_keys
    if policy == "auto_recompute":
        cls = classify_jin_key_for_policy(key)

        if cls == "local":
            return False

        if cls == "send":
            return True

        if cls == "recompute":
            return False

        if cls == "recompute_or_send":
            return True

    if policy == "odd":
        idx = parse_jin_index(key)
        return idx is not None and idx % 2 == 1

    if policy == "even":
        idx = parse_jin_index(key)
        return idx is not None and idx % 2 == 0

    raise ValueError(f"Unknown policy: {policy}")


def should_include_jin_item(
    item,
    policy="full",
    optional_keys=None,

):
    key = get_jin_key(item)

    return should_include_jin_key(
        key,
        policy=policy,
        optional_keys=optional_keys,
    )

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

def read_used_keys(path="/tmp/jin_used_keys.txt"):
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())
    
def build_plan_key_map(plan):
    """
    forward capture key를 dryrun backward key로 바꾸기 위한 map 생성.
    우선 shape 기준으로 매칭한다.
    """
    by_op_suffix = {}

    for e in plan:
        op = e["op"]
        suffix = e["suffix"]
        
        if op not in ("conv", "bn", "relu", "maxpool2d", "addmm"):
            continue

        by_op_suffix.setdefault((op, suffix), []).append(e)

    return by_op_suffix

def parse_shape_str(s):
    # "[32,512,4,4]" -> (32,512,4,4)
    s = s.strip()
    s = s.strip("[]")
    if not s:
        return tuple()
    return tuple(int(x) for x in s.split(","))


def remap_payload_to_dryrun_idx(payload, plan):
    old_tensors = payload.tensors
    new_tensors = {}

    if "model.output" in old_tensors:
        new_tensors["model.output"] = old_tensors["model.output"]

    used_old_keys = set()

    for e in plan:
        op = e["op"]
        idx = e["idx"]
        suffix = e["suffix"]
        shape = parse_shape_str(e["shape"])

        # weight/addmm mat2는 local fallback 가능하면 안 보내도 됨
        if op == "conv" and suffix == "weight":
            continue
        if op == "addmm" and suffix == "mat2":
            continue

        new_key = f"graph:{op}:{idx}:{suffix}"

        # 이미 정확한 key + shape가 있으면 사용
        if new_key in old_tensors and tuple(old_tensors[new_key].shape) == shape:
            new_tensors[new_key] = old_tensors[new_key]
            used_old_keys.add(new_key)
            continue

        # module 기반 key 중 op/suffix/shape가 맞는 첫 tensor를 사용
        found = False
        for old_key, tensor in old_tensors.items():
            if old_key in used_old_keys:
                continue

            parts = old_key.split(":")
            if len(parts) != 4:
                continue

            if parts[0] != "graph":
                continue
            if parts[1] != op:
                continue
            if parts[3] != suffix:
                continue
            if tuple(tensor.shape) != shape:
                continue

            new_tensors[new_key] = tensor
            used_old_keys.add(old_key)
            print(f"[REMAP] {old_key} -> {new_key}")
            found = True
            break

        if not found:
            print(f"[REMAP_MISS] {new_key} shape={shape}")

    payload.tensors = new_tensors
    return payload

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
    
    def capture(self,x,y,loss_fn):

        if self.role != "A":
            raise RuntimeError("Only role 'A' can capture tensors")
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        out = self.model(x)
        loss = loss_fn(out, y)  

        items = collect_saved_attrs(loss)
        payload = payload_from_saved_attrs(items)

        payload.add_tensor("model.input", x)
        payload.add_tensor("model.output",out)
        payload.add_meta("loss",{
            "value":float(loss.detach().cpu())
        })

        return payload
    
    def replay_backward(self, payload,x_dummy, y, loss_fn):

        if self.role != "B":
            raise RuntimeError("Only role 'B' can replay backward")
        
        logits, x_dummy = self.replay_engine.dummy_forward(x_dummy)

        loss = self.replay_engine.backward(
            logits, 
            y, 
            loss_fn
        )

        return loss 
    def capture_jin(
        self,
        x,
        y,
        loss_fn,
        policy="full",
        optional_keys=None,
        key_mode="module_debug",
    ):

        if self.role != "A":

            raise RuntimeError("Only role 'A' can capture tensors")
        print(f"[CAPTURE_JIN] key_mode={key_mode} policy={policy}")
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        
        module_records = []
        module_handles = []

        if key_mode in ("module_debug","graph"):
            module_records, module_handles,finalize_bn_records, finalize_relu_masks = capture_module_named_payload(self.model)

        try:
            out = self.model(x)
        finally:
            for h in module_handles:
                h.remove()

        loss = loss_fn(out, y)
        dump_backward_graph(loss)

        items = collect_saved_attrs(loss)
        jin_items = assign_jin_keys_by_autograd_order(items)
        backward_plan = build_backward_plan(jin_items)

        print("========== BACKWARD PLAN ==========")
        for i, e in enumerate(backward_plan):
            print(
                f"[PLAN] {i:03d} "
                f"op={e['op']} "
                f"idx={e['idx']} "
                f"suffix={e['suffix']} "
                f"key={e['key']} "
                f"shape={e['shape']}"
            )
        print("========== END BACKWARD PLAN ==========")

        if key_mode in ("module_debug", "graph"):
            print("[CALL] finalize_bn_records")
            finalize_bn_records(jin_items)
            print("[CALL] finalize_relu_masks")
            finalize_relu_masks()

        # all_keys = [
        #     item.get("graph_key", get_jin_key(item))
        #     for item in jin_items
        # ]

        before_count = len(jin_items)

        jin_items = [
            item for item in jin_items
            if should_include_jin_item(
                item,
                policy=policy,
                optional_keys=optional_keys,

            )
        ]

        all_keys = [
            item.get("graph_key", get_jin_key(item))
            for item in jin_items
        ]


        included_keys = [
            item.get("graph_key", get_jin_key(item))
            for item in jin_items
        ]

        all_items = jin_items + module_records
        payload = payload_from_jin_items(all_items)

        payload.meta["backward_plan"] = backward_plan

        for x in jin_items[:10]:
            print(x.get("graph_key"), x.get("jin_key"))

        if key_mode in ("module_debug","graph"):

            for r in module_records:
                key = r["graph_key"]

                if key not in all_keys:
                    all_keys.append(key)
                if key not in included_keys:
                    included_keys.append(key)


            alias_map, saved_alias_map = build_module_alias_map(
                jin_items,
                module_records,
            )

            payload.meta = getattr(payload, "meta", {})
            payload.meta["alias_map"] = alias_map
            alias_path = "/tmp/jin_payload_recv.bin" + ".alias"

            with open(alias_path, "w") as f:
                for old_key, new_key in alias_map.items():
                    f.write(f"{old_key}\t{new_key}\n")
            print(f"[ALIAS_WRITE] path={alias_path} n={len(alias_map)}")
            
            print(f"[MODULE_ALIAS_FILE] saved: {alias_path}")

            print("[MODULE_ALIAS_MAP]")
            for old_key, new_key in alias_map.items():
                print(f"  {old_key} -> {new_key}")

        payload.add_tensor("model.output", out)
        payload.add_meta("loss", {
            "value": float(loss.detach().cpu())
        })

        policy_meta = {
            "policy": policy,
            "before_count": before_count,
            "num_payload_tensors": len(jin_items),
            "all_keys":sorted(all_keys),
            "included_keys": sorted(included_keys),
            "optional_keys": sorted(list(optional_keys or [])),
        }

        # print("[A][POLICY]", policy_meta)
        payload.add_meta("tensor_policy", policy_meta)

        return payload
    
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

        resolver = SavedTensorResolver(
            payload=payload,
            local_keys=ALWAYS_LOCAL_KEYS,
            payload_path=payload_path,
        )
        policy_meta = tensor_policy or payload.meta.get("tensor_policy", {})

        required_keys = set(policy_meta.get("all_keys", []))
        required_keys = {
            k for k in required_keys
            if not is_always_local_key(k)
        }

        try:
            resolver.check_required(required_keys)
        except RuntimeError as e:
            print("[FX-ANALYSIS] missing required keys for backward execution")
            print(e)
            missing = [
                k for k in LENET_REQUIRED_KEYS
                if k not in resolver.sources or resolver.sources[k] == "missing"
            ]

            # explain_missing_keys(self.model, missing)

            payload_keys = set(resolver.jin1_keys | set(payload.tensors.keys()))

            available_nodes = build_available_nodes_from_payload(self.model,payload_keys=payload_keys)

            recompute_results = analyze_recompute_for_missing_keys(
                model=self.model,
                missing_keys=missing, 
                available_nodes=available_nodes,
            )

            available_tensors = build_available_tensors_from_jin1(
                model =self.model,
                payload_keys=payload_keys,
                payload_path=payload_path,
                device= x_dummy.device,
            )

            print("[FX-AVAILABLE][NODES]", sorted(available_tensors.keys()))

            recompute_engine = FXRecomputeEngine(self.model)

            recomputed = {}

            node_map = build_fx_maps(self.model)
            key_to_node = build_jin_key_to_fx_node(self.model)
            available_tensor_nodes = set(available_tensors.keys())

            for key, info in recompute_results.items():

                if not info["can_recompute"]:
                    continue

                if key not in key_to_node:
                    print(f"[RECOMPUTE] no FX target for key={key}")
                    continue

                target_node = key_to_node[key]

                start = find_nearest_available_start(
                    node_map=node_map,
                    node_name=target_node,
                    available_nodes=available_tensor_nodes,
                )

                if start is None:
                    print(f"[RECOMPUTE] no tensor start available for key={key}")
                    continue

                path = build_path_from_start_to_node(
                    node_map=node_map,
                    start_node=start,
                    target_node=target_node,
                )

                if not path:
                    print(f"[RECOMPUTE] empty path for key={key}")
                    continue

                start_tensor = available_tensors[start]

                out = recompute_engine.recompute_path(
                    start_tensor=start_tensor,
                    path=path,
                )

                recomputed[key] = out.detach().cpu()

                print(
                    f"[RECOMPUTE] key={key} "
                    f"result_shape={tuple(out.shape)}"
                )
            jin_payload = read_jin1_payload(payload_path)

            for key, tensor in recomputed.items():

                print(
                    f"[RECOMPUTE][INJECT] "
                    f"{key} shape={tuple(tensor.shape)}"
                )

                jin_payload.tensors[key] = tensor

            jin_payload.save_jin1(payload_path)

            print(
                f"[RECOMPUTE] updated JIN1 saved: {payload_path}"
            )
            resolver = SavedTensorResolver(

                payload=payload,
                local_keys=ALWAYS_LOCAL_KEYS,
                payload_path=payload_path,
            )

            required_keys = set(
                payload.meta.get("tensor_policy", {}).get("all_keys", [])
            )

            required_keys = {
                k for k in required_keys
                if not is_always_local_key(k)
            }

            resolver.check_required(required_keys)
        loss.backward()

        grad_dump = {}

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            grad_dump[name] = p.grad.detach().cpu().clone()

            # print(
            #     f"[B][GRAD] {name} "
            #     f"mean={p.grad.mean().item():.8f} "
            #     f"absmax={p.grad.abs().max().item():.8f}"
            # )

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
