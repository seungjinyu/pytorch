import torch
import torch.nn as nn
import torch.nn.functional as F

import os

# 사용자가 import 하는 입구
# Node A / Node B 역할 선택 
from .graph import collect_saved_attrs
from .payload import payload_from_saved_attrs
from .replay import ReplayEngine

from .keymap import assign_jin_keys
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
def build_module_alias_map(jin_items, module_records):
    """
    기존 shape/counter key -> module-name key 매핑 생성.

    주의:
    - shape만으로 매칭하면 BN의 result1/result2/running_mean/weight가 섞임.
    - 그래서 op 종류 + attr 종류별로 분리해서 매칭한다.
    - BN result1/result2는 module hook에서 직접 얻기 어렵기 때문에 여기서는 alias하지 않는다.
    """

    alias = {}

    module_by_kind = {
        "conv2d_input": [],
        "batchnorm_input": [],
        "batchnorm_running_mean": [],
        "batchnorm_running_var": [],
        "batchnorm_weight": [],
        "relu_out": [],
        "maxpool2d_input": [],
        "maxpool2d_indices": [],
        "addmm_mat1": [],
    }

    # 1. module-name records를 종류별로 분리
    for r in module_records:
        key = r["key"]

        if key.startswith("module:conv2d:") and key.endswith(":input"):
            module_by_kind["conv2d_input"].append(r)

        elif key.startswith("module:batchnorm:") and key.endswith(":input"):
            module_by_kind["batchnorm_input"].append(r)

        elif key.startswith("module:batchnorm:") and key.endswith(":running_mean"):
            module_by_kind["batchnorm_running_mean"].append(r)

        elif key.startswith("module:batchnorm:") and key.endswith(":running_var"):
            module_by_kind["batchnorm_running_var"].append(r)

        elif key.startswith("module:batchnorm:") and key.endswith(":weight"):
            module_by_kind["batchnorm_weight"].append(r)

        elif key.startswith("module:relu:") and key.endswith(":out"):
            module_by_kind["relu_out"].append(r)

        elif key.startswith("module:maxpool2d:") and key.endswith(":input"):
            module_by_kind["maxpool2d_input"].append(r)

        elif key.startswith("module:maxpool2d:") and key.endswith(":indices"):
            module_by_kind["maxpool2d_indices"].append(r)

        elif key.startswith("module:addmm:") and key.endswith(":mat1"):
            module_by_kind["addmm_mat1"].append(r)

    for kind in module_by_kind:
        module_by_kind[kind] = list(reversed(module_by_kind[kind]))

    used = set()

    def get_kind(old_key):
        if old_key.startswith("conv2d:") and old_key.endswith(":input"):
            return "conv2d_input"

        if old_key.startswith("batchnorm:"):
            # if old_key.endswith(":input"):
            #     return "batchnorm_input"
            # if old_key.endswith(":running_mean"):
            #     return "batchnorm_running_mean"
            # if old_key.endswith(":running_var"):
            #     return "batchnorm_running_var"
            # if old_key.endswith(":weight"):
            #     return "batchnorm_weight"
            return None

        if old_key.startswith("relu:") and old_key.endswith(":out"):
            return "relu_out"

        if old_key.startswith("maxpool2d:") and old_key.endswith(":input"):
            return "maxpool2d_input"

        if old_key.startswith("maxpool2d:") and old_key.endswith(":indices"):
            return "maxpool2d_indices"

        if old_key.startswith("addmm:") and old_key.endswith(":mat1"):
            return "addmm_mat1"

        return None

    # 2. old JIN key를 module-name key에 매칭
    for item in jin_items:
        old_key = item.get("jin_key")
        if old_key is None:
            continue

        kind = get_kind(old_key)
        if kind is None:
            continue

        shape = tuple(item["shape"])
        candidates = module_by_kind[kind]

        for r in candidates:
            rid = id(r)

            if rid in used:
                continue

            if tuple(r["shape"]) == shape:
                alias[old_key] = r["key"]
                used.add(rid)
                break

    return alias

def capture_module_named_payload( model , x ):
    import torch.nn as nn

    records = []
    handles = []

    def add_record(key, tensor):
        records.append({
            "key": key,
            "tensor": tensor.detach().cpu().clone(),
            "shape": tuple(tensor.shape),
        })
    
    def make_hook(name, module):

        def hook(mod, inputs, output):
            if len(inputs) == 0:
                return
            
            inp = inputs[0]

            if isinstance(mod, nn.Conv2d):
                add_record(f"module:conv2d:{name}:input", inp)

            elif isinstance(mod , nn.BatchNorm2d):
                add_record(f"module:batchnorm:{name}:input", inp)
                add_record(f"module:batchnorm:{name}:running_mean", mod.running_mean)
                add_record(f"module:batchnorm:{name}:running_var", mod.running_var)

                if mod.weight is not None:
                    add_record(f"module:batchnorm:{name}:weight", mod.weight)

            elif isinstance(mod, nn.ReLU):
                add_record(f"module:relu:{name}:out", output)
            
            elif isinstance(mod, nn.Linear):
                add_record(f"module:addmm:{name}:mat1", inp)

            elif isinstance(mod, nn.MaxPool2d):

                add_record(f"module:maxpool2d:{name}:input", inp)

                _, indices = F.max_pool2d(
                    inp,
                    kernel_size=mod.kernel_size,
                    stride=mod.stride,
                    padding=mod.padding,
                    dilation=mod.dilation,
                    ceil_mode=mod.ceil_mode,
                    return_indices=True,
                )

                add_record(f"module:maxpool2d:{name}:indices", indices)
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Linear, nn.MaxPool2d)):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    with torch.no_grad():

        model(x)

    for h in handles:
        h.remove()

    return records
                


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
        key_mode="module",
    ):

        if self.role != "A":
            raise RuntimeError("Only role 'A' can capture tensors")

        self.model.train()
        self.model.zero_grad(set_to_none=True)

        out = self.model(x)
        loss = loss_fn(out, y)

        items = collect_saved_attrs(loss)
        jin_items = assign_jin_keys(items)

        all_keys = [
            get_jin_key(item)
            for item in jin_items
        ]

        before_count = len(jin_items)

        jin_items = [
            item for item in jin_items
            if should_include_jin_item(
                item,
                policy=policy,
                optional_keys=optional_keys,

            )
        ]

        included_keys = [
            get_jin_key(item)
            for item in jin_items
        ]

        payload = payload_from_jin_items(jin_items)

        if key_mode == "module_debug":
            module_records = capture_module_named_payload(
                self.model,
                x,
            )

            for r in module_records:
                key = r["key"]
                tensor = r["tensor"]

                if key not in payload.tensors:
                    payload.add_tensor(key, tensor)

                    all_keys.append(key)
                    included_keys.append(key)

                print(
                    f"[MODULE_PAYLOAD] {key} "
                    f"shape={r['shape']}"
                )

            alias_map = build_module_alias_map(
                jin_items,
                module_records,
            )

            payload.meta = getattr(payload, "meta", {})
            payload.meta["alias_map"] = alias_map
            alias_path = "/tmp/jin_payload_recv.bin" + ".alias"

            with open(alias_path, "w") as f:
                for old_key, new_key in alias_map.items():
                    f.write(f"{old_key}\t{new_key}\n")

            print(f"[MODULE_ALIAS_FILE] saved: {alias_path}")

            print("[MODULE_ALIAS_MAP]")
            for old_key, new_key in alias_map.items():
                print(f"  {old_key} -> {new_key}")

        if policy in ("full", "auto_recompute"):
            conv1x1_records = capture_conv1x1_inputs(self.model, x)

            # backward order에 맞추기 위해 reverse
            conv1x1_records = list(reversed(conv1x1_records))

            sig_counts = {}

            for r in conv1x1_records:
                in_sig = shape_sig(r["input_shape"])
                w_sig = shape_sig(r["weight_shape"])

                sig = f"{in_sig}:{w_sig}"
                idx = sig_counts.get(sig, 0)
                sig_counts[sig] = idx + 1

                key = f"conv2d:{in_sig}:{w_sig}:{idx}:input"

                if key not in payload.tensors:
                    payload.add_tensor(key, r["input"])
                    all_keys.append(key)
                    included_keys.append(key)

                    print(
                        f"[SUPPLEMENT][CONV1x1] {key} "
                        f"module={r['name']} "
                        f"shape={r['input_shape']}"
                    )
            bn_records = capture_batchnorm_inputs(self.model, x)

            downsample_bn_override = {
                "layer2.0.downsample.1": "batchnorm:32x128x16x16:4",
                "layer3.0.downsample.1": "batchnorm:32x256x8x8:4",
                "layer4.0.downsample.1": "batchnorm:32x512x4x4:4",
            }

            for r in bn_records:
                name = r["name"]

                if name not in downsample_bn_override:
                    continue

                prefix = downsample_bn_override[name]

                tensors = {
                    "input": r["input"],
                    "running_mean": r["running_mean"],
                    "running_var": r["running_var"],
                }

                if r["weight"] is not None:
                    tensors["weight"] = r["weight"]

                for suffix, tensor in tensors.items():
                    key = f"{prefix}:{suffix}"

                    payload.tensors[key] = tensor

                    if key not in all_keys:
                        all_keys.append(key)

                    if key not in included_keys:
                        included_keys.append(key)

                    print(
                        f"[SUPPLEMENT][BN_DOWNSAMPLE] {key} "
                        f"module={name} "
                        f"shape={tuple(tensor.shape)}"
                    )
        # print("[DEBUG] payload tensor keys:", payload.tensors.keys())
        # payload.add_tensor("model.input", x.detach())
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
        except RuntimeError:
            missing = [
                k for k in LENET_REQUIRED_KEYS
                if k not in resolver.sources or resolver.sources[k] == "missing"
            ]

            explain_missing_keys(self.model, missing)

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
