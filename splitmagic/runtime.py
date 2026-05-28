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

    # model input 쪽 conv input은 우선 send 유지
    # 일반적으로 가장 마지막 conv index가 실제 model input일 가능성이 높음.
    # 정확한 자동 판별은 shape/meta 기반으로 나중에 개선.
    if key.startswith("conv2d:") and key.endswith(":input"):
        return "recompute_or_send"

    # Recomputable candidates
    if key.startswith("relu:") and key.endswith(":out"):
        return "recompute"

    if key.startswith("addmm:") and key.endswith(":mat1"):
        return "recompute"

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

        print("[A][POLICY]", policy_meta)

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
