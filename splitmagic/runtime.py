# 사용자가 import 하는 입구
# Node A / Node B 역할 선택 
from .graph import collect_saved_attrs
from .payload import payload_from_saved_attrs
from .replay import ReplayEngine

from .keymap import assign_jin_keys
from .payload import payload_from_jin_items 

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
    "conv2d:1:input",
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

    if policy == "custom":
        invalid = optional_keys - OPTIONAL_KEYS
        if invalid:
            raise ValueError(f"Invalid optional keys: {sorted(invalid)}")
        return key in REQUIRED_KEYS or key in optional_keys

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
