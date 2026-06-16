import torch
import torch.fx as fx
import torch.nn as nn

SUPPORTED_RECOMPUTE_TYPES = {
    "Conv2d",
    "BatchNorm2d",
    "ReLU",
    "MaxPool2d",
    "Linear",
    "flatten",
    "view",
    "reshape",
    
}

def build_fx_node_list(model):
    gm = fx.symbolic_trace(model)

    nodes = []
    module_map = dict(model.named_modules())

    for node in gm.graph.nodes:
        info = {
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "args": [a.name for a in node.args if hasattr(a, "name")],
            "type": None,
        }

        if node.op == "call_module":
            mod = module_map.get(node.target, None)
            if mod is not None:
                info["type"] = mod.__class__.__name__

        elif node.op == "call_function":
            info["type"] = getattr(node.target, "__name__", str(node.target))

        elif node.op == "call_method":
            info["type"] = str(node.target)

        nodes.append(info)

    return nodes


def print_fx_graph(model):
    nodes = build_fx_node_list(model)

    print("[FX][GRAPH]")
    for i, n in enumerate(nodes):
        print(
            f"  [{i}] name={n['name']} "
            f"op={n['op']} "
            f"type={n['type']} "
            f"target={n['target']} "
            f"args={n['args']}"
        )


def build_lenet_key_trace(model):
    """
    v0: LeNet/CNN sequential-style mapping.
    This does not recompute tensors yet.
    It only explains likely forward dependency path for JIN saved keys.
    """

    nodes = build_fx_node_list(model)

    conv_nodes = []
    relu_nodes = []
    pool_nodes = []
    linear_nodes = []
    flatten_nodes = []

    for n in nodes:
        typ = n["type"]

        if typ == "Conv2d":
            conv_nodes.append(n)
        elif typ == "ReLU" or typ == "relu":
            relu_nodes.append(n)
        elif typ == "MaxPool2d" or typ == "max_pool2d":
            pool_nodes.append(n)
        elif typ == "Linear":
            linear_nodes.append(n)
        elif typ in ("flatten", "view", "reshape"):
            flatten_nodes.append(n)

    traces = {}

    # Conv saved input
    for i, n in enumerate(conv_nodes):
        traces[f"conv2d:{i}:input"] = {
            "meaning": f"input to Conv2d #{i}",
            "producer_hint": n["args"][0] if n["args"] else "model input",
            "path_hint": f"forward value feeding {n['name']}",
        }

    # ReLU saved output
    for i, n in enumerate(relu_nodes):
        traces[f"relu:{i}:out"] = {
            "meaning": f"output of ReLU #{i}",
            "producer_hint": n["name"],
            "path_hint": f"{n['name']} = ReLU({n['args'][0] if n['args'] else '?'})",
        }

    # MaxPool indices/output relation
    for i, n in enumerate(pool_nodes):
        traces[f"maxpool2d:{i}:indices"] = {
            "meaning": f"indices produced by MaxPool2d #{i}",
            "producer_hint": n["name"],
            "path_hint": f"{n['name']} = MaxPool2d({n['args'][0] if n['args'] else '?'})",
            "note": "indices are usually safer to send than recompute",
        }
        traces[f"maxpool2d:{i}:input"] = {
            "meaning": f"input to MaxPool2d #{i}",
            "producer_hint": n["args"][0] if n["args"] else "?",
            "path_hint": f"value feeding {n['name']}",
        }

    # Linear/Addmm saved mat1
    for i, n in enumerate(linear_nodes):
        traces[f"addmm:{i}:mat1"] = {
            "meaning": f"input activation to Linear #{i}",
            "producer_hint": n["args"][0] if n["args"] else "?",
            "path_hint": f"value feeding {n['name']}",
        }

    return traces

def build_fx_maps(model):
    gm = fx.symbolic_trace(model)

    node_map = {}
    module_map = dict(model.named_modules())

    for node in gm.graph.nodes:
        typ = None

        if node.op == "call_module":
            mod = module_map.get(node.target, None)
            if mod is not None:
                typ = mod.__class__.__name__
        elif node.op == "call_function":
            typ = getattr(node.target, "__name__", str(node.target))
        elif node.op == "call_method":
            typ = str(node.target)

        node_map[node.name] = {
            "node": node,
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "type": typ,
            "args": [
                a.name for a in node.args
                if hasattr(a, "name")
            ],
        }

    return node_map

def trace_fx_node_recursive(node_map, node_name, depth=0, seen=None):
    if seen is None:
        seen = set()

    indent = "  " * depth

    if node_name in seen:
        print(f"{indent}{node_name} <already seen>")
        return

    seen.add(node_name)

    if node_name not in node_map:
        print(f"{indent}{node_name} <unknown>")
        return

    info = node_map[node_name]

    print(
        f"{indent}{info['name']} "
        f"(op={info['op']}, type={info['type']}, target={info['target']})"
    )

    for arg_name in info["args"]:
        trace_fx_node_recursive(
            node_map,
            arg_name,
            depth + 1,
            seen,
        )

def build_jin_key_to_fx_node(model):
    nodes = build_fx_node_list(model)

    conv_nodes = []
    relu_nodes = []
    pool_nodes = []
    linear_nodes = []
    bn_nodes = []

    for n in nodes:
        if n["type"] == "Conv2d":
            conv_nodes.append(n)
        elif n["type"] in ("ReLU", "relu"):
            relu_nodes.append(n)
        elif n["type"] in ("MaxPool2d", "max_pool2d"):
            pool_nodes.append(n)
        elif n["type"] == "Linear":
            linear_nodes.append(n)
        elif n["type"] == "BatchNorm2d":
            bn_nodes.append(n)

    # IMPORTANT:
    # Autograd saved tensors are collected in backward graph order.
    # FX nodes are in forward order.
    # So JIN indices must be matched to reversed FX op lists.
    conv_nodes = list(reversed(conv_nodes))
    relu_nodes = list(reversed(relu_nodes))
    pool_nodes = list(reversed(pool_nodes))
    linear_nodes = list(reversed(linear_nodes))
    bn_nodes = list(reversed(bn_nodes))

    mapping = {}

    for i, n in enumerate(conv_nodes):
        if n["args"]:
            mapping[f"graph:conv:{i}:input"] = n["args"][0]

    for i, n in enumerate(relu_nodes):
        mapping[f"graph:relu:{i}:result"] = n["name"]

    for i, n in enumerate(pool_nodes):
        mapping[f"graph:maxpool2d:{i}:indices"] = n["name"]
        if n["args"]:
            mapping[f"graph:maxpool2d:{i}:input"] = n["args"][0]

    for i, n in enumerate(linear_nodes):
        if n["args"]:
            mapping[f"graph:addmm:{i}:mat1"] = n["args"][0]
    for i, n in enumerate(bn_nodes):
        if n["args"]:
            mapping[f"graph:bn:{i}:input"] = n["args"][0]

        # mapping[f"graph:bn:{i}:result1"] = n["target"]
        # mapping[f"graph:bn:{i}:result2"] = n["target"]

        # mapping[f"graph:bn:{i}:running_mean"] = n["target"]
        # mapping[f"graph:bn:{i}:running_var"] = n["target"]
        # mapping[f"graph:bn:{i}:weight"] = n["target"]

    return mapping

def explain_jin_key(model, key):
    node_map = build_fx_maps(model)
    key_to_node = build_jin_key_to_fx_node(model)

    print(f"[FX-TRACE] {key}")

    if key not in key_to_node:
        print("  no FX producer mapping found")
        return None

    node_name = key_to_node[key]

    print(f"  mapped_fx_node: {node_name}")
    print("  dependency_tree:")

    trace_fx_node_recursive(
        node_map,
        node_name,
        depth=2,
    )

    return node_name


def explain_missing_keys(model, missing_keys):
    print_fx_graph(model)

    print("[FX-TRACE][MISSING_KEYS]")
    for key in missing_keys:
        explain_jin_key(model, key)

def can_recompute_fx_node(node_map, node_name, available_nodes, depth=0, seen=None):
    if seen is None:
        seen = set()

    if node_name in available_nodes:
        return True, []

    if node_name in seen:
        return True, []

    seen.add(node_name)

    if node_name not in node_map:
        return False, [f"unknown node: {node_name}"]

    info = node_map[node_name]

    if info["op"] == "placeholder":
        if node_name in available_nodes:
            return True, []
        return False, [f"missing model input: {node_name}"]

    if info["op"] == "output":
        return False, [f"cannot recompute output node directly: {node_name}"]

    if info["type"] not in SUPPORTED_RECOMPUTE_TYPES:
        return False, [f"unsupported op: {node_name} type={info['type']}"]

    reasons = []

    for arg_name in info["args"]:
        ok, why = can_recompute_fx_node(
            node_map,
            arg_name,
            available_nodes,
            depth + 1,
            seen,
        )
        if not ok:
            reasons.extend(why)

    return len(reasons) == 0, reasons

def can_recompute_jin_key(model, key, available_nodes=None):
    available_nodes = set(available_nodes or [])

    node_map = build_fx_maps(model)
    key_to_node = build_jin_key_to_fx_node(model)

    if key not in key_to_node:
        return False, [f"no FX mapping for key: {key}"]

    node_name = key_to_node[key]

    ok, reasons = can_recompute_fx_node(
        node_map=node_map,
        node_name=node_name,
        available_nodes=available_nodes,
    )

    return ok, reasons

def build_available_nodes_from_payload(model, payload_keys):
    key_to_node = build_jin_key_to_fx_node(model)

    available = set()

    available.add("x")

    for key in payload_keys:
        if key in key_to_node:
            available.add(key_to_node[key])

    return available

def find_nearest_available_start(node_map, node_name, available_nodes, seen=None):
    if seen is None:
        seen = set()

    if node_name in available_nodes:
        return node_name

    if node_name in seen:
        return None

    seen.add(node_name)

    if node_name not in node_map:
        return None

    info = node_map[node_name]

    for arg_name in info["args"]:
        start = find_nearest_available_start(
            node_map,
            arg_name,
            available_nodes,
            seen,
        )
        if start is not None:
            return start

    return None

def build_path_from_start_to_node(node_map, start_node, target_node):
    path = []

    def dfs(cur):
        path.append(cur)

        if cur == start_node:
            return True

        if cur not in node_map:
            path.pop()
            return False

        for arg_name in node_map[cur]["args"]:
            if dfs(arg_name):
                return True

        path.pop()
        return False

    ok = dfs(target_node)

    if not ok:
        return []

    return list(reversed(path))

def analyze_recompute_for_missing_keys(model, missing_keys, available_nodes=None):
    available_nodes = set(available_nodes or [])

    node_map = build_fx_maps(model)
    key_to_node = build_jin_key_to_fx_node(model)

    print("[FX-RECOMPUTE][ANALYZE]")

    results = {}

    for key in missing_keys:
        if key not in key_to_node:
            results[key] = {
                "can_recompute": False,
                "nearest_start": None,
                "path": [],
                "reasons": [f"no FX mapping for key: {key}"],
            }
            print(f"  {key}: can_recompute=NO")
            print(f"    - no FX mapping for key")
            continue

        target_node = key_to_node[key]

        ok, reasons = can_recompute_fx_node(
            node_map=node_map,
            node_name=target_node,
            available_nodes=available_nodes,
        )

        nearest_start = find_nearest_available_start(
            node_map=node_map,
            node_name=target_node,
            available_nodes=available_nodes,
        )

        path = []
        if nearest_start is not None:
            path = build_path_from_start_to_node(
                node_map=node_map,
                start_node=nearest_start,
                target_node=target_node,
            )

        results[key] = {
            "can_recompute": ok,
            "nearest_start": nearest_start,
            "path": path,
            "reasons": reasons,
        }

        status = "YES" if ok else "NO"
        # print(f"  {key}: can_recompute={status}")
        # print(f"    nearest_start={nearest_start}")
        # print(f"    path={' -> '.join(path) if path else '<none>'}")

        for r in reasons:
            print(f"    - {r}")

    return results

def build_available_tensors_from_jin1(model, payload_path, payload_keys, device=None):
    from .resolver import read_jin1_tensor

    key_to_node = build_jin_key_to_fx_node(model)
    available_tensors = {}

    for key in payload_keys:
        if key.endswith(":indices"):
            continue

        if key not in key_to_node:
            continue
        
        node_name = key_to_node[key]

        try:
            t = read_jin1_tensor(payload_path, key)
        except KeyError:
            continue

        if device is not None:
            t = t.to(device)

        available_tensors[node_name] = t

        print(
            f"[FX-AVAILABLE] {key} -> {node_name} "
            f"shape={tuple(t.shape)} dtype={t.dtype}"
        )

    return available_tensors