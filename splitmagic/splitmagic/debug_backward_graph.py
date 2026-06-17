import torch

def tensor_info(x):
    if torch.is_tensor(x):
        return {
            "shape": tuple(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
            "mean": float(x.detach().float().mean()) if x.numel() > 0 else 0.0,
        }
    return None


def get_saved_attrs(fn):
    attrs = []

    for name in dir(fn):
        if not name.startswith("_saved_"):
            continue

        try:
            value = getattr(fn, name)
        except Exception as e:
            attrs.append((name, f"<error: {e}>"))
            continue

        if torch.is_tensor(value):
            attrs.append((name, tensor_info(value)))
        elif isinstance(value, (tuple, list)):
            values = []
            for v in value:
                if torch.is_tensor(v):
                    values.append(tensor_info(v))
                else:
                    values.append(repr(v))
            attrs.append((name, values))
        else:
            attrs.append((name, repr(value)))

    return attrs


def dump_backward_graph(loss, max_depth=80):
    seen = set()
    counter = {}

    def next_index(name):
        i = counter.get(name, 0)
        counter[name] = i + 1
        return i

    def walk(fn, depth=0, edge_name="root"):
        if fn is None:
            return

        obj_id = id(fn)
        indent = "  " * depth

        node_name = type(fn).__name__

        if obj_id in seen:
            print(f"{indent}↳ [{edge_name}] {node_name} (seen)")
            return

        seen.add(obj_id)
        node_idx = next_index(node_name)

        print(f"{indent}● [{edge_name}] {node_name} #{node_idx}")

        saved_attrs = get_saved_attrs(fn)

        for attr, info in saved_attrs:
            print(f"{indent}    saved {attr}: {info}")

        if depth >= max_depth:
            print(f"{indent}    ... max_depth reached")
            return

        for i, (next_fn, input_nr) in enumerate(fn.next_functions):
            child_edge = f"next[{i}], input_nr={input_nr}"
            walk(next_fn, depth + 1, child_edge)

    print("========== BACKWARD GRAPH DUMP ==========")
    walk(loss.grad_fn)
    print("========== END BACKWARD GRAPH DUMP ==========")


def dump_loss_backward_graph(model, x, y, criterion):
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = criterion(out, y)

    print("[LOSS]", float(loss.detach()))
    dump_backward_graph(loss)

    return loss