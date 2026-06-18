import torch

def dump_autograd_graph(loss, max_depth=20, show_saved=True):
    seen = set()

    def walk(fn, depth):

        if fn is None:
            return 

        if id(fn) in seen:
            print("  " * depth + f"{fn.__class__.__name__}")
            return 
        
        seen.add(id(fn))

        print("  " * depth + f"{fn.__class__.__name__}" )

        if show_saved:
            dump_autograd_attrs(fn, depth + 1)

        if depth >= max_depth:
            print("  " * (depth + 1) + "... max depth reached ...")
            return 
        for next_fn, _ in fn.next_functions:
            walk(next_fn, depth + 1)

    walk(loss.grad_fn, 0)

def dump_autograd_attrs(fn, depth):

    indent = " " * depth

    for attr in dir(fn):
        if not attr.startswith("_saved_"):
            continue

        try:
            value = getattr(fn, attr)
        except RuntimeError as e:
            print(indent +f"{attr}: <unavailable: {e}>")
            continue
        except Exception as e:
            print(indent +f"{attr}: <error: {e}>")
            continue
        print(indent + format_saved_attr(attr, value))

    if hasattr(fn, "saved_tensors"):
        for i, t in enumerate(fn.saved_tensors):
            print(f"{indent}Saved Tensor {i}: shape={tuple(t.shape)} dtype={t.dtype}")

def format_saved_attr(name, value):
    if torch.is_tensor(value):
        return (
            f"{name}: Tensor("
            f"shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, "
            f"device={value.device}, "
            f"requires_grad={value.requires_grad}"
            f")"
        )
    if isinstance(value, (list, tuple)):
        parts = []
        for v in value:
            if torch.is_tensor(v):
                parts.append(
                    f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
                )
            else:
                parts.append(repr(v))
        return f"{name}: {type(value).__name__}({', '.join(parts)})"
    return f"{name}: {repr(value)}"

def collect_saved_attrs(loss, tensor_only=True, max_depth=1000000):
    items = []
    seen = set()

    def walk(fn, depth):
        if fn is None:
            return
        if id(fn) in seen:
            return
        seen.add(id(fn))

        node_name = fn.__class__.__name__

        attrs = [a for a in dir(fn) if a.startswith("_saved_")]


        priority = {
            "_saved_input": 0,
            "_saved_weight": 1,
            "_saved_self": 0,
            "_saved_result1": 1,
            "_saved_mat1": 0,
            "_saved_mat2": 1,
            "_saved_result": 0,
        }

        attrs.sort(key=lambda a: priority.get(a, 100))

        for attr in attrs:
            try:
                value = getattr(fn, attr)
                if "AddBackward" in node_name:
                    print("[ADD ATTR]", node_name, attr, type(value))

            except Exception:
                continue

            if tensor_only and not torch.is_tensor(value):
                continue



            if torch.is_tensor(value):

                if "Conv" in node_name or "Convolution" in node_name:
                    print(
                        "[AUTOGRAD NODE]",
                        node_name,
                        attr,
                        tuple(value.shape)
                    )
                if "AddBackward" in node_name:
                    print(
                        "[ADD NODE]",
                        node_name,
                        attr,
                        tuple(value.shape)
                    )


                items.append({
                    "node": node_name,
                    "attr": attr,
                    "shape": tuple(value.shape),
                    "dtype": value.dtype,
                    "device": value.device,
                    "requires_grad": value.requires_grad,
                    "tensor": value.detach().cpu().clone(),
                })

        if depth >= max_depth:
            return

        for next_fn, _ in fn.next_functions:
            walk(next_fn, depth + 1)

    walk(loss.grad_fn, 0)
    return items

def collect_backward_nodes(loss, max_depth=50):
    nodes = []
    seen = set()

    def walk(fn, depth):
        if fn is None:
            return 
        if id(fn) in seen:
            return 
        seen.add(id(fn))

        nodes.append({
            "node":fn,
            "name":fn.__class__.__name__,
            "depth":depth,
        })

        if depth >= max_depth:
            return
        for next_fn, _ in fn.next_functions:
            walk(next_fn, depth + 1)
    walk(loss.grad_fn, 0)
    return nodes