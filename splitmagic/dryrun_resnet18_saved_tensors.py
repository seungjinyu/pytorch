import torch
import torchvision.models as models
from collections import defaultdict


def is_tensor_like(x):
    return isinstance(x, torch.Tensor)


def tensor_info(t):
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "requires_grad": bool(t.requires_grad),
    }


def walk_backward_graph(root_fn):
    """
    Traverse autograd backward graph from loss.grad_fn.
    """
    seen = set()
    order = []

    def dfs(fn, depth=0):
        if fn is None:
            return

        fid = id(fn)
        if fid in seen:
            return

        seen.add(fid)
        order.append((fn, depth))

        for next_fn, input_nr in fn.next_functions:
            dfs(next_fn, depth + 1)

    dfs(root_fn)
    return order


def inspect_saved_tensors(fn):
    """
    Read private _saved_* fields from autograd Function node.
    """
    saved = {}

    for name in dir(fn):
        if not name.startswith("_saved_"):
            continue

        try:
            value = getattr(fn, name)
        except RuntimeError as e:
            saved[name] = f"<RuntimeError: {e}>"
            continue
        except Exception as e:
            saved[name] = f"<Error: {type(e).__name__}: {e}>"
            continue

        if is_tensor_like(value):
            saved[name] = tensor_info(value)

        elif isinstance(value, (tuple, list)):
            arr = []
            for item in value:
                if is_tensor_like(item):
                    arr.append(tensor_info(item))
                else:
                    arr.append(repr(item))
            saved[name] = arr

        else:
            saved[name] = repr(value)

    return saved


def normalize_backward_name(name):
    """
    PyTorch backward node name -> rough op name.
    나중에 C++ overwrite key 만들 때 여기 규칙을 맞추면 됨.
    """
    if name.startswith("ConvolutionBackward"):
        return "conv2d"
    if name.startswith("ReluBackward"):
        return "relu"
    if name.startswith("MaxPool2DWithIndicesBackward"):
        return "maxpool2d"
    if name.startswith("AddmmBackward"):
        return "addmm"
    if name.startswith("NativeBatchNormBackward"):
        return "batchnorm"
    if name.startswith("AddBackward"):
        return "add"
    if name.startswith("MeanBackward"):
        return "mean"
    if name.startswith("LogSoftmaxBackward"):
        return "logsoftmax"
    if name.startswith("NllLossBackward"):
        return "nllloss"
    return name


def main():
    torch.manual_seed(0)

    device = "cpu"

    model = models.resnet18(num_classes=10)
    model.eval()
    model.to(device)

    # ResNet18 ReLU inplace 끄는 게 splitmagic 디버깅에 훨씬 안전함
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False

    x = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
    y = torch.randint(0, 10, (2,), device=device)

    criterion = torch.nn.CrossEntropyLoss()

    logits = model(x)
    loss = criterion(logits, y)

    print("loss:", loss.item())
    print("loss.grad_fn:", type(loss.grad_fn).__name__)

    graph = walk_backward_graph(loss.grad_fn)

    counters = defaultdict(int)

    print("\n================ BACKWARD DRY RUN ================\n")

    for global_idx, (fn, depth) in enumerate(graph):
        bwd_name = type(fn).__name__
        op = normalize_backward_name(bwd_name)

        local_idx = counters[op]
        counters[op] += 1

        saved = inspect_saved_tensors(fn)

        indent = "  " * depth
        key_prefix = f"{op}:{local_idx}"

        print(f"{indent}[{global_idx:03d}] {bwd_name}")
        print(f"{indent}     key_prefix = {key_prefix}")

        if saved:
            for k, v in saved.items():
                print(f"{indent}     {k}: {v}")
        else:
            print(f"{indent}     saved: <none>")

        print()

    print("\n================ OP COUNTS ================\n")
    for op, count in counters.items():
        print(f"{op}: {count}")


if __name__ == "__main__":
    main()