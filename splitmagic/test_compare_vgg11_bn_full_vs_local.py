import torch


def main():
    local = torch.load("vgg11_bn_grads_local.pt")
    nodeb = torch.load("vgg11_bn_grads_full.pt")

    ok = True

    for name in local:
        if name not in nodeb:
            print(f"[MISS] {name}")
            ok = False
            continue

        diff = (local[name] - nodeb[name]).abs()
        close = torch.allclose(local[name], nodeb[name], atol=1e-5, rtol=1e-4)

        print(
            f"{name:40s} "
            f"max_diff={diff.max().item():.8f} "
            f"mean_diff={diff.mean().item():.8f} "
            f"close={close}"
        )

        if not close:
            ok = False

    print()
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()