import torch


BASE = "./resnet18_baseline_grads.pt"
SPLIT = "./resnet18_split_grads.pt"

ATOL = 1e-5
RTOL = 1e-5


def main():
    base = torch.load(BASE, map_location="cpu")
    split = torch.load(SPLIT, map_location="cpu")

    all_keys = sorted(set(base.keys()) | set(split.keys()))

    same = True

    for k in all_keys:
        if k not in base:
            print(f"[ONLY_SPLIT] {k}")
            same = False
            continue

        if k not in split:
            print(f"[ONLY_BASE] {k}")
            same = False
            continue

        a = base[k]
        b = split[k]

        if a.shape != b.shape:
            print(f"[SHAPE_DIFF] {k} base={tuple(a.shape)} split={tuple(b.shape)}")
            same = False
            continue

        diff = (a - b).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        ok = torch.allclose(a, b, atol=ATOL, rtol=RTOL)

        print(
            f"{k:35s} "
            f"max={max_diff:.10e} "
            f"mean={mean_diff:.10e} "
            f"{'OK' if ok else 'DIFF'}"
        )

        if not ok:
            same = False

    print()
    print("[FINAL SAME]", same)


if __name__ == "__main__":
    main()