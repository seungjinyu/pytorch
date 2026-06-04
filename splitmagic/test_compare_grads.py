import torch


BASE = "/tmp/baseline_grads.pt"
SPLIT = "/tmp/split_grads.pt"

ATOL = 1e-6


def main():
    a = torch.load(BASE, map_location="cpu")
    b = torch.load(SPLIT, map_location="cpu")

    all_keys = sorted(set(a.keys()) | set(b.keys()))

    same = True

    for k in all_keys:
        if k not in a:
            print(f"[ONLY_SPLIT] {k}")
            same = False
            continue

        if k not in b:
            print(f"[ONLY_BASELINE] {k}")
            same = False
            continue

        if a[k].shape != b[k].shape:
            print(f"[SHAPE_DIFF] {k} base={a[k].shape} split={b[k].shape}")
            same = False
            continue

        diff = (a[k] - b[k]).abs()

        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        ok = max_diff <= ATOL

        print(
            f"{k:45s} "
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