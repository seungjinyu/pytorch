def compare_state_dict(name, baseline_sd, split_sd, verbose=True):
    if verbose:
        print(f"\n[Compare] {name}")

    max_all = 0.0
    mean_all = 0.0
    count = 0

    for k in baseline_sd.keys():
        a = baseline_sd[k].detach().cpu()
        b = split_sd[k].detach().cpu()

        diff = (a - b).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        max_all = max(max_all, max_diff)
        mean_all += mean_diff
        count += 1

        if verbose:
            print(
                f"{k:20s} "
                f"max_diff={max_diff:.10f} "
                f"mean_diff={mean_diff:.10f}"
            )

    mean_avg = mean_all / max(count, 1)

    if verbose:
        print(f"\n[Summary] max_all={max_all:.10f}, mean_avg={mean_avg:.10f}")

    return {
        "max_all": max_all,
        "mean_avg": mean_avg,
    }