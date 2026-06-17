def match_saved_tensors(report):
    matches = []

    for saved in report.saved_tensors:
        candidates = []

        for module in report.module_trace:
            if saved.shape == module["input_shape"]:
                candidates.append(f"{module['name']}.input")

            if saved.shape == module["output_shape"]:
                candidates.append(f"{module['name']}.output")

        best_key = choose_best_key(saved, candidates)

        matches.append({
            "index": saved.index,
            "shape": saved.shape,
            "dtype": saved.dtype,
            "requires_grad": saved.requires_grad,
            "candidates": candidates,
            "key": best_key,
        })

    return matches


def choose_best_key(saved, candidates):
    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        return candidates[0]

    return f"saved_tensor:{saved.index}"