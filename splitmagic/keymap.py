def assign_jin_keys(items):
    counters = {
        "addmm": 0,
        "conv2d": 0,
        "relu": 0,
        "maxpool2d": 0,
        "batchnorm": 0,
    }

    mapped = []
    global_idx = 0


    for item in items:
        node = item["node"]
        attr = item["attr"]

        jin_key = None
        counter_name = None
        advance = False

        if node == "AddmmBackward0":
            idx = counters["addmm"]

            if attr == "_saved_mat1":
                jin_key = f"addmm:{idx}:mat1"
                counter_name = "addmm"

            elif attr == "_saved_mat2":
                jin_key = f"addmm:{idx}:mat2"
                counter_name = "addmm"
                advance = True

        elif node == "ConvolutionBackward0":
            idx = counters["conv2d"]

            if attr == "_saved_input":
                jin_key = f"conv2d:{idx}:input"
                counter_name = "conv2d"

            elif attr == "_saved_weight":
                jin_key = f"conv2d:{idx}:weight"
                counter_name = "conv2d"
                advance = True

        elif node == "ReluBackward0":
            idx = counters["relu"]

            if attr == "_saved_result":
                jin_key = f"relu:{idx}:out"
                counter_name = "relu"
                advance = True

        elif node == "MaxPool2DWithIndicesBackward0":
            idx = counters["maxpool2d"]

            if attr == "_saved_self":
                jin_key = f"maxpool2d:{idx}:input"
                counter_name = "maxpool2d"

            elif attr == "_saved_result1":
                jin_key = f"maxpool2d:{idx}:indices"
                counter_name = "maxpool2d"
                advance = True

        if jin_key is None:
            continue

        new_item = dict(item)
        new_item["jin_key"] = jin_key
        new_item["global_idx"] = global_idx
        mapped.append(new_item)

        if advance and counter_name is not None:
            counters[counter_name] += 1
            global_idx += 1
    return mapped