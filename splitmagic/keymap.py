def shape_sig(shape):
    return "x".join(str(int(s)) for s in shape)

def assign_jin_keys(items):
    counters = {
        "addmm": 0,
        "conv2d": 0,
        "relu": 0,
        "maxpool2d": 0,
        "batchnorm": 0,
    }

    shape_counters = {}
    pending_conv_inputs = []
    conv_sig_counters = {}

    mapped = []
    global_idx = 0

    bn_sig_counters = {}
    current_bn_sig = None
    current_bn_idx = None

    pending_bn = {}


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

        elif "ConvolutionBackward" in node:
            idx = counters["conv2d"]

            if attr == "_saved_input":
                pending_conv_inputs.append(dict(item))
                continue

            elif attr == "_saved_weight":
                if not pending_conv_inputs:
                    continue

                pending_input = pending_conv_inputs.pop(0)

                in_sig = shape_sig(pending_input["shape"])
                w_sig = shape_sig(item["shape"])

                sig = f"{in_sig}:{w_sig}"
                n = conv_sig_counters.get(sig, 0)
                conv_sig_counters[sig] = n + 1

                input_item = dict(pending_input)
                input_item["jin_key"] = f"conv2d:{in_sig}:{w_sig}:{n}:input"
                input_item["global_idx"] = global_idx
                mapped.append(input_item)

                print(
                    f"[KEYMAP][CONV_INPUT] "
                    f"key={input_item['jin_key']} "
                    f"shape={input_item['shape']}"
                )
                
                weight_item = dict(item)
                weight_item["jin_key"] = f"conv2d:{idx}:weight"
                weight_item["global_idx"] = global_idx
                mapped.append(weight_item)

                counters["conv2d"] += 1
                global_idx += 1
                continue
        elif "NativeBatchNormBackward" in node:
            if attr == "_saved_input":
                current_bn_sig = shape_sig(item["shape"])
                current_bn_idx = bn_sig_counters.get(current_bn_sig, 0)
                bn_sig_counters[current_bn_sig] = current_bn_idx + 1

                jin_key = f"batchnorm:{current_bn_sig}:{current_bn_idx}:input"

            elif attr == "_saved_running_mean":
                if current_bn_sig is None:
                    continue
                jin_key = f"batchnorm:{current_bn_sig}:{current_bn_idx}:running_mean"

            elif attr == "_saved_running_var":
                if current_bn_sig is None:
                    continue
                jin_key = f"batchnorm:{current_bn_sig}:{current_bn_idx}:running_var"

            elif attr == "_saved_weight":
                if current_bn_sig is None:
                    continue
                jin_key = f"batchnorm:{current_bn_sig}:{current_bn_idx}:weight"

            elif attr == "_saved_result1":
                if current_bn_sig is None:
                    continue
                jin_key = f"batchnorm:{current_bn_sig}:{current_bn_idx}:result1"

            elif attr == "_saved_result2":
                if current_bn_sig is None:
                    continue
                jin_key = f"batchnorm:{current_bn_sig}:{current_bn_idx}:result2"
                advance = True

            else:
                continue

            counter_name = "batchnorm"

            counter_name = "batchnorm"
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
        print(
            f"[KEYMAP][BN] "
            f"key={new_item['jin_key']} "
            f"shape={new_item['shape']}"
        )

        if advance and counter_name is not None:
            counters[counter_name] += 1
            global_idx += 1
    return mapped