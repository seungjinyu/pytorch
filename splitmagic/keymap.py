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

    bn_global_idx = 0
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

                print(
                    f"[KEY_ASSIGN] node={node} attr={attr} "
                    f"key={input_item.get('jin_key')} "
                    f"shape={tuple(input_item['shape'])}"
                )
                mapped.append(input_item)

                print(
                    f"[KEYMAP][CONV_INPUT] "
                    f"key={input_item['jin_key']} "
                    f"shape={input_item['shape']}"
                )
                
                weight_item = dict(item)
                weight_item["jin_key"] = f"conv2d:{idx}:weight"
                weight_item["global_idx"] = global_idx

                print(
                    f"[KEY_ASSIGN] node={node} attr={attr} "
                    f"key={weight_item.get('jin_key')} "
                    f"shape={tuple(weight_item['shape'])}"
                )
                mapped.append(weight_item)

                counters["conv2d"] += 1
                global_idx += 1
                continue
        elif "NativeBatchNormBackward" in node:
            if attr == "_saved_input":
                current_bn_idx = bn_global_idx
                bn_global_idx += 1

                jin_key = f"batchnorm:{current_bn_idx}:input"

            elif attr == "_saved_running_mean":
                if current_bn_idx is None:
                    continue
                jin_key = f"batchnorm:{current_bn_idx}:running_mean"

            elif attr == "_saved_running_var":
                if current_bn_idx is None:
                    continue
                jin_key = f"batchnorm:{current_bn_idx}:running_var"

            elif attr == "_saved_weight":
                if current_bn_idx is None:
                    continue
                jin_key = f"batchnorm:{current_bn_idx}:weight"

            elif attr == "_saved_result1":
                jin_key = f"batchnorm:{current_bn_idx}:result1"

            elif attr == "_saved_result2":
                jin_key = f"batchnorm:{current_bn_idx}:result2"

            else:
                continue
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

        print(
            f"[KEY_ASSIGN] node={node} attr= {attr} "
            f"key={new_item.get('jin_key')} "
            f"shape={tuple(new_item['shape'])}"
        )

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


def shape_sig(shape):
    return "x".join(str(x) for x in shape)


def assign_jin_keys_by_autograd_order(items):
    conv_counts = {}
    bn_counts = {}
    relu_i = 0
    addmm_i = 0
    pool_i = 0
    add_i = 0

    bn_global_idx = 0
    current_bn_idx = None

    mapped = []

    current_conv_key = None
    current_bn_key = None

    for item in items:
        node = item["node"]
        attr = item["attr"]
        shape = tuple(item["shape"])

        new_item = dict(item)

        if node == "ConvolutionBackward0":
            if attr == "_saved_input":
                input_sig = shape_sig(shape)
                current_conv_input_sig = input_sig
                current_conv_input_item = new_item
                continue

            elif attr == "_saved_weight":
                weight_sig = shape_sig(shape)

                # previous saved_input item과 weight를 묶어서 key 생성
                input_sig = shape_sig(tuple(current_conv_input_item["shape"]))
                sig = f"{input_sig}:{weight_sig}"

                idx = conv_counts.get(sig, 0)
                conv_counts[sig] = idx + 1

                current_conv_input_item["jin_key"] = (
                    f"conv2d:{input_sig}:{weight_sig}:{idx}:input"
                )

                print(
                    f"[KEY_ASSIGN] node={node} attr={attr} "
                    f"key={current_conv_input_item.get('jin_key')} "
                    f"shape={tuple(current_conv_input_item['shape'])}"
                )
                mapped.append(current_conv_input_item)

                print(
                    f"[KEY_ASSIGN] node={node} attr={attr} "
                    f"key={new_item.get('jin_key')} "
                    f"shape={tuple(new_item['shape'])}"
                )
                new_item["jin_key"] = f"conv2d:{idx}:weight"
                mapped.append(new_item)
        elif node == "NativeBatchNormBackward0":

            if attr == "_saved_input":
                current_bn_idx = bn_global_idx
                bn_global_idx += 1

                new_item["jin_key"] = (
                    f"batchnorm:{current_bn_idx}:input"
                )

                mapped.append(new_item)

            elif current_bn_idx is not None:

                if attr == "_saved_running_mean":
                    new_item["jin_key"] = (
                        f"batchnorm:{current_bn_idx}:running_mean"
                    )

                elif attr == "_saved_running_var":
                    new_item["jin_key"] = (
                        f"batchnorm:{current_bn_idx}:running_var"
                    )

                elif attr == "_saved_weight":
                    new_item["jin_key"] = (
                        f"batchnorm:{current_bn_idx}:weight"
                    )

                elif attr == "_saved_result1":
                    new_item["jin_key"] = (
                        f"batchnorm:{current_bn_idx}:result1"
                    )

                elif attr == "_saved_result2":
                    new_item["jin_key"] = (
                        f"batchnorm:{current_bn_idx}:result2"
                    )

                else:
                    continue

                mapped.append(new_item)

        elif node == "ReluBackward0":
            if attr == "_saved_result":
                new_item["jin_key"] = f"relu:{relu_i}:out"
                relu_i += 1

                print(
                    f"[KEY_ASSIGN] node={node} attr={attr} "
                    f"key={new_item.get('jin_key')} "
                    f"shape={tuple(new_item['shape'])}"
                )
                mapped.append(new_item)

        elif node == "AddmmBackward0":
            if attr == "_saved_mat1":
                new_item["jin_key"] = f"addmm:{addmm_i}:mat1"

                print(
                    f"[KEY_ASSIGN] node={node} attr={attr} "
                    f"key={new_item.get('jin_key')} "
                    f"shape={tuple(new_item['shape'])}"
                )
                mapped.append(new_item)

            elif attr == "_saved_mat2":
                new_item["jin_key"] = f"addmm:{addmm_i}:mat2"
                addmm_i += 1

                print(
                    f"[KEY_ASSIGN] node={node} attr={attr} "
                    f"key={new_item.get('jin_key')} "
                    f"shape={tuple(new_item['shape'])}"
                )
                mapped.append(new_item)

        elif "MaxPool" in node:
            if attr == "_saved_self":
                new_item["jin_key"] = f"maxpool2d:{pool_i}:input"

                print(
                    f"[KEY_ASSIGN] node={node} attr={attr} "
                    f"key={new_item.get('jin_key')} "
                    f"shape={tuple(new_item['shape'])}"
                )
                mapped.append(new_item)

            elif attr in ("_saved_indices", "_saved_result1"):
                new_item["jin_key"] = f"maxpool2d:{pool_i}:indices"
                pool_i += 1

                print(
                    f"[KEY_ASSIGN] node={node} attr={attr} "
                    f"key={new_item.get('jin_key')} "
                    f"shape={tuple(new_item['shape'])}"
                )
                mapped.append(new_item)
                
        elif node == "AddBackward0":

            if attr == "_saved_self":
                new_item["jin_key"] = f"add:{add_i}:self"

                print(
                    f"[KEY_ASSIGN] node={node} attr={attr} "
                    f"key={new_item.get('jin_key')} "
                    f"shape={tuple(new_item['shape'])}"
                )
                mapped.append(new_item)

            elif attr == "_saved_other":
                new_item["jin_key"] = f"add:{add_i}:other"

                print(
                    f"[KEY_ASSIGN] node={node} attr={attr} "
                    f"key={new_item.get('jin_key')} "
                    f"shape={tuple(new_item['shape'])}"
                )
                mapped.append(new_item)
                add_i += 1


    return mapped