# 무엇을 무엇으로 변경할지 

def normalize_key(node, attr):
    return f"{node}.{attr}"

def build_overwrite_plan(payload , node_b_items):
    plan = []

    payload_index = {}

    for key, meta in payload.meta.items():
        if "node" not in meta or "attr" not in meta:
            continue

        short_key = normalize_key(meta["node"], meta["attr"])

        # 같은 종류가 여러 개 있을 수 있으므로 list 로 저장 

        payload_index.setdefault(short_key, []).append({
            "payload_key": key,
            "meta": meta,
        })

    used_count = {}

    for item in node_b_items:
        short_key = normalize_key(item["node"], item["attr"])

        if short_key not in payload_index:
            continue

        idx = used_count.get(short_key, 0)

        if idx >= len(payload_index[short_key]):
            continue

        payload_entry = payload_index[short_key][idx]
        used_count[short_key] = idx + 1

        plan.append({
            "node": item["node"],
            "attr": item["attr"],
            "target_shape": item["shape"],
            "payload_key": payload_entry["payload_key"],
            "payload_shape": payload_entry["meta"]["shape"],
        })

    return plan 

def print_overwrite_plan(plan):
    print("=== Overwrite Plan ===")
    for i,p in enumerate(plan):
        print(
            f"[{i}] "
            f"{p['node']}.{p['attr']} "
            f"target_shape={p['target_shape']} "
            f"<- {p['payload_key']} "
            f"payload_shape={p['payload_shape']}"
        )