import torch


class Payload:
    def __init__(self):
        self.tensors = {}
        self.meta = {}

    def add_tensor(self, key, tensor):
        self.tensors[key] = tensor.detach().cpu().clone()

    def add_meta(self, key, value):
        self.meta[key] = value

    def summary(self):
        print("=== Payload Summary ===")
        print("Tensors:")
        for k, v in self.tensors.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")

        print("Meta:")
        for k, v in self.meta.items():
            print(f"  {k}: {v}")

    def save_jin1(self, path):
        import struct

        dtype_map = {
            torch.float32: 0,
            torch.float64: 1,
            torch.int64: 2,
            torch.int32: 3,
            torch.int16: 4,
            torch.int8: 5,
            torch.uint8: 6,
            torch.bool: 7,
        }

        with open(path, "wb") as f:

            f.write(b"JIN1")
            f.write(struct.pack("<I", len(self.tensors)))

            for key, tensor in self.tensors.items():
                t = tensor.detach().cpu().contiguous()

                if t.dtype not in dtype_map:
                    raise TypeError(f"Unsupported dtype: {t.dtype}")

                key_bytes = key.encode("utf-8")
                f.write(struct.pack("<I", len(key_bytes)))
                f.write(key_bytes)

                f.write(struct.pack("<B", dtype_map[t.dtype]))
                f.write(struct.pack("<B", t.dim()))

                for s in t.shape:
                    f.write(struct.pack("<q", int(s)))

                raw = t.numpy().tobytes()
                f.write(struct.pack("<Q", len(raw)))
                f.write(raw)
            
def payload_from_report(report, matches):
    payload = Payload()

    for saved, match in zip(report.saved_tensors, matches):
        key = match["key"]

        payload.add_tensor(key, saved.tensor)
        payload.add_meta(key, {
            "index": saved.index,
            "shape": saved.shape,
            "dtype": saved.dtype,
            "requires_grad": saved.requires_grad,
            "candidates": match["candidates"],
        })

    return payload

def payload_from_saved_attrs(items):
    payload = Payload()

    for i, item in enumerate(items):

        key = f"{item['node']}.{item['attr']}:{i}"

        payload.add_tensor(key, item["tensor"])
        payload.add_meta(key, {
            "node":item["node"],
            "attr":item["attr"],
            "shape": item["shape"],
            "dtype": item["dtype"],
            "requires_grad": item["requires_grad"],
        })

    return payload

def payload_from_jin_items(items):
    payload = Payload()

    for item in items:
        # key = item["jin_key"]
        tensor = item["tensor"]
        graph_key = item.get("graph_key")

        if graph_key is None:
            continue

        payload.add_tensor(graph_key, tensor)

        payload.add_meta(graph_key, {
            "node": item["node"],
            "attr": item["attr"],
            "shape": item["shape"],
            "dtype": str(item["dtype"]),
            "requires_grad": item["requires_grad"],
            "graph_key": graph_key,
            "legacy_jin_key": item.get("jin_key"),
        })

    return payload

