import torch
import time 

class Payload:
    def __init__(self):
        self.tensors = {}
        self.meta = {}

    def add_tensor(self, key, tensor):
        t0 = time.perf_counter()

        self.tensors[key] = tensor.detach().cpu().clone()

        t1 = time.perf_counter()

        if not hasattr(self, "_profile_add_tensor_ms"):
            self._profile_add_tensor_ms = 0.0
            self._profile_add_tensor_count = 0
            self._profile_add_tensor_bytes = 0

        self._profile_add_tensor_ms += (t1 - t0) * 1000
        self._profile_add_tensor_count += 1
        self._profile_add_tensor_bytes += (
            self.tensors[key].numel() * self.tensors[key].element_size()
        )

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

        copy_ms = 0.0
        header_ms = 0.0
        raw_convert_ms = 0.0
        raw_write_ms = 0.0

        with open(path, "wb") as f:

            f.write(b"JIN1")
            f.write(struct.pack("<I", len(self.tensors)))

            for key, tensor in self.tensors.items():
                t0 = time.perf_counter()
                t = tensor.detach().cpu().contiguous()
                t1 = time.perf_counter()
                copy_ms += (t1 - t0) * 1000

                if t.dtype not in dtype_map:
                    raise TypeError(f"Unsupported dtype: {t.dtype}")
                
                t0 = time.perf_counter()
                key_bytes = key.encode("utf-8")
                f.write(struct.pack("<I", len(key_bytes)))
                f.write(key_bytes)

                f.write(struct.pack("<B", dtype_map[t.dtype]))
                f.write(struct.pack("<B", t.dim()))

                for s in t.shape:
                    f.write(struct.pack("<q", int(s)))
                t1 = time.perf_counter()
                header_ms += (t1 - t0) * 1000

                t0 = time.perf_counter()
                raw = t.numpy().tobytes()
                t1 = time.perf_counter()
                raw_convert_ms += (t1 - t0) * 1000

                t0 = time.perf_counter()
                f.write(struct.pack("<Q", len(raw)))
                f.write(raw)
                t1 = time.perf_counter()
                raw_write_ms += (t1-t0) * 1000

        total_profile_ms = copy_ms + header_ms + raw_convert_ms + raw_write_ms

        print(
            f"[Payload][SAVE_JIN1_PROFILE] "
            f"num_tensors={len(self.tensors)} "
            f"copy_ms={copy_ms:.3f} "
            f"header_ms={header_ms:.3f} "
            f"raw_convert_ms={raw_convert_ms:.3f} "
            f"raw_write_ms={raw_write_ms:.3f} "
            f"measured_total_ms={total_profile_ms:.3f}",
            flush=True,
        )

    def to_jin1_bytes(self):
        import io
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

        buf = io.BytesIO()

        buf.write(b"JIN1")
        buf.write(struct.pack("<I", len(self.tensors)))

        for key, tensor in self.tensors.items():
            t = tensor.detach().cpu().contiguous()

            if t.dtype not in dtype_map:
                raise TypeError(f"Unsupported dtype: {t.dtype}")

            key_bytes = key.encode("utf-8")
            buf.write(struct.pack("<I", len(key_bytes)))
            buf.write(key_bytes)

            buf.write(struct.pack("<B", dtype_map[t.dtype]))
            buf.write(struct.pack("<B", t.dim()))

            for s in t.shape:
                buf.write(struct.pack("<q", int(s)))

            raw = t.numpy().tobytes()
            buf.write(struct.pack("<Q", len(raw)))
            buf.write(raw)

        return buf.getvalue()
    
    def print_add_tensor_profile(self, prefix="[Payload][ADD_TENSOR_PROFILE]"):
        ms = getattr(self, "_profile_add_tensor_ms", 0.0)
        count = getattr(self, "_profile_add_tensor_count", 0)
        nbytes = getattr(self, "_profile_add_tensor_bytes", 0)

        print(
            f"{prefix} "
            f"count={count} "
            f"total_mb={nbytes / 1024 / 1024:.3f} "
            f"total_ms={ms:.3f} "
            f"avg_ms={(ms / count) if count else 0.0:.6f}",
            flush=True,
        )
            
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


