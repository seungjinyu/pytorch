def read_jin1_keys(path):
    import struct

    keys = []

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"JIN1":
            raise ValueError(f"Invalid JIN1 file: {path}")

        n_tensors = struct.unpack("<I", f.read(4))[0]

        for _ in range(n_tensors):
            key_len = struct.unpack("<I", f.read(4))[0]
            key = f.read(key_len).decode("utf-8")
            keys.append(key)

            dtype_id = struct.unpack("<B", f.read(1))[0]
            ndim = struct.unpack("<B", f.read(1))[0]

            shape = []
            for _ in range(ndim):
                shape.append(struct.unpack("<q", f.read(8))[0])

            raw_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(raw_len, 1)

    return set(keys)       

class SavedTensorResolver:
    def __init__(self, payload, local_keys = None, payload_path=None):
        self.payload = payload
        self.local_keys = local_keys
        self.sources = {}

        self.alias_map = getattr(self.payload, "meta", {}).get("alias_map", {})

        self.jin1_keys = set()
        if payload_path is not None:
            self.jin1_keys = read_jin1_keys(payload_path)

    def has_payload(self, key):
        lookup_key = self.alias(key)
        return (
            lookup_key in self.payload.tensors or lookup_key in self.jin1_keys
        )
    def is_local(self, key):
        return key in self.local_keys
    
    def can_recompute(self, key):
        # not implemented yet
        return False
    def alias(self, key):
        return self.alias_map.get(key, key)
    def resolve(self, key):
        lookup_key = self.alias(key)

        if lookup_key != key:
            print(f"[ALIAS_RESOLVE] {key} -> {lookup_key}")

        if lookup_key in self.payload.tensors:
            self.sources[key] = "payload_python"
            return self.payload.tensors[lookup_key], "payload_python"

        if lookup_key in self.jin1_keys:
            self.sources[key] = "payload_jin1"
            return None, "payload_jin1"

        if self.is_local(key):
            self.sources[key] = "local"
            return None, "local"

        self.sources[key] = "missing"
        raise RuntimeError(f"Missing saved tensor: {key}")
    
    def check_required(self, required_keys):
        resolved = []
        missing = []

        for key in required_keys:
            try:
                _, source = self.resolve(key)
                resolved.append((key, source))
            except RuntimeError:
                missing.append(key)

        # print("[RESOLVER][RESOLVED]")
        # for key, source in resolved:
        #     print(f"  {key}: {source}")

        print("[RESOLVER][MISSING]")
        for key in missing:
            if len(key) == 0:
                print(f"Nothing missing")
            else :
                print(f"  {key}")
            
        if missing:
            raise RuntimeError(
                "Missing required saved tensors: "
                + ", ".join(missing)
            )

        return resolved


def read_jin1_tensor(path, target_key):
    import struct
    import torch
    import numpy as np

    id_to_dtype = {
        0: torch.float32,
        1: torch.float64,
        2: torch.int64,
        3: torch.int32,
        4: torch.int16,
        5: torch.int8,
        6: torch.uint8,
        7: torch.bool,
    }

    torch_to_numpy = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"JIN1":
            raise ValueError(f"Invalid JIN1 file: {path}")

        n_tensors = struct.unpack("<I", f.read(4))[0]

        for _ in range(n_tensors):
            key_len = struct.unpack("<I", f.read(4))[0]
            key = f.read(key_len).decode("utf-8")

            dtype_id = struct.unpack("<B", f.read(1))[0]
            ndim = struct.unpack("<B", f.read(1))[0]

            shape = []
            for _ in range(ndim):
                shape.append(struct.unpack("<q", f.read(8))[0])

            raw_len = struct.unpack("<Q", f.read(8))[0]

            if key == target_key:
                raw = f.read(raw_len)

                torch_dtype = id_to_dtype[dtype_id]
                np_dtype = torch_to_numpy[torch_dtype]

                arr = np.frombuffer(raw, dtype=np_dtype).copy()
                tensor = torch.from_numpy(arr).reshape(shape)

                return tensor

            else:
                f.seek(raw_len, 1)

    raise KeyError(f"Key not found in JIN1: {target_key}")


def read_jin1_payload(path):
    import struct
    import torch
    import numpy as np
    import time 

    from splitmagic.payload import Payload

    id_to_dtype = {
        0: torch.float32,
        1: torch.float64,
        2: torch.int64,
        3: torch.int32,
        4: torch.int16,
        5: torch.int8,
        6: torch.uint8,
        7: torch.bool,
    }

    torch_to_numpy = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }

    profile_t0 = time.perf_counter()

    header_ms = 0.0
    meta_ms = 0.0
    raw_read_ms = 0.0
    tensor_create_ms = 0.0
    payload_add_ms = 0.0

    payload = Payload()

    with open(path, "rb") as f:
        t0 = time.perf_counter()

        magic = f.read(4)

        if magic != b"JIN1":
            raise ValueError(f"Invalid JIN1 file: {path}")

        n_tensors = struct.unpack("<I", f.read(4))[0]
        t1 = time.perf_counter()

        header_ms += (t1-t0) * 1000

        for _ in range(n_tensors):

            t0 = time.perf_counter()

            key_len = struct.unpack("<I", f.read(4))[0]
            key = f.read(key_len).decode("utf-8")

            dtype_id = struct.unpack("<B", f.read(1))[0]
            ndim = struct.unpack("<B", f.read(1))[0]

            shape = []

            for _ in range(ndim):
                shape.append(struct.unpack("<q", f.read(8))[0])

            raw_len = struct.unpack("<Q", f.read(8))[0]
            t1 = time.perf_counter()

            meta_ms += (t1 - t0) * 1000

            t0 = time.perf_counter()
            raw = f.read(raw_len)
            t1 = time.perf_counter()
            raw_read_ms += (t1 - t0) * 1000

            t0 = time.perf_counter()

            torch_dtype = id_to_dtype[dtype_id]
            np_dtype = torch_to_numpy[torch_dtype]

            arr = np.frombuffer(raw, dtype=np_dtype).copy()

            tensor = torch.from_numpy(arr).reshape(shape)

            t1 = time.perf_counter()
            tensor_create_ms += (t1 - t0) * 1000

            t0 = time.perf_counter()
            payload.add_tensor(key, tensor)
            t1 = time.perf_counter()
            payload_add_ms += (t1 - t0) * 1000

    profile_t1 = time.perf_counter()
    total_ms = (profile_t1 - profile_t0) * 1000

    print(
        f"[Payload][READ_JIN1_PROFILE] "
        f"num_tensors={len(payload.tensors)} "
        f"header_ms={header_ms:.3f} "
        f"meta_ms={meta_ms:.3f} "
        f"raw_read_ms={raw_read_ms:.3f} "
        f"tensor_create_ms={tensor_create_ms:.3f} "
        f"payload_add_ms={payload_add_ms:.3f} "
        f"total_read_jin1_ms={total_ms:.3f}",
        flush=True,
    )
    payload.print_add_tensor_profile(
        prefix="[Payload][READ_JIN1_ADD_TENSOR_PROFILE]"
    )

    return payload

def read_jin1_payload_bytes(payload_bytes):
    import io
    import struct
    import torch
    import numpy as np

    from splitmagic.payload import Payload

    id_to_dtype = {
        0: torch.float32,
        1: torch.float64,
        2: torch.int64,
        3: torch.int32,
        4: torch.int16,
        5: torch.int8,
        6: torch.uint8,
        7: torch.bool,
    }

    torch_to_numpy = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }

    payload = Payload()

    f = io.BytesIO(payload_bytes)

    magic = f.read(4)
    if magic != b"JIN1":
        raise ValueError("Invalid JIN1 bytes")

    n_tensors = struct.unpack("<I", f.read(4))[0]

    for _ in range(n_tensors):
        key_len = struct.unpack("<I", f.read(4))[0]
        key = f.read(key_len).decode("utf-8")

        dtype_id = struct.unpack("<B", f.read(1))[0]
        ndim = struct.unpack("<B", f.read(1))[0]

        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack("<q", f.read(8))[0])

        raw_len = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(raw_len)

        torch_dtype = id_to_dtype[dtype_id]
        np_dtype = torch_to_numpy[torch_dtype]

        arr = np.frombuffer(raw, dtype=np_dtype).copy()
        tensor = torch.from_numpy(arr).reshape(shape)

        payload.tensors[key] = tensor

    return payload