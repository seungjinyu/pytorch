import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
import os 
import time 
import csv

from .payload import  payload_from_jin_items
from .resolver import SavedTensorResolver

from contextlib import contextmanager

from .fx_trace import (
    build_available_tensors_from_payload,
    build_fx_maps,
    build_jin_key_to_fx_node,
    find_nearest_available_start,
    build_path_from_start_to_node,
)
from .recompute import FXRecomputeEngine

from torch.utils._python_dispatch import TorchDispatchMode

ALWAYS_LOCAL_KEYS = set()

@contextmanager
def nvtx_range(name):
    print(f"[NVTX] {name}", flush=True)

    try:
        torch.cuda.nvtx.range_push(name)
    except Exception:
        pass

    try:
        yield
    finally:
        try:
            torch.cuda.nvtx.range_pop()
        except Exception:
            pass

def jin_patch_tensor_from_python(key, tensor, step):
    import ctypes
    import torch
    import numpy as np

    lib = ctypes.CDLL(torch._C.__file__)

    fn = lib.jin_patch_tensor
    fn.argtypes = [
        ctypes.c_char_p,      # key
        ctypes.c_void_p,      # raw data
        ctypes.c_uint64,      # nbytes
        ctypes.c_int32,       # dtype_id
        ctypes.c_void_p,      # shape ptr
        ctypes.c_uint64,      # ndim
        ctypes.c_int64,       # step
    ]
    fn.restype = None

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

    t = tensor.detach().cpu().contiguous()

    if t.dtype not in dtype_map:
        raise TypeError(f"Unsupported dtype: {t.dtype}")

    arr = t.numpy()
    shape_arr = np.array(list(t.shape), dtype=np.int64)

    fn(
        key.encode("utf-8"),
        ctypes.c_void_p(arr.ctypes.data),
        ctypes.c_uint64(arr.nbytes),
        ctypes.c_int32(dtype_map[t.dtype]),
        ctypes.c_void_p(shape_arr.ctypes.data),
        ctypes.c_uint64(t.dim()),
        ctypes.c_int64(step),
    )

def jin_set_payload_bytes_from_python(payload_bytes, step):
    import ctypes
    import torch

    lib = ctypes.CDLL(torch._C.__file__)

    fn = lib.jin_set_payload_bytes
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int64,
    ]
    fn.restype = None

    buf = ctypes.create_string_buffer(payload_bytes)
    fn(
        ctypes.cast(buf, ctypes.c_void_p),
        ctypes.c_uint64(len(payload_bytes)),
        ctypes.c_int64(step),
    )

def jin_patch_payload_bytes_from_python(payload_bytes, step):
    import ctypes
    import torch

    lib = ctypes.CDLL(torch._C.__file__)

    fn = lib.jin_patch_payload_bytes
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_int64,
    ]
    fn.restype = None

    buf = ctypes.create_string_buffer(payload_bytes)
    fn(
        ctypes.cast(buf, ctypes.c_void_p),
        ctypes.c_uint64(len(payload_bytes)),
        ctypes.c_int64(step),
    )

# Find seed keyss
def build_node_to_payload_key(fx_key_map):

    return{
        node_name: key
        for key, node_name in fx_key_map.items()
    }

def find_recompute_seed_keys(
    missing_keys,
    key_to_fx_node,
    fx_path_finder,
):
    seed_nodes = set()

    for key in missing_keys:
        target_node = key_to_fx_node.get(key)
        if target_node is None:
            continue

        start, path = fx_path_finder.find_start_and_path(target_node)

        if start is None or path is None:
            continue

        seed_nodes.add(start)

    node_to_key = build_node_to_payload_key(key_to_fx_node)

    seed_keys = {
        node_to_key[n]
        for n in seed_nodes
        if n in node_to_key
    }

    return seed_keys

def relu_mask_key_for(relu_key):
    # graph:relu:16:result -> graph:relu_mask:16:result
    return relu_key.replace("graph:relu:", "graph:relu_mask:", 1)


def has_relu_mask_for(key, payload):
    if not (key.startswith("graph:relu:") and key.endswith(":result")):
        return False

    mask_key = relu_mask_key_for(key)
    return mask_key in payload.tensors

def is_always_local_key(key):
    if key.startswith("conv2d:") and key.endswith(":weight"):
        return True
    if key.startswith("addmm:") and key.endswith(":mat2"):
        return True
    if key.startswith("graph:conv:") and key.endswith(":weight"):
        return True
    if key.startswith("graph:addmm:") and key.endswith(":mat2"):
        return True
    if key.startswith("graph:mm:") and key.endswith(":mat2"):
        return True
    return False

# read the dryrun plan and be ready to send it in the payload 
def read_dryrun_plan(path="/tmp/jin_dryrun_plan.tsv"):
    plan = []
    
    if not os.path.exists(path):
        return plan 

    with open(path,"r") as f :
        for line in f :
            line = line.rstrip("\n")

            if not line:
                continue 
            
            parts = line.split("\t")
            if len(parts) == 5 :
                row_id, op, idx, suffix, shape = parts 
            elif len(parts) == 4 :
                op, idx, suffix, shape = parts 
                row_id = len(plan)
            else:
                raise ValueError(f"bad dryrun plan line: {line}")
            # row,op,idx,suffix,shape = line.split("\t")

            plan.append({
                "row_id":int(row_id),
                "op":op,
                "idx":int(idx),
                "suffix":suffix,
                "shape": shape,
            })
    return plan

def keys_from_dryrun_plan(plan):
    keys = set()

    for e in plan:
        op = e["op"]
        idx = e["idx"]
        suffix = e["suffix"]

        # local parameter라 안 보내도 되는 것
        if op == "conv" and suffix == "weight":
            continue
        if op == "addmm" and suffix == "mat2":
            continue
        if op == "mm" and suffix == "mat2":
            continue

        keys.add(f"graph:{op}:{idx}:{suffix}")

    return keys

def parse_shape_str(s):
    # "[32,512,4,4]" -> (32,512,4,4)
    s = s.strip()
    s = s.strip("[]")
    if not s:
        return tuple()
    return tuple(int(x) for x in s.split(","))

def get_required_keys_from_plan(plan):
    required = set()

    for e in plan:
        op = e["op"]
        idx = e["idx"]
        suffix = e["suffix"]

        key = f"graph:{op}:{idx}:{suffix}"

        if is_always_local_key(key):
            continue

        required.add(key)

    return required


def get_missing_keys(required_keys, payload, payload_path=None):
    resolver = SavedTensorResolver(
        payload=payload,
        local_keys=ALWAYS_LOCAL_KEYS,
        payload_path=payload_path,
    )

    missing = sorted([
        k for k in required_keys
        if (not is_always_local_key(k))
        and (k not in resolver.sources or resolver.sources[k] == "missing")
    ])

    return missing

def _tensor_mb(x):
    if isinstance(x, torch.Tensor):
        return (
            x.numel()
            * x.element_size()
            / 1024
            / 1024
        )

    if isinstance(x, (tuple, list)):
        return sum(
            _tensor_mb(v)
            for v in x
        )

    if isinstance(x, dict):
        return sum(
            _tensor_mb(v)
            for v in x.values()
        )

    return 0.0


def _tensor_shape(x):
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))

    if isinstance(x, (tuple, list)):
        shapes = []

        for value in x:
            shape = _tensor_shape(value)

            if shape:
                shapes.append(shape)

        return "|".join(shapes)

    if isinstance(x, dict):
        shapes = []

        for value in x.values():
            shape = _tensor_shape(value)

            if shape:
                shapes.append(shape)

        return "|".join(shapes)

    return ""


def _forward_op_type(
    gm: fx.GraphModule,
    node: fx.Node,
):
    if node.op == "call_module":
        module = gm.get_submodule(
            str(node.target)
        )

        mapping = {
            nn.Conv2d: "Conv",
            nn.BatchNorm2d: "BN",
            nn.ReLU: "ReLU",
            nn.MaxPool2d: "MaxPool",
            nn.AdaptiveAvgPool2d: "AdaptiveAvgPool",
            nn.Linear: "Linear",
            nn.Identity: "Identity",
        }

        for module_type, name in mapping.items():
            if isinstance(module, module_type):
                return name

        return type(module).__name__

    if node.op == "call_function":
        target_name = getattr(
            node.target,
            "__name__",
            str(node.target),
        )

        if target_name in (
            "add",
            "add_",
        ):
            return "Add"

        if target_name == "flatten":
            return "Flatten"

        return target_name

    if node.op == "call_method":
        if str(node.target) == "flatten":
            return "Flatten"

        return str(node.target)

    return node.op

class ForwardFXProfiler(fx.Interpreter):
    def __init__(
        self,
        gm,
        csv_path=None,
    ):
        super().__init__(gm)

        self.gm = gm
        self.csv_path = csv_path
        self.rows = []

    def run_node(self, node):
        # placeholder/output은 실제 layer가 아니므로
        # profiling 대상에서 제외
        if node.op in (
            "placeholder",
            "output",
            "get_attr",
        ):
            return super().run_node(node)

        args, kwargs = self.fetch_args_kwargs_from_env(
            node
        )

        op_type = _forward_op_type(
            self.gm,
            node,
        )

        input_shape = _tensor_shape(args)
        input_mb = _tensor_mb(args)

        range_name = (
            f"FORWARD/{op_type}/{node.name}"
        )

        # --------------------------------------------------
        # NVTX range = actual forward operator execution
        # --------------------------------------------------
        with nvtx_range(range_name):
            result = super().run_node(node)

        output_shape = _tensor_shape(result)
        output_mb = _tensor_mb(result)

        self.rows.append({
            "node_name": node.name,
            "op_type": op_type,
            "range_name": range_name,
            "input_shape": input_shape,
            "input_mb": input_mb,
            "output_shape": output_shape,
            "output_mb": output_mb,
        })

        return result

    def save_csv(self):
        if not self.csv_path:
            return

        directory = os.path.dirname(
            self.csv_path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(
            self.csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            fieldnames = [
                "node_name",
                "op_type",
                "range_name",
                "input_shape",
                "input_mb",
                "output_shape",
                "output_mb",
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(self.rows)

def is_causal_lm_model(model):
    config = getattr(model, "config", None)

    if config is None:
        return False

    return getattr(
        config,
        "model_type",
        None,
    ) in {
        "gpt_neo",
        "gpt2",
        "gptj",
        "gpt_neox",
    }


def run_model_forward(model, x):

    if is_causal_lm_model(model):

        outputs = model(
            input_ids=x,
            use_cache=False,
        )

        return outputs.logits

    return model(x)
class CausalLMSavedTensorCaptureMode(TorchDispatchMode):
    """
    Capture the exact forward tensors that the custom JIN
    Backward0 implementations will need later.

    Important:
    - runs during Node A forward only
    - no backward on Node A
    - keys come from Node B dry-run execution plan
    """

    def __init__(self, queues, tensors):
        super().__init__()

        self.queues = queues
        self.tensors = tensors


    def _pop_key(self, op, suffix):
        q = self.queues.get(
            (op, suffix),
            None,
        )

        if not q:
            raise RuntimeError(
                "[A][CAUSAL_LM_PLAN_KEY_EMPTY] "
                f"op={op} suffix={suffix}"
            )

        return q.pop(0)


    def _save(self, op, suffix, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"[A][CAUSAL_LM_CAPTURE] "
                f"{op}:{suffix} is not Tensor: "
                f"{type(tensor)}"
            )

        key = self._pop_key(
            op,
            suffix,
        )

        self.tensors[key] = (
            tensor
            .detach()
            .cpu()
            .contiguous()
        )


    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        if kwargs is None:
            kwargs = {}

        # Run the REAL forward operator.
        out = func(
            *args,
            **kwargs,
        )

        schema_name = func._schema.name

        # ====================================================
        # MM
        #
        # MmBackward0:
        #   self  -> payload
        #   mat2  -> local model weight
        # ====================================================

        if schema_name == "aten::mm":

            self_tensor = args[0]

            if (
                isinstance(self_tensor, torch.Tensor)
                and self_tensor.numel() > 0
            ):
                self._save(
                    "mm",
                    "self",
                    self_tensor,
                )

        # ====================================================
        # ADDMM
        #
        # addmm(self, mat1, mat2)
        #
        # AddmmBackward0:
        #   mat1 -> payload
        #   mat2 -> local model weight
        # ====================================================

        elif schema_name == "aten::addmm":

            mat1 = args[1]

            self._save(
                "addmm",
                "mat1",
                mat1,
            )

        # ====================================================
        # BMM
        #
        # Both operands are activation-dependent.
        # ====================================================

        elif schema_name == "aten::bmm":

            self_tensor = args[0]
            mat2 = args[1]

            self._save(
                "bmm",
                "self",
                self_tensor,
            )

            self._save(
                "bmm",
                "mat2",
                mat2,
            )

        # ====================================================
        # SOFTMAX
        #
        # SoftmaxBackward0 saves forward result.
        # ====================================================

        elif schema_name in {
            "aten::_softmax",
            "aten::_safe_softmax",
        }:

            if isinstance(out, torch.Tensor):
                self._save(
                    "softmax",
                    "result",
                    out,
                )

        # ====================================================
        # NATIVE LAYER NORM
        #
        # aten::native_layer_norm returns:
        #
        #   output, mean, rstd
        #
        # NativeLayerNormBackward0 needs:
        #   input
        #   result1 = mean
        #   result2 = rstd
        # ====================================================

        elif schema_name == "aten::native_layer_norm":

            input_tensor = args[0]

            if (
                not isinstance(out, tuple)
                or len(out) < 3
            ):
                raise RuntimeError(
                    "[A][LAYERNORM_CAPTURE] "
                    f"unexpected output type={type(out)}"
                )

            _, mean, rstd = out

            self._save(
                "layernorm",
                "input",
                input_tensor,
            )

            self._save(
                "layernorm",
                "result1",
                mean,
            )

            self._save(
                "layernorm",
                "result2",
                rstd,
            )

        # ====================================================
        # TANH
        #
        # TanhBackward0 saves result.
        # ====================================================

        elif schema_name == "aten::tanh":

            if isinstance(out, torch.Tensor):
                self._save(
                    "tanh",
                    "result",
                    out,
                )

        # ====================================================
        # POW
        #
        # GELU uses x ** 3.
        #
        # PowBackward0 saves self.
        # ====================================================

        elif schema_name == "aten::pow":

            self_tensor = args[0]

            if (
                isinstance(self_tensor, torch.Tensor)
                and self_tensor.numel() > 1
            ):
                self._save(
                    "pow",
                    "self",
                    self_tensor,
                )

        # ====================================================
        # MUL
        #
        # Match our Functions.cpp rule:
        # only tensor activations with numel > 1.
        #
        # Scalar constants are deliberately excluded.
        # ====================================================

        elif schema_name == "aten::mul":

            if len(args) < 2:
                return out

            self_tensor = args[0]
            other = args[1]

            # ========================================================
            # IMPORTANT:
            #
            # GELU contains several multiplications:
            #
            #   scalar * tensor
            #   scalar * tensor
            #   ...
            #   tensor * tensor
            #
            # Functions.cpp dryrun plan records the activation pair
            # that MulBackward0 actually needs for our JIN overwrite.
            #
            # Therefore consume the JIN queue ONLY when BOTH
            # operands are real non-scalar tensors.
            # ========================================================

            both_activation_tensors = (
                isinstance(self_tensor, torch.Tensor)
                and isinstance(other, torch.Tensor)
                and self_tensor.numel() > 1
                and other.numel() > 1
            )

            if both_activation_tensors:

                if (
                    ("mul", "self") in self.queues
                    and self.queues[("mul", "self")]
                ):
                    self._save(
                        "mul",
                        "self",
                        self_tensor,
                    )

                if (
                    ("mul", "other") in self.queues
                    and self.queues[("mul", "other")]
                ):
                    self._save(
                        "mul",
                        "other",
                        other,
                    )

        # ====================================================
        # EMBEDDING
        #
        # embedding(weight, indices, ...)
        #
        # EmbeddingBackward0 saves indices.
        # ====================================================

        elif schema_name == "aten::embedding":

            indices = args[1]

            self._save(
                "embedding",
                "indices",
                indices,
            )

        return out
class SplitRuntime:
    def __init__(self, model, role: str):
        self.model = model 
        self.role = role.upper()

        profile_csv_path = os.environ.get(
            "SPLITMAGIC_RECOMPUTE_PROFILE_CSV"
        )
        self.recompute_cost_db = None

        if profile_csv_path:
            try:
                import csv

                with open(
                    profile_csv_path,
                    "r",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    reader = csv.DictReader(f)

                    if reader.fieldnames is None:
                        raise RuntimeError(
                            f"Layer profile has no header: "
                            f"{profile_csv_path}"
                        )

                    if "node_name" not in reader.fieldnames:
                        raise RuntimeError(
                            f"Layer profile missing node_name: "
                            f"{profile_csv_path}"
                        )

                    if "median_ms" in reader.fieldnames:
                        metric = "median_ms"
                    elif "avg_ms" in reader.fieldnames:
                        metric = "avg_ms"
                    else:
                        raise RuntimeError(
                            "Layer profile requires median_ms "
                            "or avg_ms column"
                        )

                    for row in reader:
                        node_name = (
                            row.get("node_name") or ""
                        ).strip()

                        value_raw = (
                            row.get(metric) or ""
                        ).strip()

                        if not node_name or not value_raw:
                            continue

                        value = float(value_raw)

                        if value < 0:
                            raise ValueError(
                                f"Negative layer cost: "
                                f"node={node_name} "
                                f"value={value}"
                            )

                        self.recompute_layer_costs[
                            node_name
                        ] = value

                if not self.recompute_layer_costs:
                    raise RuntimeError(
                        f"No layer costs loaded from: "
                        f"{profile_csv_path}"
                    )

                print(
                    f"[SplitRuntime][LAYER_COST_LOAD] "
                    f"path={profile_csv_path} "
                    f"metric={metric} "
                    f"nodes={len(self.recompute_layer_costs)}",
                    flush=True,
                )

            except Exception as exc:
                print(
                    "[SplitRuntime][RECOMPUTE_COST_DB_ERROR] "
                    f"type={type(exc).__name__} "
                    f"error={exc}"
                )

        self.enable_recompute_profile = (
            os.getenv("JIN_RECOMPUTE_PROFILE", "0") == "1"
        )

        try:
            import torch.fx as fx

            self.fx_gm = fx.symbolic_trace(self.model)
            print("[SplitRuntime][FX_TRACE] ok", flush=True)
        except Exception as e:
            self.fx_gm = None
            self.fx_node_map = None
            self.fx_key_to_node = None

        if self.role not in ("A","B"):
            raise ValueError("Role must be either 'A' or 'B'")

    def profile_forward_layers(
        self,
        x,
        csv_path=None,
    ):
        if self.fx_gm is None:
            raise RuntimeError(
                "[FORWARD_PROFILE] FX graph is not available"
            )

        profiler = ForwardFXProfiler(
            self.fx_gm,
            csv_path=csv_path,
        )

        with torch.no_grad():
            out = profiler.run(x)

        profiler.save_csv()

        print(
            f"[FORWARD_PROFILE] "
            f"nodes={len(profiler.rows)} "
            f"csv={csv_path}",
            flush=True,
        )

        return out

    def estimate_selected_cost(
        self,
        *,
        payload,
        selected_keys: set[str],
        cost_table,
    ):
        """
        selected_keys를 동시에 payload에서 제거한다고 가정하고,
        남아 있는 activation에서 각 target까지 필요한 경로를
        다시 계산하여 전체 recompute + inject 비용을 추정한다.
        """

        if not selected_keys:
            return {
                "operator_ms": 0.0,
                "output_to_cpu_ms": 0.0,
                "patch_ms": 0.0,
                "inject_ms": 0.0,
                "total_ms": 0.0,
                "missing": [],
                "executed_nodes": [],
                "paths": {},
            }

        if not self.recompute_layer_costs:
            return {
                "operator_ms": 0.0,
                "output_to_cpu_ms": 0.0,
                "patch_ms": 0.0,
                "inject_ms": 0.0,
                "total_ms": 0.0,
                "missing": [
                    "recompute_layer_costs:not_loaded"
                ],
                "executed_nodes": [],
                "paths": {},
            }

        # payload 원본을 건드리지 않기 위해
        # selected key를 제외한 임시 payload를 만든다.
        import copy

        trial_payload = copy.copy(payload)
        trial_payload.tensors = {
            key: tensor
            for key, tensor in payload.tensors.items()
            if key not in selected_keys
        }

        # alias도 selected key를 가리키면 사용할 수 없으므로 제거한다.
        original_aliases = dict(
            getattr(payload, "aliases", {})
            or getattr(payload, "meta", {}).get(
                "aliases",
                {},
            )
        )

        trial_payload.aliases = {
            alias_key: canonical_key
            for alias_key, canonical_key
            in original_aliases.items()
            if (
                alias_key not in selected_keys
                and canonical_key not in selected_keys
            )
        }

        trial_payload.meta = dict(
            getattr(payload, "meta", {})
        )
        trial_payload.meta["aliases"] = (
            trial_payload.aliases
        )

        if (
            self.fx_node_map is None
            or self.fx_key_to_node is None
        ):
            return {
                "operator_ms": 0.0,
                "output_to_cpu_ms": 0.0,
                "patch_ms": 0.0,
                "inject_ms": 0.0,
                "total_ms": 0.0,
                "missing": [
                    "fx_context:not_available"
                ],
                "executed_nodes": [],
                "paths": {},
            }

        # 남아 있는 payload tensor를 FX node로 변환한다.
        available_tensors = (
            build_available_tensors_from_payload(
                model=self.model,
                payload=trial_payload,
                device="cpu",
                key_to_node=self.fx_key_to_node,
            )
        )

        available_nodes = set(
            available_tensors.keys()
        )



        node_map = self.fx_node_map
        key_to_node = self.fx_key_to_node

        missing: list[str] = []
        executed_nodes: set[str] = set()
        paths: dict[str, list[str]] = {}

        for key in sorted(selected_keys):
            target_node = key_to_node.get(key)

            if target_node is None:
                missing.append(
                    f"{key}:missing_target"
                )
                continue

            # selected key의 target은 trial에서 없는 값이어야 한다.
            available_nodes.discard(target_node)

            start = find_nearest_available_start(
                node_map=node_map,
                node_name=target_node,
                available_nodes=available_nodes,
            )

            if start is None:
                missing.append(
                    f"{key}:no_available_start"
                )
                continue

            path = build_path_from_start_to_node(
                node_map=node_map,
                start_node=start,
                target_node=target_node,
            )

            if not path:
                missing.append(
                    f"{key}:no_path"
                )
                continue

            paths[key] = list(path)

            # path[0]은 이미 보유한 start activation이다.
            for node_name in path[1:]:
                executed_nodes.add(node_name)

        operator_ms = 0.0

        for node_name in sorted(executed_nodes):
            node_cost = self.recompute_layer_costs.get(
                node_name
            )

            if node_cost is None:
                missing.append(
                    f"{node_name}:missing_layer_cost"
                )
                continue

            operator_ms += float(node_cost)

        output_to_cpu_ms = 0.0
        patch_ms = 0.0

        for key in selected_keys:
            profile = cost_table.get(key)

            if profile is None:
                missing.append(
                    f"{key}:missing_key_cost"
                )
                continue

            output_to_cpu_ms += float(
                profile.output_to_cpu_ms
            )

            patch_ms += float(
                profile.patch_ms
            )

        total_ms = (
            operator_ms
            + output_to_cpu_ms
            + patch_ms
        )

        return {
            "operator_ms": operator_ms,
            "output_to_cpu_ms": output_to_cpu_ms,
            "patch_ms": patch_ms,

            # 기존 코드와 로그 호환용
            "inject_ms": patch_ms,

            "total_ms": total_ms,
            "missing": missing,
            "executed_nodes": sorted(
                executed_nodes
            ),
            "paths": paths,
        }
    def capture_causal_lm_forward_plan(
        self,
        x,
        plan,
    ):
        if self.role != "A":
            raise RuntimeError(
                "Only role 'A' can capture tensors"
            )

        tensors = {}

        # ========================================================
        # Build queues from Node B backward dry-run plan.
        #
        # dryrun plan:
        #   backward execution order
        #
        # Node A capture:
        #   forward execution order
        #
        # Therefore reverse each op/suffix queue.
        # ========================================================

        queues = {}

        for e in plan:
            op = e["op"]
            idx = e["idx"]
            suffix = e["suffix"]

            # B already owns these parameters.
            if (
                op == "conv"
                and suffix == "weight"
            ):
                continue

            if (
                op == "addmm"
                and suffix == "mat2"
            ):
                continue

            if (
                op == "mm"
                and suffix == "mat2"
            ):
                continue

            key = (
                f"graph:{op}:"
                f"{idx}:{suffix}"
            )

            queues.setdefault(
                (op, suffix),
                [],
            ).append(key)

        # backward order -> forward order
        for queue_key in queues:
            queues[queue_key] = list(
                reversed(
                    queues[queue_key]
                )
            )

        # ========================================================
        # REAL Node A forward
        # ========================================================

        with CausalLMSavedTensorCaptureMode(
            queues=queues,
            tensors=tensors,
        ):
            out = run_model_forward(
                self.model,
                x,
            )

        # Model output is needed by B for straight-through output.
        tensors["model.output"] = (
            out.detach()
            .cpu()
            .contiguous()
        )

        # ========================================================
        # Verify every required plan key was captured.
        # ========================================================

        leftovers = {
            f"{op}:{suffix}": len(queue)
            for (op, suffix), queue
            in queues.items()
            if len(queue) > 0
        }

        if leftovers:
            raise RuntimeError(
                "[A][CAUSAL_LM_PLAN_KEYS_LEFTOVER] "
                f"{leftovers}"
            )

        # ========================================================
        # Build existing SplitMagic Payload.
        # ========================================================

        items = []

        for key, tensor in tensors.items():

            items.append({
                "key": key,
                "jin_key": key,
                "graph_key": key,
                "tensor": tensor,
                "node": key,
                "attr": key,
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
                "requires_grad": False,
            })

        payload = payload_from_jin_items(
            items
        )

        payload.meta = getattr(
            payload,
            "meta",
            {},
        )

        payload.meta[
            "dryrun_backward_plan"
        ] = plan

        payload.meta[
            "model_family"
        ] = "causal_lm"

        payload.meta[
            "sequence_length"
        ] = int(x.size(1))

        payload.meta[
            "tensor_policy"
        ] = {
            "policy":
                "causal_lm_forward_dispatch",

            "included_keys":
                sorted(tensors.keys()),

            "num_payload_tensors":
                len(tensors),

            "all_keys":
                sorted(tensors.keys()),
        }

        print(
            "[A][CAUSAL_LM_CAPTURE] "
            f"seq={x.size(1)} "
            f"keys={len(tensors)}",
            flush=True,
        )

        return payload
    def capture_jin_forward_plan(self, x, plan):
        #
        # Step 1. Build lookup queues from backward execution plan
        # Step 2. Define tensors saving and forward hooks
        # Step 3. Register forward hooks to capture the saved tensors
        # Step 4. Run the forward pass and capture
        # Step 5. Verify no plan keys are left
        # Step 6. Build payload from captured tensors
        #

        # Check that the role is 'A' since only Node A can capture tensors during the forward pass
        if is_causal_lm_model(
            self.model
        ):
            return (
                self.capture_causal_lm_forward_plan(
                    x=x,
                    plan=plan,
                )
            )
        
        if self.role != "A":
            raise RuntimeError("Only role 'A' can capture tensors")
    
        tensors = {}

        handles = []

        # Step 1: Build lookup queues from backward execution plan
        queues = {}

        for e in plan:
            op = e["op"]
            suffix = e["suffix"]
            idx = e["idx"]

            # B local parameter라 payload로 안 보냄
            if op == "conv" and suffix == "weight":
                continue
            if op == "addmm" and suffix == "mat2":
                continue
            if op == "mm" and suffix == "mat2":
                continue

            key = f"graph:{op}:{idx}:{suffix}"
            queues.setdefault((op, suffix), []).append(key)

        # Dry-run plan is generated in backward execution order,
        # but forward hooks are called in forward execution order,
        # Reverse each ( op, suffix ) queue so that pop_key() returns
        # the key that corresponds to the current forward hook. 
        for k in queues:
            queues[k] = list(reversed(queues[k]))

        def pop_key(op, suffix):
            q = queues.get((op, suffix), None)
            if not q:
                raise RuntimeError(f"[A][PLAN_KEY_EMPTY] op={op} suffix={suffix}")
            return q.pop(0)

        # Step 2: Define tensors saving and forward hooks
        # save_tensor는 hook에서 호출되어 tensor를 저장
        def save_tensor(key, tensor):
            tensors[key] = tensor.detach().cpu().contiguous()

        # make_hook는 각 모듈에 대한 forward hook을 생성
        def make_hook(module):
            def hook(mod, inputs, output):
                if len(inputs) == 0:
                    return

                inp = inputs[0]

                if isinstance(mod, nn.Conv2d):
                    # ConvBackward needs the forward input tensor.
                    key = pop_key("conv", "input")
                    save_tensor(key, inp)

                elif isinstance(mod, nn.BatchNorm2d):
                    # BatchNormBackward needs the forward input and normalization statistics.
                    save_tensor(pop_key("bn", "input"), inp)

                    # BN weight
                    if mod.weight is not None:
                        save_tensor(pop_key("bn", "weight"), mod.weight)

                    save_tensor(pop_key("bn", "running_mean"), mod.running_mean)
                    save_tensor(pop_key("bn", "running_var"), mod.running_var)

                    # BN result1/result2 = save_mean / save_invstd
                    dims = (0, 2, 3)
                    mean = inp.detach().mean(dim=dims)
                    var = inp.detach().var(dim=dims, unbiased=False)
                    invstd = torch.rsqrt(var + mod.eps)

                    save_tensor(pop_key("bn", "result1"), mean)
                    save_tensor(pop_key("bn", "result2"), invstd)

                elif isinstance(mod, nn.ReLU):
                    # ReLUBackward needs the forward output/result tensor. 
                    key = pop_key("relu", "result")
                    save_tensor(key, output)
                
                elif isinstance(mod, nn.Linear):
                    # Linear is represented as AddmmBackward in autograd.
                    # AddmmBackward needs the mat1, which corresponds to the forward input.
                    key = pop_key("addmm", "mat1")
                    save_tensor(key, inp)

                elif isinstance(mod, nn.MaxPool2d):
                    # MaxPool2dBackward needs both the forward input and pooling indices.
                    save_tensor(pop_key("maxpool2d", "input"), inp)

                    _, indices = F.max_pool2d(
                        inp,
                        kernel_size=mod.kernel_size,
                        stride=mod.stride,
                        padding=mod.padding,
                        dilation=mod.dilation,
                        ceil_mode=mod.ceil_mode,
                        return_indices=True,
                    )

                    save_tensor(pop_key("maxpool2d", "indices"), indices)

            return hook

        # Step 3: Register forward hooks to capture the saved tensors
        for _, m in self.model.named_modules():
            if isinstance(
                m,
                (
                    nn.Conv2d,
                    nn.BatchNorm2d,
                    nn.ReLU,
                    nn.Linear,
                    nn.MaxPool2d,
                ),
            ):
                handles.append(m.register_forward_hook(make_hook(m)))

        # Step 4: Run the forward pass and capture
        try:
            out = run_model_forward(
                self.model,
                x,
            )
        finally:
            for h in handles:
                h.remove()

        tensors["model.output"] = out.detach().cpu().contiguous()

        # Step 5: Verify no plan keys are left
        # If any queue still has keys, the plan expected a tensor
        # that was not captured by the forward hooks.
        # This usually means a mismatch between the dry-run plan
        # and the actual model forward execution.
        leftovers = {
            f"{op}:{suffix}": len(q)
            for (op, suffix), q in queues.items()
            if len(q) > 0
        }

        if leftovers:
            raise RuntimeError(f"[A][PLAN_KEYS_LEFTOVER] {leftovers}")

        # Step 6: Build payload from captured tensors
        items = []
        for key, tensor in tensors.items():
            items.append({
                "key": key,
                "jin_key": key,
                "graph_key": key,
                "tensor": tensor,
                "node": key,
                "attr": key,
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
                "requires_grad": False,
            })

        
        payload = payload_from_jin_items(items)

        payload.meta = getattr(payload, "meta", {})
        payload.meta["dryrun_backward_plan"] = plan
        payload.meta["tensor_policy"] = {
            "policy": "forward_only_plan_keys",
            "included_keys": sorted(tensors.keys()),
            "num_payload_tensors": len(tensors),
            "all_keys": sorted(tensors.keys()),
        }

        return payload    
    
    def info(self):
        return f"SplitRuntime(role={self.role}, model={self.model.__class__.__name__})"
    
    def backward_jin(
        self,
        x_dummy,
        y,
        payload,
        loss_fn,
        payload_path=None,
        tensor_policy=None,
        dryrun_backward_plan=None,
        dropout_seed=None,

    ):
        if self.role != "B":
            raise RuntimeError("backward_jin() is only available for Node B")
        
        profile_t0 = time.perf_counter()
        t0 = time.perf_counter()

        with nvtx_range("B_zero_grad"):

            self.model.train()

            if is_causal_lm_model(self.model):
                print(
                    "[B][CAUSAL_LM_MODE] "
                    "train mode inside backward_jin",
                    flush=True,
                )

            self.model.zero_grad(
                set_to_none=True
            )

        t1 = time.perf_counter()
        zero_grad_ms = (t1 - t0) * 1000

        t0 = time.perf_counter()

        # dummy forward profiling 
        with nvtx_range("B_dummy_forward"):

            if is_causal_lm_model(self.model):

                if dropout_seed is None:
                    raise RuntimeError(
                        "[B] dropout_seed is required "
                        "for causal LM train-mode backward"
                    )

                torch.manual_seed(
                    int(dropout_seed)
                )

                print(
                    f"[B][DROPOUT_SEED] "
                    f"seed={int(dropout_seed)}",
                    flush=True,
                )

            out_dummy = run_model_forward(
                self.model,
                x_dummy,
            )

        # out_dummy = self.model(x_dummy)
        t1 = time.perf_counter()
        dummy_forward_ms = (t1 - t0) * 1000

        if not is_causal_lm_model(
            self.model
        ):

            from .fx_trace import (
                debug_fx_shapes
            )

            with nvtx_range(
                "B_debug_fx_shapes"
            ):
                debug_fx_shapes(
                    self.model,
                    x_dummy,
                )

        t0 = time.perf_counter()

        # loss_build
        with nvtx_range("B_loss_build"):

            if "model.output" not in payload.tensors:
                raise KeyError(
                    "payload does not contain 'model.output'"
                )

            out_real = (
                payload.tensors["model.output"]
                .detach()
                .to(out_dummy.device)
            )

            # --------------------------------------------------------
            # Forward:
            #     value == out_real (Node A)
            #
            # Backward:
            #     d(out)/d(out_dummy) == 1
            #
            # out_dummy - out_dummy.detach() is exactly zero
            # while retaining the gradient path through out_dummy.
            # --------------------------------------------------------

            out = (
                out_real
                + (
                    out_dummy
                    - out_dummy.detach()
                )
            )

            loss = loss_fn(
                out,
                y,
            )

            if is_causal_lm_model(self.model):

                print(
                    "[B][OUTPUT_CHECK] "
                    f"exact_A_output="
                    f"{torch.equal(out.detach(), out_real)} "
                    f"max_diff="
                    f"{(out.detach() - out_real).abs().max().item()}",
                    flush=True,
                )

        t1 = time.perf_counter()
        loss_build_ms= ( t1 - t0 ) * 1000

        print("[B][BACKWARD] start")
        plan = dryrun_backward_plan or []

        if not plan:
            raise RuntimeError("[B][RECOMPUTE] missing dryrun_backward_plan")
        
        t0 = time.perf_counter()

        with nvtx_range("B_find_missing_keys"):

            required_keys = get_required_keys_from_plan(plan)

            aliases = getattr(payload, "meta", {}).get("aliases", {})
            payload.meta = getattr(payload, "meta",{})
            payload.meta["aliases"] = aliases

            for k in [
                "graph:maxpool2d:0:input",
                "graph:maxpool2d:0:indices",
            ]:
                print(
                    f"[MAXPOOL_CHECK] "
                    f"key={k} "
                    f"in_payload={k in payload.tensors} "
                    f"in_alias={k in aliases}",
                    flush=True,
                )

            missing_keys = sorted([
                k for k in required_keys
                if (not is_always_local_key(k))
                and (k not in payload.tensors)
                and (k not in aliases)
                and (not has_relu_mask_for(k, payload))
            ])

        build_menu = (
            os.environ.get(
                "JIN_BUILD_RECOMPUTE_MENU",
                "0",
            )
            == "1"
        )

        step = int(
            os.environ.get(
                "JIN_STEP",
                "0",
            )
        )

        if build_menu and step == 0:
            candidate_keys = sorted(
                key
                for key in required_keys
                if is_recomputable_key(key)
                and key in payload.tensors
            )

            self.build_recompute_menu(
                candidate_keys=candidate_keys,
                payload=payload,
                device=x_dummy.device,
                profile_csv=os.environ.get(
                    "JIN_RECOMPUTE_PROFILE_PATH",
                    "./recompute_layer_profile.csv",
                ),
                menu_csv=os.environ.get(
                    "JIN_RECOMPUTE_MENU_PATH",
                    "./recompute_menu.csv",
                ),
            )

        t1 = time.perf_counter()
        required_check_ms = (t1 - t0) * 1000
        print(
            f"[B][ALIAS_AWARE] aliases={len(aliases)}",
            flush=True,
        )
    
        print(
            f"[B][RECOMPUTE_CHECK] "
            f"required={len(required_keys)} "
            f"missing={len(missing_keys)} "
            f"first={missing_keys[:10]}",
            flush=True,
        )

        recompute_ms = 0.0
        injected_ms = 0.0
        recomputed_mb = 0.0

        missing_count_before = len(missing_keys)

        recompute_stats = {
            "estimated_grouped_ms": 0.0,
            "recompute_plan_ms": 0.0,
        }

        if missing_keys:

            print(
                f"[B][RECOMPUTE_TODO] missing keys exist: {missing_keys[:20]} "
                f"first={missing_keys[:10]}",
                flush=True
            )

            t0 = time.perf_counter()

            # recompute profile
            with nvtx_range("B_recompute_total"):

                with nvtx_range("B_recompute_missing_tensors"):

                    recomputed, recompute_stats = self.recompute_missing_keys(
                        missing_keys=missing_keys,
                        payload=payload,
                        payload_path=payload_path,
                        device=x_dummy.device,
                    )

            recomputed_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in recomputed.values()
            )

            recomputed_mb = (
                recomputed_bytes / 1024 / 1024
            )

            t1 = time.perf_counter()
            recompute_ms = (t1 - t0 ) * 1000

            t0 = time.perf_counter()
            with nvtx_range("B_injected_recompute_tensors"):
                inject_recomputed_tensors(
                    payload=payload,
                    payload_path=payload_path,
                    recomputed=recomputed,
                )
            
            t1 = time.perf_counter()
            injected_ms = (t1 - t0 ) * 1000

            missing_keys = sorted([
                k for k in required_keys
                if (not is_always_local_key(k))
                and (k not in payload.tensors)
                and (k not in aliases)
                and (not has_relu_mask_for(k, payload))
            ])

            print(
                f"[B][RECOMPUTE_AFTER] still_missing={len(missing_keys)} "
                f"first={missing_keys[:10]}",
                flush=True,
            )

            if missing_keys:
                raise RuntimeError(
                    f"[B][RECOMPUTE_FAIL] still missing keys: {missing_keys[:20]}"
                )                            
            # missing_keys = []
        # Node A cost-policy prediction from the recompute menu.
        node_a_predicted_operator_ms = float(
            payload.meta.get(
                "predicted_operator_ms",
                0.0,
            )
        )

        node_a_predicted_recompute_ms = float(
            payload.meta.get(
                "predicted_recompute_ms",
                0.0,
            )
        )

        node_a_predicted_inject_ms = float(
            payload.meta.get(
                "predicted_inject_ms",
                0.0,
            )
        )

        # Node B validation prediction:
        # sum of profiled costs for actually executed operators.
        predicted_operator_ms = float(
            recompute_stats.get(
                "predicted_operator_ms",
                0.0,
            )
        )

        actual_operator_ms = float(
            recompute_stats.get(
                "actual_operator_ms",
                0.0,
            )
        )

        actual_recompute_ms = float(
            recompute_stats.get(
                "actual_recompute_ms",
                0.0,
            )
        )

        recompute_overhead_ms = max(
            0.0,
            recompute_ms - actual_recompute_ms,
        )

        recompute_prediction_ratio = (
            actual_operator_ms / predicted_operator_ms
            if predicted_operator_ms > 0.0
            else 0.0
        )

        print(
            f"[B][RECOMPUTE_COST_VALIDATION] "
            f"node_a_predicted_operator_ms="
            f"{node_a_predicted_operator_ms:.3f} "
            f"node_a_predicted_recompute_ms="
            f"{node_a_predicted_recompute_ms:.3f} "
            f"predicted_operator_ms={predicted_operator_ms:.3f} "
            f"actual_operator_ms={actual_operator_ms:.3f} "
            f"actual_recompute_ms={actual_recompute_ms:.3f} "
            f"recompute_wall_ms={recompute_ms:.3f} "
            f"recompute_overhead_ms={recompute_overhead_ms:.3f} "
            f"prediction_ratio={recompute_prediction_ratio:.3f}",
            flush=True,
        )

        t0 = time.perf_counter()

        with nvtx_range("B_torch_backward"):
            loss.backward()

        t1 = time.perf_counter()

        torch_backward_ms = (t1 - t0) * 1000

        profile_t1 = time.perf_counter()
        total_backward_jin_ms = (profile_t1 - profile_t0) * 1000
        print(
            f"[B][PROFILE] "
            f"zero_grad_ms={zero_grad_ms:.3f} "
            f"dummy_forward_ms={dummy_forward_ms:.3f} "
            f"loss_build_ms={loss_build_ms:.3f} "
            f"required_check_ms={required_check_ms:.3f} "
            f"recompute_ms={recompute_ms:.3f} "
            f"injected_ms={injected_ms:.3f} "
            f"torch_backward_ms={torch_backward_ms:.3f} "
            # f"grad_dump_ms={grad_dump_ms:.3f} "
            f"total_backward_jin_ms={total_backward_jin_ms:.3f}",
            flush=True,
        )

        print("[B][BACKWARD] done")

        self.last_experiment_metrics = {
            "missing_count": missing_count_before,

            # Node A selection result.
            "node_a_selection_policy": payload.meta.get(
                "selection_policy",
                "unknown",
            ),
            "node_a_drop_ratio": float(
                payload.meta.get(
                    "drop_ratio",
                    0.0,
                )
            ),
            "node_a_saved_mb": float(
                payload.meta.get(
                    "saved_mb",
                    0.0,
                )
            ),
            "node_a_dropped_count": int(
                payload.meta.get(
                    "dropped_count",
                    0,
                )
            ),

            # Node A cost-policy estimates.
            "node_a_predicted_operator_ms": (
                node_a_predicted_operator_ms
            ),
            "node_a_predicted_recompute_ms": (
                node_a_predicted_recompute_ms
            ),
            "node_a_predicted_inject_ms": (
                node_a_predicted_inject_ms
            ),

            # Node B recomputation validation.
            "estimated_grouped_ms": recompute_stats.get(
                "estimated_grouped_ms",
                0.0,
            ),
            "predicted_operator_ms": (
                predicted_operator_ms
            ),
            "actual_recompute_ms": actual_recompute_ms,
            "actual_operator_ms": actual_operator_ms,
            "recompute_wall_ms": recompute_ms,
            "recompute_overhead_ms": (
                recompute_overhead_ms
            ),
            "recompute_prediction_ratio": (
                recompute_prediction_ratio
            ),
            "recomputed_mb": recomputed_mb,
            "recompute_executed_node_count": (
                recompute_stats.get(
                    "recompute_executed_node_count",
                    0,
                )
            ),
            "recompute_profiled_node_count": (
                recompute_stats.get(
                    "recompute_profiled_node_count",
                    0,
                )
            ),
            "recompute_missing_profile_count": (
                recompute_stats.get(
                    "recompute_missing_profile_count",
                    0,
                )
            ),
            "recompute_plan_ms": recompute_stats.get(
                "recompute_plan_ms",
                0.0,
            ),
            "inject_ms": injected_ms,
            "torch_backward_ms": torch_backward_ms,
            "backward_jin_ms": total_backward_jin_ms,
        }

        return loss
    def recompute_missing_keys(
        self,
        missing_keys,
        payload,
        payload_path,
        device,
    ):
        
        estimated_grouped_ms = 0.0
        recomputable = [
            k for k in missing_keys
            if is_recomputable_key(k)
        ]
        maxpool_indices_keys = [
            k for k in recomputable
            if (
                k.startswith("graph:maxpool2d:")
                and k.endswith(":indices")
            )
        ]

        regular_recomputable = [
            k for k in recomputable
            if k not in maxpool_indices_keys
        ]

        non_recomputable = [
            k for k in missing_keys
            if not is_recomputable_key(k)
        ]

        if non_recomputable:
            print(
                f"[B][RECOMPUTE_NON_RECOMPUTABLE] "
                f"n={len(non_recomputable)} "
                f"first={non_recomputable[:10]}",
                flush=True,
            )

        payload_keys = set(payload.tensors.keys())

        available_tensors = build_available_tensors_from_payload(
            model=self.model,
            payload=payload,
            device=device,
        )

        node_map = build_fx_maps(self.model)

        reverse_order = True

        key_to_node = build_jin_key_to_fx_node(
            self.model,
            reverse_order=reverse_order,
        )

        modules = dict(self.model.named_modules())

        maxpool_nodes = []

        if self.fx_gm is not None:
            for node in self.fx_gm.graph.nodes:
                if node.op != "call_module":
                    continue

                module = modules.get(str(node.target))

                if isinstance(module, nn.MaxPool2d):
                    maxpool_nodes.append(node.name)

        # JIN maxpool numbering follows backward-plan order,
        # while FX graph nodes are in forward order.
        maxpool_nodes = list(reversed(maxpool_nodes))

        print(
            f"[B][MAXPOOL_NODE_MAP] "
            f"nodes={maxpool_nodes}",
            flush=True,
        )

        available_tensor_nodes = set(available_tensors.keys())

        gm = getattr(self, "fx_gm", None)

        recompute_engine = FXRecomputeEngine(
            self.model,
            gm=gm,
            node_values=dict(available_tensors)
        )

        profile_csv = os.environ.get(
            "JIN_RECOMPUTE_PROFILE_PATH",
            "./recompute_layer_profile.csv",
        )

        profile_concurrency = int(
            os.environ.get(
                "JIN_RECOMPUTE_PROFILE_CONCURRENCY",
                "1",
            )
        )
        # recompute_engine.load_profile_db_csv(profile_csv)
        recompute_engine.load_profile_db_csv(
            profile_csv,
            concurrency=profile_concurrency,
            metric="avg_ms",
        )

        recomputed = {}
        recompute_records = []

        from collections import defaultdict

        groups = defaultdict(list)

        find_ms = 0.0
        path_ms = 0.0
        estimated_grouped_ms = 0.0

        with nvtx_range("B_recompute_plan"):

            # for key in recomputable:
            for key in regular_recomputable:
                if key not in key_to_node:
                    print(f"[B][RECOMPUTE_SKIP] no FX target for key={key}")
                    continue

                target_node = key_to_node[key]

                t0 = time.perf_counter()
                start = find_nearest_available_start(
                    node_map=node_map,
                    node_name=target_node,
                    available_nodes=available_tensor_nodes,
                )
                t1 = time.perf_counter()

                if start is None:
                    print(f"[B][RECOMPUTE_SKIP] no start tensor for key={key}")
                    continue
                
                path = build_path_from_start_to_node(
                    node_map=node_map,
                    start_node=start,
                    target_node=target_node,
                )
                t2 = time.perf_counter()

                find_ms += (t1 - t0) * 1000
                path_ms += (t2 - t1) * 1000

                if not path:
                    print(f"[B][RECOMPUTE_SKIP] empty path for key={key}")
                    continue

                cost_ms, missing_profile_nodes = (
                    recompute_engine.estimate_recompute_cost(path)
                )

                groups[start].append(
                    (
                        key,
                        target_node,
                        path,
                        cost_ms,
                        missing_profile_nodes,
                    )
                )

                # print(
                #     f"[B][RECOMPUTE_PATH] "
                #     f"key={key} "
                #     f"start={start} "
                #     f"target={target_node} "
                #     f"estimated_ms={cost_ms:.6f} "
                #     f"missing_profile={missing_profile_nodes} "
                #     f"path={' -> '.join(path)}",
                #     flush=True,
                # )

        # print(
        #     f"[B][RECOMPUTE_GROUPS] "
        #     f"num_groups={len(groups)} "
        #     f"sizes={[(s, len(v)) for s, v in groups.items()]}",
        #     flush=True,
        # )
        print(
            f"[RECOMPUTE_PLAN_PROFILE] "
            f"find_start_ms={find_ms:.3f} "
            f"build_path_ms={path_ms:.3f}"
        )

        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

        recompute_exec_t0 = time.perf_counter()

        gpu_outputs = {}

        with nvtx_range("B_recompute_execute"):

            for start, items in groups.items():

                # start tensor도 recompute_engine cache에 등록
                #
                if start not in recompute_engine.node_values:
                    recompute_engine.node_values[start] = available_tensors[start]

                items = sorted(
                    items,
                    key=lambda x: len(x[2]),
                    reverse=True,
                )

                for (
                    key,
                    target_node,
                    path,
                    estimated_ms,
                    missing_profile_nodes,
                ) in items:

                    if target_node in recompute_engine.node_values:
                        out = recompute_engine.node_values[target_node]

                    else:
                        start_tensor = recompute_engine.node_values[start]

                        # 실제 실행되는 path만 grouped estimate에 포함
                        estimated_grouped_ms += estimated_ms

                        out = recompute_engine.recompute_path(
                            start_tensor=start_tensor,
                            path=path,
                        )

                    if out is None:
                        print(
                            f"[B][RECOMPUTE_UNSAFE_SKIP] key={key} "
                            f"start={start} path={' -> '.join(path)}",
                            flush=True,
                        )
                        continue

                    gpu_outputs[key] = out


                    available_tensors[target_node] = out
                    available_tensor_nodes.add(target_node)

                    print(
                        f"[B][RECOMPUTE_OK] key={key} "
                        f"start={start} shape={tuple(out.shape)}",
                        flush=True,
                    )
                # ResNet-18 ImageNet MaxPool indices 전용 recompute
        for indices_key in maxpool_indices_keys:
            # graph:maxpool2d:3:indices
            #                  ↑
            pool_idx = int(
                indices_key.split(":")[2]
            )

            if pool_idx >= len(maxpool_nodes):
                print(
                    f"[B][MAXPOOL_INDICES_SKIP] "
                    f"key={indices_key} "
                    f"reason=pool_index_out_of_range "
                    f"pool_idx={pool_idx} "
                    f"num_pools={len(maxpool_nodes)}",
                    flush=True,
                )
                continue

            pool_node_name = maxpool_nodes[pool_idx]

            input_key = (
                f"graph:maxpool2d:{pool_idx}:input"
            )

            # MaxPool에 들어가는 실제 input.
            # 앞에서 recompute했다면 gpu_outputs에 있음.
            pool_input = gpu_outputs.get(input_key)

            if pool_input is None:
                pool_input = payload.tensors.get(input_key)

                if pool_input is not None:
                    pool_input = pool_input.to(device)

            if pool_input is None:
                print(
                    f"[B][MAXPOOL_INDICES_SKIP] "
                    f"key={indices_key} "
                    f"pool_node={pool_node_name} "
                    f"reason=missing_pool_input",
                    flush=True,
                )
                continue

            cache_key = (
                f"{pool_node_name}__indices"
            )

            pool_indices = (
                recompute_engine.node_values.get(
                    cache_key
                )
            )

            # 아직 MaxPool 자체가 실행되지 않았다면
            # recompute.py에서 실행한다.
            if pool_indices is None:
                _, pool_indices = (
                    recompute_engine.recompute_maxpool_from_input(
                        node_name=pool_node_name,
                        input_tensor=pool_input,
                        use_nvtx=True,
                    )
                )

            gpu_outputs[indices_key] = pool_indices

            print(
                f"[B][MAXPOOL_INDICES_FROM_RECOMPUTE] "
                f"key={indices_key} "
                f"pool_idx={pool_idx} "
                f"pool_node={pool_node_name} "
                f"cache_key={cache_key} "
                f"input_shape={tuple(pool_input.shape)} "
                f"indices_shape={tuple(pool_indices.shape)} "
                f"dtype={pool_indices.dtype}",
                flush=True,
            )
        # GPU recompute가 끝날 때까지 기다린 후 측정 종료
        if (
            torch.cuda.is_available()
            and torch.device(device).type == "cuda"
        ):
            torch.cuda.synchronize(device)

        actual_recompute_ms = (
            time.perf_counter() - recompute_exec_t0
        ) * 1000.0

        # CPU 복사는 별도 측정
        output_copy_t0 = time.perf_counter()

        with nvtx_range("B_recompute_output_to_cpu"):
            for key, out in gpu_outputs.items():
                 with nvtx_range(
                    f"OUTPUT_TO_CPU/{key}"
                ):
                    recomputed[key] = (
                        out.detach()
                        .cpu()
                        .contiguous()
                    )
        output_copy_ms = (
            time.perf_counter() - output_copy_t0
        ) * 1000.0

        executed_nodes = list(
            recompute_engine.executed_recompute_nodes
        )
        actual_operator_ms = (
            recompute_engine.actual_operator_ms_total
        )

        predicted_operator_ms = 0.0
        profiled_node_count = 0
        missing_profile_nodes = []

        for node_name in executed_nodes:
            node_cost = recompute_engine.profile_db.get(
                node_name
            )

            if node_cost is None:
                missing_profile_nodes.append(node_name)
                continue

            predicted_operator_ms += node_cost
            profiled_node_count += 1


        print(
            f"[B][RECOMPUTE_OPERATOR_ESTIMATE] "
            f"executed_node_count={len(executed_nodes)} "
            f"profiled_node_count={profiled_node_count} "
            f"missing_profile_count="
            f"{len(missing_profile_nodes)} "
            f"predicted_operator_ms="
            f"{predicted_operator_ms:.3f} "
            f"missing_profile_nodes="
            f"{missing_profile_nodes[:10]}",
            flush=True,
        )

        recompute_overhead_ms = max(
            0.0,
            actual_recompute_ms - actual_operator_ms,
        )

        operator_prediction_ratio = (
            actual_operator_ms / predicted_operator_ms
            if predicted_operator_ms > 0
            else float("nan")
        )

        prediction_ratio = (
            actual_recompute_ms / predicted_operator_ms
            if predicted_operator_ms > 0
            else float("nan")
        )

        print(
            f"[B][RECOMPUTE_COST_VALIDATION] "
            f"predicted_operator_ms={predicted_operator_ms:.3f} "
            f"estimated_grouped_ms={estimated_grouped_ms:.3f} "
            f"actual_operator_ms={actual_operator_ms:.3f} "
            f"actual_recompute_ms={actual_recompute_ms:.3f} "
            f"recompute_overhead_ms={recompute_overhead_ms:.3f} "
            f"operator_over_predicted={operator_prediction_ratio:.3f}",
            flush=True,
        )


        stats = {
            "estimated_grouped_ms": estimated_grouped_ms,
            "recompute_plan_ms": find_ms + path_ms,

            "predicted_operator_ms": predicted_operator_ms,
            "actual_operator_ms": actual_operator_ms,
            "actual_recompute_ms": actual_recompute_ms,
            "recompute_output_to_cpu_ms": output_copy_ms,

            "recompute_overhead_ms": recompute_overhead_ms,

            "recompute_executed_node_count": len(
                executed_nodes
            ),
            "recompute_profiled_node_count": (
                profiled_node_count
            ),
            "recompute_missing_profile_count": len(
                missing_profile_nodes
            ),
        }
        return recomputed, stats
    
    def build_recompute_menu(
        self,
        candidate_keys,
        payload,
        device,
        profile_csv="./recompute_layer_profile.csv",
        menu_csv="./recompute_menu.csv",
    ):
        import csv

        available_tensors_base = build_available_tensors_from_payload(
            model=self.model,
            payload=payload,
            device=device,
        )

        node_map = build_fx_maps(self.model)

        key_to_node = build_jin_key_to_fx_node(
            self.model,
            reverse_order=True,
        )
        modules = dict(self.model.named_modules())

        maxpool_nodes = []

        for node in self.fx_gm.graph.nodes:
            if node.op != "call_module":
                continue

            module = modules.get(str(node.target))

            if isinstance(module, nn.MaxPool2d):
                maxpool_nodes.append(node.name)

        # JIN graph:maxpool2d:i 번호는 backward plan 순서이므로
        # forward FX 순서의 반대로 맞춘다.
        maxpool_nodes = list(reversed(maxpool_nodes))

        print(
            f"[B][MAXPOOL_NODE_MAP] "
            f"nodes={maxpool_nodes}",
            flush=True,
        )

        engine = FXRecomputeEngine(
            self.model,
            gm=getattr(self, "fx_gm", None),
            node_values=dict(available_tensors_base),
        )

        profile_concurrency = int(
            os.environ.get(
                "JIN_RECOMPUTE_PROFILE_CONCURRENCY",
                "1",
            )
        )

        engine.load_profile_db_csv(
            profile_csv,
            concurrency=profile_concurrency,
            # metric="avg_ms",
            metric="avg_ms",
        )

        rows = []

        for key in candidate_keys:
            tensor = payload.tensors.get(key)

            if tensor is None:
                continue

            target_node = key_to_node.get(key)

            if target_node is None:
                continue

            available_tensors = dict(available_tensors_base)
            available_tensors.pop(target_node, None)

            start = find_nearest_available_start(
                node_map=node_map,
                node_name=target_node,
                available_nodes=set(available_tensors.keys()),
            )

            tensor_mb = (
                tensor.numel()
                * tensor.element_size()
                / 1024
                / 1024
            )

            if start is None:
                rows.append({
                    "key": key,
                    "tensor_mb": tensor_mb,
                    "start": "",
                    "target": target_node,
                    "path_len": 0,
                    "recompute_ms": "",
                    "missing_profile": "",
                    "path": "",
                })
                continue

            path = build_path_from_start_to_node(
                node_map=node_map,
                start_node=start,
                target_node=target_node,
            )

            if not path:
                continue

            cost_ms, missing_profile = (
                engine.estimate_recompute_cost(path)
            )

            rows.append({
                "key": key,
                "tensor_mb": tensor_mb,
                "start": start,
                "target": target_node,
                "path_len": len(path),
                "recompute_ms": cost_ms,
                "missing_profile": ",".join(missing_profile),
                "path": " -> ".join(path),
            })

        rows.sort(
            key=lambda row: row["tensor_mb"],
            reverse=True,
        )

        with open(menu_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "key",
                    "tensor_mb",
                    "start",
                    "target",
                    "path_len",
                    "recompute_ms",
                    "missing_profile",
                    "path",
                ],
            )

            writer.writeheader()
            writer.writerows(rows)

        print(
            f"[RECOMPUTE_MENU_SAVE] "
            f"path={menu_csv} "
            f"candidates={len(rows)}",
            flush=True,
        )

        return rows
def inject_recomputed_tensors(payload, payload_path, recomputed):
    profile_t0 = time.perf_counter()

    total_bytes = 0

    with nvtx_range("INJECT/python_payload_update"):
        t0 = time.perf_counter()
        for key, tensor in recomputed.items():
            t = tensor.detach().cpu().contiguous()
            payload.tensors[key] = t
            total_bytes += t.numel() * t.element_size()
        t1 = time.perf_counter()
        inject_mem_ms = (t1 - t0) * 1000

    print(
        f"[PY][SET_MEM_PAYLOAD] "
        f"JIN_ROLE={os.environ.get('JIN_ROLE')} "
        f"JIN_STEP={os.environ.get('JIN_STEP')} "
        f"JIN_PAYLOAD_PATH={os.environ.get('JIN_PAYLOAD_PATH')} "
        f"alias_exists={os.path.exists(os.environ.get('JIN_PAYLOAD_PATH', '') + '.alias')}",
        flush=True,
    )

    t0 = time.perf_counter()

    step = int(os.environ.get("JIN_STEP", "0"))

    patch_tensor_ms = 0.0

    with nvtx_range("INJECT/jin_patch_all"):

        for key, tensor in recomputed.items():
            tt0 = time.perf_counter()
            with nvtx_range(
                f"INJECT/Patch/{key}"
            ):
                jin_patch_tensor_from_python(
                    key=key,
                    tensor=tensor,
                    step=step,
                )
                tt1 = time.perf_counter()
                patch_tensor_ms += (tt1 - tt0) * 1000

    t1 = time.perf_counter()

    to_jin1_bytes_ms = 0.0
    set_payload_bytes_ms = patch_tensor_ms

    profile_t1 = time.perf_counter()
    total_ms = (profile_t1 - profile_t0) * 1000

    print(
        f"[B][RECOMPUTE_INJECT_PROFILE] "
        f"n={len(recomputed)} "
        f"mb={total_bytes / 1024 / 1024:.3f} "
        f"inject_mem_ms={inject_mem_ms:.3f} "
        f"to_jin1_bytes_ms={to_jin1_bytes_ms:.3f} "
        f"set_payload_bytes_ms={set_payload_bytes_ms:.3f} "
        f"read_jin1_ms=0.000 "
        f"save_jin1_ms=0.000 "
        f"total_ms={total_ms:.3f} "
        f"keys={list(recomputed.keys())[:10]}",
        flush=True,
    )

def is_recomputable_key(key):
    if key.startswith("graph:conv:") and key.endswith(":input"):
        return True
    if key.startswith("graph:relu:") and key.endswith(":result"):
        return True
    if key.startswith("graph:addmm:") and key.endswith(":mat1"):
        return True
    if key.startswith("graph:bn:") and key.endswith(":input"):
        return True
    if key.startswith("graph:maxpool2d:") and key.endswith(":input"):
        return True
    if key.startswith("graph:maxpool2d:") and key.endswith(":indices"):
        return True

    # BN result1/result2는 bn input에서 special recompute
    # if key.startswith("graph:bn:") and key.endswith(":result1"):
    #     return True
    # if key.startswith("graph:bn:") and key.endswith(":result2"):
    #     return True

    return False


