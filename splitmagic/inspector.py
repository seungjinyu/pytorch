# saved tensor 자동 조사 
# torch.autograd.graph.saved_tensors_hook 사용
# pytorch 가 backward 용으로 저장한 tensor 목록 추적 

from torch.autograd.graph import saved_tensors_hooks
from .hooks import ModuleTrace

class SavedTensorInfo:
    def __init__(self, index, tensor):
        self.index = index 
        self.shape = tuple(tensor.shape)
        self.dtype = str(tensor.dtype)
        self.device = str(tensor.device)
        self.requires_grad = tensor.requires_grad
        self.tensor = tensor.detach().cpu().clone()
    def __repr__(self):
        return (
            f"[{self.index}] "
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"device={self.device}, "
            f"requires_grad={self.requires_grad} "
        
        )

class InspectReport:
    def __init__(self, saved_tensors, module_trace):
        self.saved_tensors = saved_tensors
        self.module_trace = module_trace

    def print(self):
        print("=== saved tensors ===")
        for info in self.saved_tensors:
            print(info)
        print("\n=== module trace ===")

        for r in self.module_trace:
            print(
                f"{r['name']} ({r['type']}): "
                f"in={r['input_shape']} -> out={r['output_shape']}"
            )

def inspect_saved_tensors(model, x, y, loss_fn ):

    saved_infos = []
    tracer = ModuleTrace()

    def pack_hook(tensor):
        idx = len(saved_infos)
        saved_infos.append(SavedTensorInfo(idx, tensor))
        return tensor
    
    def unpack_hook(tensor):
        return tensor
    
    model.zero_grad(set_to_none=True)
    tracer.attach(model)

    try:
        with saved_tensors_hooks(pack_hook, unpack_hook):
            out = model(x)
            loss = loss_fn(out, y)
        loss.backward()
    finally:
        tracer.detach()
        
    return InspectReport(
        saved_tensors =saved_infos, 
        module_trace  = tracer.records
    )