# forward hook , saved tensor hook 관리
# module forward hook 등록 
# layer 이름, input / output shape 기록 

class ModuleTrace:
    def __init__(self):
        self.records = []
        self.handles = []
    def attach(self, model):
        for name, module in model.named_modules():
            if name == "":
                continue

            handle = module.register_forward_hook(
                self._make_hook(name,module)
            )
            self.handles.append(handle)

    def _make_hook(self, name, module):

        def hook(mod, inputs, output):

            in_shape = None
            out_shape = None

            if len(inputs) > 0 and hasattr(inputs[0],"shape"):
                in_shape = tuple(inputs[0].shape)
            if hasattr(output,"shape"):
                out_shape = tuple(output.shape)
            
            self.records.append({
                "name": name,
                "type": module.__class__.__name__,
                "input_shape": in_shape,
                "output_shape": out_shape,
            })

        return hook
    
    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        
    def clear(self):
        self.records.clear()