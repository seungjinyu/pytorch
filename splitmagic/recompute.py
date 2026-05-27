class ReplaceModuleInput:
    def __init__(self, module, replacement):
        self.module = module
        self.replacement = replacement
        self.handle = None

    def __enter__(self):
        def pre_hook(module, inputs):
            return (self.replacement,)

        self.handle = self.module.register_forward_pre_hook(
            pre_hook,
            prepend=True,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()