import torch 

class ReplayEngine():
    def __init__(self, model):
        self.model = model
    def dummy_forward(self,x_dummy):
        self.model.train()
        self.model.zero_grad(set_to_none=True)

        x_dummy = x_dummy.detach().cpu().clone().requires_grad_(True)
        out = self.model(x_dummy)

        return out, x_dummy
    
    def backward(self, logits, y, loss_fn):
        loss = loss_fn(logits,y)
        loss.backward()
        return loss