import torch


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model, momentum):
        self.model = model
        self.momentum = momentum
        self.shadow = {}
        self.backup = {}
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.float().clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.momentum) * param.data.float() + self.momentum * self.shadow[name]
                self.shadow[name] = new_average

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.data.dtype)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for name in self.shadow:
            if name in state_dict:
                self.shadow[name] = state_dict[name].float().clone().to(self.shadow[name].device)
