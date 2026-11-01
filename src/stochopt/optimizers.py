"""Optimizer implementations used in the benchmark."""

import torch
from torch.optim import Optimizer


class MomentumSGD(Optimizer):
    """SGD with coupled L2 weight decay and classical momentum.

    Update: g = grad + weight_decay * parameter;
    velocity = momentum * velocity + g; parameter -= lr * velocity.
    The first velocity is g, matching torch.optim.SGD without dampening or Nesterov.
    """

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("MomentumSGD does not support sparse gradients")
                if group["weight_decay"]:
                    gradient = gradient.add(parameter, alpha=group["weight_decay"])
                if group["momentum"]:
                    state = self.state[parameter]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = gradient.clone()
                    else:
                        state["momentum_buffer"].mul_(group["momentum"]).add_(gradient)
                    gradient = state["momentum_buffer"]
                parameter.add_(gradient, alpha=-group["lr"])
        return loss
