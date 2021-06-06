import torch
import numpy as np
from tianshou.utils.net.common import Net


class InputNorm(torch.nn.Module):
    def __init__(self, net: Net):
        super().__init__()
        self.net = net
        self.output_dim = self.net.output_dim
        shape = net.model.model[0].in_features
        self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=net.device))
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32, device=net.device))
        self.register_buffer("m2", torch.zeros(shape, dtype=torch.float32, device=net.device))
        # self.register_buffer("min", torch.full([shape], float("inf"), dtype=torch.float32, device=net.device))
        # self.register_buffer("max", torch.full([shape], -float("inf"), dtype=torch.float32, device=net.device))

    def forward(self, x, state=None, info={}):
        x = torch.as_tensor(x, device=self.net.device)
        mean, std = self.get_mean_std()
        std[(std == 0) | torch.isnan(std)] = 1
        x = (x - mean) / std
        # if self.count > 0:
        #     offset = self.min.detach().clone()
        #     scale = (self.max - self.min) / 2
        #     offset[self.min == self.max] = -1
        #     scale[self.min == self.max] = 1
        #     x = (x - offset) / scale - 1
        result = self.net.forward(x, state, info)
        return result

    def update(self, x):
        x = torch.as_tensor(x, device=self.net.device)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        # self.min = torch.minimum(self.min, x)
        # self.max = torch.maximum(self.max, x)
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def get_mean_std(self):
        return self.mean, torch.sqrt(torch.abs(self.m2 / self.count))
