import torch
import numpy as np
from tianshou.utils.net.common import MLP, Net


class InputNorm(torch.nn.Module):
    def __init__(
            self,
            num_obs,
            state_shape,
            action_shape=(0,),
            hidden_sizes=(),
            device=torch.device("cuda"),
            concat=False,
    ):
        super().__init__()
        assert len(state_shape) == 1 and len(action_shape) == 1
        self.device = device
        self.state_dim = state_shape[0]
        self.action_dim = action_shape[0]
        self.input_dim = self.state_dim if not concat else self.state_dim + self.action_dim

        # self.model = Net(state_shape, action_shape, hidden_sizes=hidden_sizes, device=device, concat=concat)
        # self.output_dim = self.model.output_dim
        self.obs_len = 6 * num_obs**2
        self.reg_encoder = MLP(self.input_dim - self.obs_len, hidden_sizes=hidden_sizes, device=device)
        self.obs_encoder = MLP(6, hidden_sizes=hidden_sizes, device=device)
        self.decoder = MLP(self.reg_encoder.output_dim + self.obs_encoder.output_dim,
                           hidden_sizes=hidden_sizes, device=device, output_dim=0 if concat else self.action_dim)
        self.output_dim = self.decoder.output_dim

        self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer("mean", torch.zeros(self.input_dim, dtype=torch.float32, device=device))
        self.register_buffer("m2", torch.zeros(self.input_dim, dtype=torch.float32, device=device))
        # self.register_buffer("min", torch.full([shape], float("inf"), dtype=torch.float32, device=net.device))
        # self.register_buffer("max", torch.full([shape], -float("inf"), dtype=torch.float32, device=net.device))

    def forward(self, x, state=None, info={}):
        assert self.count != 0
        x = torch.as_tensor(x, device=self.device)
        mean, std = self.get_mean_std()
        std[(std == 0) | torch.isnan(std)] = 1
        x = (x - mean) / std
        # if self.count > 0:
        #     offset = self.min.detach().clone()
        #     scale = (self.max - self.min) / 2
        #     offset[self.min == self.max] = -1
        #     scale[self.min == self.max] = 1
        #     x = (x - offset) / scale - 1
        # return self.model(x, state)
        reg = torch.cat([x[:, :12], x[:, 12 + self.obs_len:]], dim=-1)
        reg_encoded = self.reg_encoder(reg)
        obs = x[:, 12:12 + self.obs_len]
        if self.obs_len > 0:
            obs_encoded = self.obs_encoder(obs.reshape(-1, self.obs_len // 6, 6))
            obs_encoded = torch.amax(obs_encoded, 1)
        else:
            obs_encoded = torch.zeros(x.shape[0], self.obs_encoder.output_dim, device=self.device)

        x = torch.cat([reg_encoded, obs_encoded], dim=-1)
        result = self.decoder(x)
        return result, state

    def update(self, x):
        x = torch.as_tensor(x, device=self.device)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        # self.min = torch.minimum(self.min, x)
        # self.max = torch.maximum(self.max, x)
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def get_mean_std(self):
        return self.mean, torch.sqrt(torch.abs(self.m2 / self.count))
