import torch
import torch.nn as nn
from einops import rearrange


class SimpleSSM(nn.Module):
    def __init__(self, d_model, d_state=8):
        super().__init__()
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_state * 2, bias=False)
        self.out_proj = nn.Linear(d_state, d_model, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A = nn.Parameter(-torch.exp(torch.log(A)))

    def forward(self, x):
        B, L, C = x.shape

        u, v = self.in_proj(x).chunk(2, dim=-1)

        h = torch.zeros(B, self.d_state, device=x.device)
        outputs = []

        decay = torch.sigmoid(self.A)

        for t in range(L):
            h = decay * h + u[:, t]
            outputs.append(h * torch.sigmoid(v[:, t]))

        y = torch.stack(outputs, dim=1)
        return self.out_proj(y)


class Mamba1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.ssm = SimpleSSM(channels)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


class FourDirectionalMamba(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.row_lr = Mamba1D(channels)
        self.row_rl = Mamba1D(channels)
        self.col_tb = Mamba1D(channels)
        self.col_bt = Mamba1D(channels)

        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.01)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        a = rearrange(x, "b c h w -> (b h) w c")
        a = self.row_lr(a)
        a = rearrange(a, "(b h) w c -> b c h w", b=B, h=H)

        b = rearrange(x.flip(-1), "b c h w -> (b h) w c")
        b = self.row_rl(b)
        b = rearrange(b, "(b h) w c -> b c h w", b=B, h=H).flip(-1)

        c = rearrange(x, "b c h w -> (b w) h c")
        c = self.col_tb(c)
        c = rearrange(c, "(b w) h c -> b c h w", b=B, w=W)

        d = rearrange(x.flip(-2), "b c h w -> (b w) h c")
        d = self.col_bt(d)
        d = rearrange(d, "(b w) h c -> b c h w", b=B, w=W).flip(-2)

        out = (a + b + c + d) / 4.0
        out = residual + self.gamma * out

        return self.bn(out)