import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(num_channels: int, max_groups: int = 32) -> int:
    return min(num_channels // 4, max_groups)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor=(2, 2), kernel_size=(3, 3)):
        super().__init__()
        self.factor = factor
        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=factor, padding=(pad_h, pad_w))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, factor=(2, 2), kernel_size=(3, 3), mode="nearest"):
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,
                              padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, shape=None) -> torch.Tensor:
        x = F.interpolate(x, size=shape, scale_factor=self.factor if shape is None else None, mode=self.mode)
        return self.act(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0, groupnorm_groups=32, norm_eps=1e-6):
        super().__init__()
        self.norm1 = nn.GroupNorm(_gn_groups(in_channels, groupnorm_groups), in_channels, eps=norm_eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_gn_groups(out_channels, groupnorm_groups), out_channels, eps=norm_eps)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual


class FrequencyAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, residual_connections=True,
                 groupnorm_groups=32, norm_eps=1e-6):
        super().__init__()
        self.attention_residual_connections = residual_connections
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=False)
        self.norm = nn.GroupNorm(_gn_groups(in_channels, groupnorm_groups), in_channels, eps=norm_eps)
        mlp_hidden = in_channels * 2
        self.mlp = nn.Sequential(nn.Linear(in_channels, mlp_hidden), nn.SiLU(), nn.Linear(mlp_hidden, in_channels))
        self.ff_norm = nn.GroupNorm(_gn_groups(in_channels, groupnorm_groups), in_channels, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, f, T = x.shape
        residual = x
        x = self.norm(x)
        x = x.permute(0, 3, 2, 1).contiguous().view(B * T, f, C)
        x_q = x.permute(1, 0, 2).contiguous()
        attn_output, _ = self.mha(x_q, x_q, x_q)
        attn_output = attn_output.permute(1, 0, 2).contiguous().view(B, T, f, C)
        x = attn_output.permute(0, 3, 2, 1).contiguous()
        if self.attention_residual_connections:
            x = x + residual
        y = self.ff_norm(x)
        B, C, f, T = y.shape
        y2 = y.permute(0, 3, 2, 1).contiguous().view(B * T * f, C)
        y2 = self.mlp(y2).view(B, T, f, C).permute(0, 3, 2, 1).contiguous()
        return x + y2


class Conv1dResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        ng = _gn_groups(channels, 32)
        self.norm1 = nn.GroupNorm(ng, channels)
        self.norm2 = nn.GroupNorm(ng, channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding="same")
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class Latent1DBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, kernel_size=3):
        super().__init__()
        ng = _gn_groups(in_channels, 32)
        self.prenorm = nn.GroupNorm(ng, in_channels)
        self.proj = nn.Conv1d(in_channels, out_channels, 1)
        self.blocks = nn.ModuleList([Conv1dResBlock(out_channels, kernel_size) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Fd, T = x.shape
        x = x.reshape(B, C * Fd, T)
        x = self.prenorm(x)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return x


class Latent1DBottleneckDec(nn.Module):
    def __init__(self, in_channels, enc_channels, latent_freq, num_layers=2, kernel_size=3):
        super().__init__()
        ng = _gn_groups(enc_channels, 32)
        self.blocks = nn.ModuleList([Conv1dResBlock(in_channels, kernel_size) for _ in range(num_layers)])
        self.proj = nn.Conv1d(in_channels, enc_channels * latent_freq, 1)
        self.postnorm = nn.GroupNorm(ng, enc_channels)
        self.enc_channels = enc_channels
        self.latent_freq = latent_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.proj(x)
        B, _, T = x.shape
        x = x.reshape(B, self.enc_channels, self.latent_freq, T)
        return self.postnorm(x)


class FrequencyWiseScaling(nn.Module):
    def __init__(self, num_freq_bins: int, init_scale: float = 1.0):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, 1, num_freq_bins, 1) * init_scale)

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if inverse:
            return x / (self.gain + 1e-8)
        return x * self.gain
