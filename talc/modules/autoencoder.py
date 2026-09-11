import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DownSampleBlock, UpSampleBlock, ResidualBlock, \
    FrequencyAttention, FrequencyWiseScaling, \
    Latent1DBottleneck, Latent1DBottleneckDec, _gn_groups
from .config import AudioConfig, AutoencoderConfig


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_blocks=2, add_downsample=True,
                 downsample_factor=(2, 2), downsample_kernel_size=(3, 3), add_attention=True,
                 attention_num_heads=4, attention_residual_connections=True, dropout_p=0.0,
                 groupnorm_groups=32, norm_eps=1e-6):
        super().__init__()
        self.add_attention = add_attention
        self.add_downsample = add_downsample
        self.blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                groupnorm_groups=groupnorm_groups,
                dropout_p=dropout_p,
                norm_eps=norm_eps,
            ) for i in range(num_residual_blocks)
        ])
        if add_downsample:
            self.downsample = DownSampleBlock(out_channels, out_channels,
                                              factor=downsample_factor, kernel_size=downsample_kernel_size)
        if add_attention:
            self.attention = FrequencyAttention(
                in_channels=out_channels, num_heads=attention_num_heads,
                residual_connections=attention_residual_connections, norm_eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.add_downsample:
            x = self.downsample(x)
        if self.add_attention:
            x = self.attention(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels_per_block=(32, 64, 128, 256),
                 residual_layers_per_block=2, add_attention_per_block=(False, False, True, True),
                 attention_num_heads=4, attention_residual_connections=True, dropout_p=0.0,
                 groupnorm_groups=32, norm_eps=1e-6, downsample_factor=((2, 2),),
                 downsample_kernel_size=((3, 3),)):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels_per_block[0], kernel_size=3, padding=1)
        self.encoder_blocks = nn.ModuleList()
        ch = channels_per_block[0]
        for i, out_ch in enumerate(channels_per_block):
            is_last = i == len(channels_per_block) - 1
            self.encoder_blocks.append(EncoderBlock(
                in_channels=ch, out_channels=out_ch,
                num_residual_blocks=residual_layers_per_block,
                add_downsample=not is_last,
                downsample_factor=tuple(downsample_factor[i]),
                downsample_kernel_size=tuple(downsample_kernel_size[i]),
                add_attention=add_attention_per_block[i],
                attention_num_heads=attention_num_heads,
                attention_residual_connections=attention_residual_connections,
                dropout_p=dropout_p, groupnorm_groups=groupnorm_groups,
                norm_eps=norm_eps,
            ))
            ch = out_ch
        self.out_norm = nn.GroupNorm(num_channels=ch, num_groups=_gn_groups(ch, groupnorm_groups), eps=norm_eps)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List]:
        x = self.conv_in(x)
        shapes = [x.shape[-2:]]
        for block in self.encoder_blocks:
            shapes.append(x.shape[-2:])
            x = block(x)
        shapes.pop()
        x = self.conv_out(self.act(self.out_norm(x)))
        return x, shapes


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_blocks=2, add_upsample=True,
                 upsample_factor=(2, 2), upsample_kernel_size=(3, 3), add_attention=True,
                 attention_num_heads=4, attention_residual_connections=True, dropout_p=0.0,
                 groupnorm_groups=32, norm_eps=1e-6):
        super().__init__()
        self.add_attention = add_attention
        self.add_upsample = add_upsample
        self.blocks = nn.ModuleList([
            ResidualBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                groupnorm_groups=groupnorm_groups,
                dropout_p=dropout_p,
                norm_eps=norm_eps,
            ) for i in range(num_residual_blocks)
        ])
        if add_upsample:
            self.upsample = UpSampleBlock(out_channels, out_channels,
                                          factor=upsample_factor, kernel_size=upsample_kernel_size)
        if add_attention:
            self.attention = FrequencyAttention(
                in_channels=out_channels, num_heads=attention_num_heads,
                residual_connections=attention_residual_connections, norm_eps=norm_eps)

    def forward(self, x: torch.Tensor, input_shape=None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.add_upsample:
            x = self.upsample(x, shape=input_shape)
        if self.add_attention:
            x = self.attention(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, channels_per_block=(256, 128, 64, 32),
                 residual_layers_per_block=2, add_attention_per_block=(True, True, False, False),
                 attention_num_heads=4, attention_residual_connections=True, dropout_p=0.0,
                 groupnorm_groups=32, norm_eps=1e-6, upsample_factor=((2, 2),),
                 upsample_kernel_size=((3, 3),)):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels_per_block[0], kernel_size=3, padding=1)
        self.decoder_blocks = nn.ModuleList()
        ch = channels_per_block[0]
        for i, out_ch in enumerate(channels_per_block):
            is_last = i == len(channels_per_block) - 1
            self.decoder_blocks.append(DecoderBlock(
                in_channels=ch, out_channels=out_ch,
                num_residual_blocks=residual_layers_per_block,
                add_upsample=not is_last,
                upsample_factor=tuple(upsample_factor[i]),
                upsample_kernel_size=tuple(upsample_kernel_size[i]),
                add_attention=add_attention_per_block[i],
                attention_num_heads=attention_num_heads,
                attention_residual_connections=attention_residual_connections,
                dropout_p=dropout_p, groupnorm_groups=groupnorm_groups,
                norm_eps=norm_eps,
            ))
            ch = out_ch
        self.out_norm = nn.GroupNorm(num_channels=ch, num_groups=_gn_groups(ch, groupnorm_groups), eps=norm_eps)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor, input_shapes=None) -> torch.Tensor:
        x = self.conv_in(x)
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, input_shape=input_shapes[i])
        x = self.conv_out(self.act(self.out_norm(x)))
        return x


class Autoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, audio_config: AudioConfig):
        super().__init__()
        freq_ds = math.prod(f[0] for f in config.up_down_factors)
        latent_freq = (audio_config.hop_length * 2) // freq_ds
        enc_ch = config.channels_per_block[-1]

        self.encoder = Encoder(
            in_channels=config.in_channels,
            out_channels=enc_ch,
            channels_per_block=config.channels_per_block,
            residual_layers_per_block=config.residual_layers_per_encoder_block,
            add_attention_per_block=config.attention_for_block,
            attention_num_heads=config.attention_heads,
            attention_residual_connections=config.attention_residual_connections,
            dropout_p=config.dropout_p,
            groupnorm_groups=config.groupnorm_groups,
            norm_eps=config.norm_eps,
            downsample_factor=config.up_down_factors,
            downsample_kernel_size=config.up_down_kernel_size,
        )

        self.decoder = Decoder(
            in_channels=enc_ch,
            out_channels=config.out_channels,
            channels_per_block=config.channels_per_block[::-1],
            residual_layers_per_block=config.residual_layers_per_decoder_block,
            add_attention_per_block=config.attention_for_block[::-1],
            attention_num_heads=config.attention_heads,
            attention_residual_connections=config.attention_residual_connections,
            dropout_p=config.dropout_p,
            groupnorm_groups=config.groupnorm_groups,
            norm_eps=config.norm_eps,
            upsample_factor=config.up_down_factors[::-1],
            upsample_kernel_size=config.up_down_kernel_size[::-1],
        )

        self.freq_scaling_encoder = FrequencyWiseScaling(
            num_freq_bins=audio_config.hop_length * 2, init_scale=1.0,
        )

        self.bottleneck_enc = Latent1DBottleneck(enc_ch * latent_freq, config.bottleneck_1d_channels,
                                                 num_layers=config.bottleneck_1d_layers,
                                                 kernel_size=config.bottleneck_1d_kernel_size)
        self.bottleneck_dec = Latent1DBottleneckDec(config.bottleneck_1d_channels, enc_ch, latent_freq,
                                                    num_layers=config.bottleneck_1d_layers,
                                                    kernel_size=config.bottleneck_1d_kernel_size)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """Spectrogram ``(B, 2, F, T)`` → latents ``(B, C, T_lat)`` and the decoder's target shapes."""
        z, input_shapes = self.encoder(self.freq_scaling_encoder(x))
        z = self.bottleneck_enc(z)
        # Latents were layer-normalized per frame (no affine) during training; the decoder expects it.
        z = F.layer_norm(z.transpose(1, 2), z.shape[1:2]).transpose(1, 2)
        return z, input_shapes

    def decode(self, z: torch.Tensor, input_shapes: List) -> torch.Tensor:
        return self.decoder(self.bottleneck_dec(z), input_shapes=input_shapes[::-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, input_shapes = self.encode(x)
        return self.decode(z, input_shapes)[..., :x.shape[-2], :x.shape[-1]]
