from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class AudioConfig:
    target_sr: int = 44100
    data_channels: int = 2
    snippet_samples: int = 52736
    num_snippets_per_file: int = 4
    n_fft: int = 2048
    hop_length: int = 512
    alpha_rescale: float = 0.65
    beta_rescale: float = 0.35
    volume_norm_params: Optional[Tuple[float, float]] = None


@dataclass
class AutoencoderConfig:
    use_ema: bool = True
    ema_momentum: float = 0.9999
    in_channels: int = 2
    out_channels: int = 2
    channels_per_block: Tuple[int, ...] = (64, 128, 256, 512, 640)
    residual_layers_per_encoder_block: int = 2
    residual_layers_per_decoder_block: int = 3
    attention_for_block: Tuple[int, ...] = (0, 0, 1, 1, 1)
    attention_residual_connections: bool = True
    attention_heads: int = 4
    up_down_factors: Tuple[Tuple[int, int], ...] = ((4, 2), (4, 2), (2, 2), (2, 1), (1, 1))
    up_down_kernel_size: Tuple[Tuple[int, int], ...] = ((5, 3), (3, 3), (3, 3), (3, 3), (3, 3))
    dropout_p: float = 0.0
    groupnorm_groups: int = 32
    norm_eps: float = 1e-6
    bottleneck_1d_channels: int = 64
    bottleneck_1d_layers: int = 4
    bottleneck_1d_kernel_size: int = 3
