import torch
import torch.nn.functional as F


def normalize_complex(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    eps = 1e-10
    magnitude = torch.clamp(x.abs(), min=eps)
    angle = torch.angle(x)
    return (beta * magnitude ** alpha).to(torch.complex64) * torch.exp(1j * angle.to(torch.complex64))


def denormalize_complex(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    eps = 1e-10
    x = x / beta
    magnitude = torch.clamp(x.abs(), min=eps, max=1e6)
    angle = torch.angle(x)
    return (magnitude ** (1.0 / alpha)).to(torch.complex64) * torch.exp(1j * angle.to(torch.complex64))


def _stft(wv: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=wv.device, dtype=wv.dtype)
    return torch.stft(wv, n_fft=n_fft, hop_length=hop_length, window=window,
                      center=False, return_complex=True)


def _synthesis_window(n_fft: int, hop_length: int, device, dtype) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=device, dtype=dtype)
    denom = win ** 2
    overlaps = -(-(n_fft) // hop_length)
    denom = F.pad(denom, (0, overlaps * hop_length - n_fft))
    denom = denom.reshape(overlaps, hop_length).sum(0, keepdim=True)
    denom = denom.tile(overlaps, 1).reshape(overlaps * hop_length)
    return win / denom[:n_fft]


def _overlap_and_add(frames: torch.Tensor, hop_length: int) -> torch.Tensor:
    *outer, T, frame_len = frames.shape
    outer_rank = len(outer)
    L = frame_len + hop_length * (T - 1)
    segments = -(-(frame_len) // hop_length)
    frames = F.pad(frames, (0, segments * hop_length - frame_len, 0, segments))
    shape = outer + [T + segments, segments, hop_length]
    frames = frames.reshape(shape)
    perm = list(range(outer_rank)) + [outer_rank + 1, outer_rank, outer_rank + 2]
    frames = frames.permute(perm)
    shape = outer + [(T + segments) * segments, hop_length]
    frames = frames.reshape(shape)
    frames = frames[..., :(T + segments - 1) * segments, :]
    shape = outer + [segments, T + segments - 1, hop_length]
    frames = frames.reshape(shape)
    frames = frames.sum(-3)
    return frames.reshape(outer + [(T + segments - 1) * hop_length])[..., :L]


def _istft(spec: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    x = torch.fft.irfft(spec, n=n_fft, dim=1)
    synth_win = _synthesis_window(n_fft, hop_length, spec.device, torch.float32)
    x = x * synth_win.unsqueeze(-1)
    return _overlap_and_add(x.permute(0, 2, 1), hop_length)


def _encode(spec: torch.Tensor, hop_length: int, alpha: float, beta: float) -> torch.Tensor:
    spec = spec[:, :2 * hop_length, :]
    spec = normalize_complex(spec, alpha, beta)
    return torch.stack((spec.real, spec.imag), dim=1)


def _decode(realimag: torch.Tensor, n_fft: int, alpha: float, beta: float) -> torch.Tensor:
    if realimag.dtype != torch.float32:
        realimag = realimag.float()
    expected_F = n_fft // 2 + 1
    Fbin = realimag.shape[-2]
    if Fbin < expected_F:
        realimag = F.pad(realimag, (0, 0, 0, expected_F - Fbin))
    elif Fbin > expected_F:
        realimag = realimag[..., :expected_F, :]
    return denormalize_complex(torch.complex(realimag[:, 0], realimag[:, 1]), alpha, beta)


def waveform_to_realimag(
    wv: torch.Tensor,
    n_fft: int,
    hop_length: int,
    snippet_samples: int,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """
    Convert a waveform to a real/imaginary spectrogram tensor suitable for the autoencoder.

    The input must be an *extended* waveform of length ``snippet_samples + n_fft``,
    containing ``n_fft // 2`` real-context samples prepended and appended around the
    core ``snippet_samples`` samples (center=False STFT with real padding).

    Args:
        wv: ``(snippet_samples + n_fft,)`` or ``(B, snippet_samples + n_fft)``
    Returns:
        ``(2, F, T)`` or ``(B, 2, F, T)`` float tensor
    """
    is_unbatched = wv.ndim == 1
    if is_unbatched:
        wv = wv.unsqueeze(0)
    if not wv.dtype.is_floating_point:
        wv = wv.float()
    T = snippet_samples // hop_length + 1
    spec = _stft(wv, n_fft, hop_length)
    realimag = _encode(spec[:, :, :T], hop_length, alpha, beta)
    return realimag.squeeze(0) if is_unbatched else realimag


def realimag_to_waveform(
    realimag: torch.Tensor,
    n_fft: int,
    hop_length: int,
    snippet_samples: int,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """
    Convert a real/imaginary spectrogram tensor back to a waveform.

    Args:
        realimag: ``(2, F, T)`` or ``(B, 2, F, T)`` float tensor
    Returns:
        ``(snippet_samples,)`` or ``(B, snippet_samples)`` float tensor
    """
    is_unbatched = realimag.ndim == 3
    if is_unbatched:
        realimag = realimag.unsqueeze(0)
    spec = _decode(realimag, n_fft, alpha, beta)
    wv = _istft(spec, n_fft, hop_length)
    start = n_fft // 2
    wv = wv[..., start: start + snippet_samples]
    wv = wv.clamp(-1.0, 1.0)
    return wv.squeeze(0) if is_unbatched else wv
