from pathlib import Path
from typing import List, Union, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import yaml
from huggingface_hub import hf_hub_download

from .modules import AudioConfig, AutoencoderConfig, Autoencoder, EMA
from .audio import waveform_to_realimag, realimag_to_waveform

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "autoencoder_44k_1p2s_topo_1d.yaml"

# Hub repo holding every released checkpoint as <variant>.pt at the repo root.
HF_REPO = "malex1106/talc"

VARIANTS = ("baseline", "isotropic", "talc")

DEFAULT_CHUNK_SEC = 60.0


class TALC:
    """
    TALC audio codec — encode audio to latents and decode back to audio.

    Weights are downloaded from the Hugging Face Hub on first use, or loaded from a
    local checkpoint.  Then use ``encode``, ``decode``, or ``reconstruct``.

    Example::

        model = TALC(variant="talc")               # downloads + caches
        model = TALC("talc.pt")                    # or a local checkpoint
        waveform = model.reconstruct("song.wav", output_path="out.wav")

        wav     = model.load_audio("song.wav")       # mono, at model.sample_rate
        latents = model.encode(wav)                  # (C, T_lat)
        audio   = model.decode(latents, len(wav))    # (len(wav),)

        # Shorter chunks than the 60 s default use less memory
        latents = model.encode(wav, chunk_sec=10.0)
        audio   = model.decode(latents, len(wav), chunk_sec=10.0)
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None,
                 variant: str = "talc", device: Optional[str] = None):
        """
        Args:
            model_path: a checkpoint file, or a directory containing ``ema_shadow.pt``
                        (and optionally ``config.yaml``).  When omitted, ``<variant>.pt``
                        is downloaded from the Hugging Face Hub and cached locally.
            variant: released checkpoint to download — one of ``"talc"``,
                     ``"isotropic"``, or ``"baseline"``.  Ignored when ``model_path``
                     is given.
            device: torch device string, e.g. ``"cuda"`` or ``"cpu"``.
                    Defaults to CUDA if available.

        A ``config.yaml`` in a model directory overrides the bundled ``audio`` and
        ``autoencoder`` settings.
        """
        if model_path is None:
            if variant not in VARIANTS:
                raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")
            ckpt_path = Path(hf_hub_download(HF_REPO, f"{variant}.pt"))
            config_path = None
        elif Path(model_path).is_dir():
            ckpt_path = Path(model_path) / "ema_shadow.pt"
            config_path = Path(model_path) / "config.yaml"
        else:
            ckpt_path = Path(model_path)
            config_path = None

        with open(DEFAULT_CONFIG) as f:
            cfg = yaml.safe_load(f)

        if config_path is not None and config_path.exists():
            with open(config_path) as f:
                model_cfg = yaml.safe_load(f) or {}
            for section in ("audio", "autoencoder"):
                cfg[section].update(model_cfg.pop(section, None) or {})
            cfg.update(model_cfg)

        audio_cfg = AudioConfig(**cfg["audio"])
        ae_cfg = AutoencoderConfig(**cfg["autoencoder"])
        mixed_precision = cfg.get("mixed_precision", True)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if mixed_precision else torch.float32

        # Build model and apply EMA weights
        self.autoencoder = Autoencoder(ae_cfg, audio_cfg).eval()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # EMA.load_state_dict silently skips absent names, which would leave random weights.
        missing = [n for n, _ in self.autoencoder.named_parameters() if n not in state]
        if missing:
            raise RuntimeError(
                f"{ckpt_path} is missing {len(missing)} parameters (e.g. {missing[0]!r}); "
                f"it is not a TALC EMA checkpoint for this architecture."
            )
        ema = EMA(self.autoencoder, ae_cfg.ema_momentum)
        ema.load_state_dict(state)
        ema.apply_shadow()
        del ema

        self.autoencoder = self.autoencoder.to(device=self.device, dtype=self.dtype)

        self.sample_rate: int = audio_cfg.target_sr
        self._n_fft: int = audio_cfg.n_fft
        self._hop: int = audio_cfg.hop_length
        self._alpha: float = audio_cfg.alpha_rescale
        self._beta: float = audio_cfg.beta_rescale

        # chunk_samples -> (input_shapes, T_lat); filled by encode or _get_shapes_for_chunk.
        self._shapes_cache: dict = {}

    @classmethod
    def from_pretrained(cls, model_path: Optional[Union[str, Path]] = None,
                        variant: str = "talc", device: Optional[str] = None) -> "TALC":
        """Alias for the constructor — load a pretrained TALC model."""
        return cls(model_path, variant, device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_shapes_for_chunk(self, chunk_samples: int) -> Tuple[list, int]:
        """Return (input_shapes, T_lat) for the given chunk size, using a cache."""
        if chunk_samples not in self._shapes_cache:
            F_bins = self._hop * 2
            T_stft = chunk_samples // self._hop + 1
            # TODO: derive the shapes from the encoder's conv kernel/stride/padding instead of
            # running this dummy forward pass.
            dummy = torch.zeros(1, 2, F_bins, T_stft, device=self.device, dtype=self.dtype)
            with torch.no_grad():
                z, input_shapes = self.autoencoder.encode(dummy)
            self._shapes_cache[chunk_samples] = (input_shapes, z.shape[-1])
        return self._shapes_cache[chunk_samples]

    def _chunking(self, n: int, chunk_sec: float) -> Tuple[int, List[int]]:
        """Chunk size and start offsets; short audio is one chunk, and the last chunk ends at ``n``."""
        chunk = min(int(chunk_sec * self.sample_rate), n)
        starts = list(range(0, n - chunk + 1, chunk))
        if starts[-1] + chunk < n:
            starts.append(n - chunk)
        return chunk, starts

    def _make_ext_chunks(self, waveform: torch.Tensor, chunk: int, starts: List[int]) -> List[torch.Tensor]:
        """Cut ``chunk``-sample segments at ``starts``, with n_fft//2 real context on each side."""
        context = self._n_fft // 2
        ext_len = chunk + self._n_fft
        ext_chunks = []
        for start in starts:
            ext_start = start - context
            pre_pad = max(0, -ext_start)
            load_start = max(0, ext_start)
            seg = waveform[load_start: load_start + ext_len - pre_pad]
            if pre_pad > 0:
                seg = F.pad(seg, (pre_pad, 0))
            if seg.size(0) < ext_len:
                seg = F.pad(seg, (0, ext_len - seg.size(0)))
            ext_chunks.append(seg)

        return ext_chunks

    def _ri_from_ext(self, ext_wv: torch.Tensor, chunk_samples: int) -> torch.Tensor:
        """Extended waveform → (1, 2, F, T) spectrogram tensor on device."""
        ri = waveform_to_realimag(
            ext_wv.to(self.device),
            n_fft=self._n_fft,
            hop_length=self._hop,
            snippet_samples=chunk_samples,
            alpha=self._alpha,
            beta=self._beta,
        )
        return ri.to(self.dtype).unsqueeze(0)

    def _wav_from_ri(self, ri: torch.Tensor, chunk_samples: int) -> torch.Tensor:
        """(1, 2, F, T) or (2, F, T) reconstruction tensor → (chunk_samples,) waveform."""
        if ri.ndim == 4:
            ri = ri.squeeze(0)
        return realimag_to_waveform(
            ri.float(),
            n_fft=self._n_fft,
            hop_length=self._hop,
            snippet_samples=chunk_samples,
            alpha=self._alpha,
            beta=self._beta,
        ).cpu()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_audio(self, path: Union[str, Path]) -> torch.Tensor:
        """Load an audio file as a mono float32 waveform at ``sample_rate``."""
        waveform, sr = torchaudio.load(str(path))  # (C, L)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.size(0) > 1:
            waveform = waveform.mean(0)
        else:
            waveform = waveform.squeeze(0)
        return waveform.clamp(-1.0, 1.0)

    def encode(
        self,
        audio: Union[str, Path, torch.Tensor],
        sr: Optional[int] = None,
        chunk_sec: float = DEFAULT_CHUNK_SEC,
    ) -> torch.Tensor:
        """
        Encode audio to latents.

        Args:
            audio: path to an audio file, or a 1D float32 waveform tensor.
            sr: sample rate of the waveform tensor (ignored when ``audio`` is a path).
                Required if ``sr != model.sample_rate``.
            chunk_sec: chunk duration in seconds.  Smaller values use less memory
                       but add chunk boundaries.  Must match ``chunk_sec``
                       passed to ``decode`` when decoding separately.

        Returns:
            Latent tensor of shape ``(C, T_lat)``, float32 on CPU.  Chunks are
            concatenated along the last axis; audio shorter than one chunk is
            encoded in one piece, and for longer audio the last chunk overlaps
            the one before it.
        """
        if isinstance(audio, (str, Path)):
            waveform = self.load_audio(audio)
        else:
            waveform = audio.float()
            if sr is not None and sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        chunk, starts = self._chunking(waveform.size(0), chunk_sec)
        all_latents = []
        for ext_wv in self._make_ext_chunks(waveform, chunk, starts):
            ri = self._ri_from_ext(ext_wv, chunk)
            with torch.no_grad():
                z, input_shapes = self.autoencoder.encode(ri)
            all_latents.append(z.squeeze(0).float().cpu())
        self._shapes_cache[chunk] = (input_shapes, z.shape[-1])

        return torch.cat(all_latents, dim=-1)

    def decode(
        self,
        latents: torch.Tensor,
        length: int,
        chunk_sec: float = DEFAULT_CHUNK_SEC,
    ) -> torch.Tensor:
        """
        Decode latents to a waveform.

        Args:
            latents: tensor as returned by ``encode`` — shape ``(C, T_lat)``.
            length: number of samples that were encoded; determines the chunk layout.
            chunk_sec: chunk duration in seconds used when encoding; must match
                       the value passed to ``encode``.

        Returns:
            Mono waveform ``(length,)`` at ``model.sample_rate``, float32 on CPU.
        """
        chunk, starts = self._chunking(length, chunk_sec)
        input_shapes, T_lat = self._get_shapes_for_chunk(chunk)
        if latents.shape[-1] != len(starts) * T_lat:
            raise ValueError(
                f"Expected {len(starts) * T_lat} latent frames for length={length} and "
                f"chunk_sec={chunk_sec}, got {latents.shape[-1]}."
            )

        wav_chunks = []
        for i, start in enumerate(starts):
            z = latents[..., i * T_lat:(i + 1) * T_lat]
            z = z.to(device=self.device, dtype=self.dtype).unsqueeze(0)
            with torch.no_grad():
                recon = self.autoencoder.decode(z, input_shapes)
            overlap = starts[i - 1] + chunk - start if i > 0 else 0
            wav_chunks.append(self._wav_from_ri(recon, chunk)[overlap:])

        return torch.cat(wav_chunks, dim=0)

    def reconstruct(
        self,
        audio: Union[str, Path, torch.Tensor],
        sr: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        chunk_sec: float = DEFAULT_CHUNK_SEC,
    ) -> torch.Tensor:
        """
        Encode then decode audio (full round-trip reconstruction).

        The output is trimmed to the exact original audio length.

        Args:
            audio: path to an audio file, or a 1D waveform tensor.
            sr: sample rate of the waveform tensor (if applicable).
            output_path: if given, write the reconstruction to this WAV file.
            chunk_sec: chunk duration in seconds.

        Returns:
            Mono waveform ``(L,)`` at ``model.sample_rate``, float32 on CPU.
        """
        if isinstance(audio, (str, Path)):
            waveform = self.load_audio(audio)
        else:
            waveform = audio.float()
            if sr is not None and sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        result = self.decode(self.encode(waveform, chunk_sec=chunk_sec), waveform.size(0), chunk_sec)

        if output_path is not None:
            sf.write(str(output_path), result.numpy(), self.sample_rate)

        return result
