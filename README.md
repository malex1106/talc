# TALC

**Topology-Aware Latent Corruption** — a convolutional audio autoencoder
operating on complex STFT representations.

TALC encodes 44.1 kHz mono audio into a compact 1D latent sequence of **64 channels at
~10.8 Hz** (a 64× reduction versus raw mono samples) and
decodes it back to a waveform.

This package contains inference code only.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10, `torch` ≥ 2.1, `torchaudio` ≥ 2.1, `soundfile`, `pyyaml`, and
`huggingface_hub`. For the demo notebook, install the extras with
`pip install -e ".[notebook]"`.

## Quick start

```python
from talc import TALC

model = TALC(variant="talc")   # weights download on first use, then cache

# Full round-trip
waveform = model.reconstruct("song.wav", output_path="reconstruction.wav")

# Or encode and decode separately
wav     = model.load_audio("song.wav")     # mono, at model.sample_rate
latents = model.encode(wav)                # (64, T_lat), float32 on CPU
audio   = model.decode(latents, len(wav))  # (len(wav),)
```

[`demo.ipynb`](demo.ipynb) walks through loading a model and encoding/decoding an audio
sample, with audio players to compare input and output.

`encode` accepts either a path or a 1D waveform tensor. Audio is converted to mono and
resampled to 44.1 kHz automatically; pass `sr=` when handing in a tensor that isn't
already at the model's sample rate. `decode` needs the number of encoded samples, which
tells it how the audio was chunked.

## Released checkpoints

All released models share one architecture and differ only in the latent corruption used
during training. Select one with `variant`:

| `variant` | Training-time latent corruption |
|---|---|
| `"talc"` | Topology-aware channel-wise corruption (γ=2, w_min=0.1), t ~ U(0, 0.25) — the proposed model |
| `"isotropic"` | Uniform across all latent dimensions, t ~ U(0, 0.1) |
| `"baseline"` | None |

Corruption was applied to 99% of training examples and is inactive at inference, so the
three differ only in their weights.

Weights live in a single Hub repo, [`malex1106/talc`](https://huggingface.co/malex1106/talc),
and are downloaded and cached on first use (under `HF_HOME`, like any other Hub model):

```
malex1106/talc
├── baseline.pt
├── isotropic.pt
└── talc.pt
```

To use a local checkpoint instead, pass the `.pt` file, or a directory containing
the model weights (`variant` is then ignored):

```python
model = TALC("talc.pt")
```

An optional `config.yaml` in the model directory overrides the bundled `audio` and
`autoencoder` settings, for checkpoints with a different architecture.

## Choosing the chunk size

Audio is processed in chunks of 60 s by default. Audio shorter than one chunk is processed
in one piece at its own length, without padding. For longer audio, the last chunk is shifted
back to end at the final sample, overlapping the chunk before it; `decode` drops the overlap.
The model is fully convolutional, so if 60 s chunks don't fit in memory, use shorter ones,
at the cost of more chunk boundaries:

```python
latents = model.encode(wav, chunk_sec=10.0)
audio   = model.decode(latents, len(wav), chunk_sec=10.0)

# or in a single call
audio = model.reconstruct(wav, chunk_sec=10.0)
```

`decode` must be given the same `chunk_sec` used for `encode`. Because of the overlapping
last chunk, the latents of audio longer than one chunk repeat up to one chunk's worth of
frames.

## API

- `TALC(model_path=None, variant="talc", device=None)`

Load a checkpoint. `model_path` may be a `.pt` file or a directory containing
`ema_shadow.pt`; with no `model_path`, `<variant>.pt` is downloaded from the Hub and
cached. `device` defaults to CUDA when available. Also available as
`TALC.from_pretrained(...)`.

Attributes: `sample_rate`, `device`, `dtype`, `autoencoder`.

- `load_audio(path) -> Tensor`

Load an audio file as a mono float32 waveform at `sample_rate`.

- `encode(audio, sr=None, chunk_sec=60.0) -> Tensor`

Encode to latents of shape `(64, T_lat)`, float32 on CPU, covering the whole input.

- `decode(latents, length, chunk_sec=60.0) -> Tensor`

Decode latents to a mono waveform of `length` samples at `sample_rate`. `length` is the
number of samples that were encoded, and `chunk_sec` must match the value used for
`encode`.

- `reconstruct(audio, sr=None, output_path=None, chunk_sec=60.0) -> Tensor`

Encode and decode in one call, equivalent to `decode(encode(audio), len(audio))`. Writes a
WAV file when `output_path` is given.

## Notes

Chunks are encoded with `n_fft // 2` samples of real audio context on each side, so the
STFT sees true signal rather than zero padding at chunk edges. Decoded chunks are
concatenated directly.

Inference runs in bfloat16 by default, matching training. Latents are always returned in
float32.

<!--
## Citation

```bibtex
@inproceedings{talc,
  title     = {Topology-Aware Latent Corruption: Can Noise Order Audio Autoencoder Latents?},
  author    = {Fichtinger, Alexander and Schl{\"u}ter, Jan and Widmer, Gerhard},
  year      = {2026},
}
```
-->
