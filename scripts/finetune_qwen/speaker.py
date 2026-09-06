"""Extract local Qwen ECAPA embeddings and conservatively reject speaker changes."""

import json
from pathlib import Path

import numpy as np

RATE = 24000


def unit(vector: np.ndarray) -> np.ndarray:
    """Normalize a finite, nonzero speaker embedding."""
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValueError("Invalid speaker embedding")
    return vector / norm


def windows(audio: np.ndarray, seconds: float = 3.0) -> list[np.ndarray]:
    """Cover the entire clip with overlapping windows, including its last samples."""
    size = round(seconds * RATE)
    if audio.ndim != 1 or len(audio) < RATE or not np.isfinite(audio).all():
        raise ValueError("Speaker audio must be finite mono audio of at least one second")
    if len(audio) <= size:
        return [audio]
    starts = sorted(set(range(0, len(audio) - size + 1, size // 2)) | {len(audio) - size})
    return [audio[start : start + size] for start in starts]


class SpeakerEncoder:
    """Load only the frozen speaker encoder, never the multi-gigabyte talker."""

    def __init__(self, base: Path, device: str = "cpu"):
        import torch
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
        from safetensors import safe_open

        torch.set_num_threads(4)
        config = json.loads((base / "config.json").read_text())["speaker_encoder_config"]
        self.model = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**config))
        weights = {}
        with safe_open(base / "model.safetensors", framework="pt", device="cpu") as saved:
            names = saved.keys()
            for name in names:
                if name.startswith("speaker_encoder."):
                    value = saved.get_tensor(name)
                    if value.ndim == 3:
                        value = value.transpose(1, 2).contiguous()
                    weights[name.removeprefix("speaker_encoder.")] = value.float()
        self.model.load_state_dict(weights, strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.device = device

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Encode unchanged 24 kHz mono samples using the upstream mel front end."""
        import torch
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        windows(audio)  # Validate before calling native kernels.
        with torch.inference_mode():
            mels = mel_spectrogram(
                torch.from_numpy(np.asarray(audio, dtype=np.float32))[None],
                n_fft=1024,
                num_mels=128,
                sampling_rate=RATE,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=12000,
            ).transpose(1, 2)
            return unit(self.model(mels.to(self.device)).cpu().numpy())


def speaker_scores(embedding: np.ndarray, targets: np.ndarray, negatives: np.ndarray) -> dict:
    """Require similarity to target references and separation from other voices."""
    vector = unit(embedding)
    positive = float(np.max(targets @ vector))
    negative = float(np.max(negatives @ vector))
    return {"target": positive, "negative": negative, "margin": positive - negative}


def accepted(scores: list[dict], *, minimum_target: float, minimum_margin: float) -> bool:
    """Every full-clip/window decision must pass; averaging cannot hide a guest."""
    return bool(scores) and all(
        np.isfinite([score["target"], score["negative"], score["margin"]]).all()
        and score["target"] >= minimum_target
        and score["margin"] >= minimum_margin
        for score in scores
    )
