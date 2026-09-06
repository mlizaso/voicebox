"""Discover operator-installed local Qwen voices without accepting remote paths."""

import hashlib
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

from .. import config

VOICE_PREFIX = "finetuned:"


def is_finetuned_profile(profile) -> bool:
    """Identify a local checkpoint bound to an existing CustomVoice profile."""
    return (
        getattr(profile, "voice_type", None) == "preset"
        and getattr(profile, "preset_engine", None) == "qwen_custom_voice"
        and str(getattr(profile, "preset_voice_id", "") or "").startswith(VOICE_PREFIX)
    )


def file_sha256(path: Path) -> str:
    """Hash a local model artifact without reading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def registry_directory() -> Path:
    """Return the local-only registry; the HTTP API cannot write model paths."""
    return config.get_data_dir() / "finetuned_voices"


def read_voice(voice_id: str, *, verify_weights: bool = False) -> dict:
    """Resolve a registered voice and reject changed or missing checkpoints."""
    slug = voice_id.removeprefix(VOICE_PREFIX)
    if not voice_id.startswith(VOICE_PREFIX) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", slug):
        raise ValueError("Invalid fine-tuned voice identifier")
    registration = registry_directory() / f"{slug}.json"
    if not registration.is_file():
        raise ValueError(f"Fine-tuned voice is not installed: {voice_id}")
    spec = json.loads(registration.read_text(encoding="utf-8"))
    if spec.get("voice_id") != voice_id:
        raise ValueError(f"Fine-tuned voice registration does not match {voice_id}")
    return validate_checkpoint(spec, verify_weights=verify_weights)


def validate_checkpoint(spec: dict, *, verify_weights: bool = False) -> dict:
    """Validate an operator-selected checkpoint before registering or loading it."""
    voice_id = spec["voice_id"]
    model = Path(spec["model_path"])
    if not model.is_absolute() or not model.is_dir():
        raise ValueError(f"Invalid or missing checkpoint for {voice_id}")
    manifest_path = model / "voicebox_finetune.json"
    if file_sha256(manifest_path) != spec["manifest_sha256"]:
        raise ValueError(f"Fine-tuned checkpoint identity changed: {voice_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != 1
        or manifest.get("dtype") != "bfloat16"
        or manifest.get("language") != "es"
        or manifest.get("model_size") != "1.7B"
        or manifest.get("speaker") != spec["speaker"]
        or not spec.get("name")
    ):
        raise ValueError(f"Unsupported fine-tuned checkpoint: {voice_id}")
    required = {"model.safetensors", "config.json", "speech_tokenizer/model.safetensors", "tokenizer_config.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError(f"Incomplete fine-tuned checkpoint manifest: {voice_id}")
    for relative, expected in manifest["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (model / path).is_file():
            raise ValueError(f"Missing or invalid model artifact: {relative}")
        if verify_weights and file_sha256(model / path) != expected:
            raise ValueError(f"Fine-tuned model artifact changed: {relative}")
    return {**spec, "manifest": manifest}


def list_voices() -> list[dict]:
    """Expose only public preset metadata, never local checkpoint paths."""
    voices = []
    directory = registry_directory()
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            voice = read_voice(f"{VOICE_PREFIX}{path.stem}")
            voices.append(
                {
                    "voice_id": voice["voice_id"],
                    "name": voice["name"],
                    "gender": voice.get("gender", "male"),
                    "language": "es",
                }
            )
    return voices


@asynccontextmanager
async def loaded_backend_for_profile(engine: str, model_size: str, *, profile_id: str, db, profile=None):
    """Bind local profiles before loading, so they never download a preset base."""
    from ..backends.mlx_tts_lifecycle import loaded_tts_backend_for_request

    if engine == "qwen_custom_voice":
        if profile is None:
            from .profiles import get_profile

            profile = await get_profile(profile_id, db)
        if is_finetuned_profile(profile):
            from ..backends.qwen_finetuned_backend import get_local_backend

            voice = read_voice(profile.preset_voice_id)
            backend = get_local_backend()
            async with backend.voice_request(voice, model_size) as selected_backend:
                yield selected_backend
            return
    async with loaded_tts_backend_for_request(engine, model_size) as backend:
        yield backend
