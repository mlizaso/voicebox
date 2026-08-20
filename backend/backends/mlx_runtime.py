"""Lightweight identity for Voicebox's guarded MLX Qwen TTS runtime."""

import ast
import sys
from collections.abc import Mapping
from functools import lru_cache
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from types import MappingProxyType

MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS: Mapping[str, str] = MappingProxyType(
    {
        "mlx-audio": "0.4.1",
        "mlx": "0.32.0",
        "mlx-lm": "0.31.1",
        # mlx-metal supplies the actual Darwin runtime imported as ``mlx``.
        "mlx-metal": "0.32.0",
        # Exact audiobook resume also covers reference preprocessing/decoding
        # and Qwen text tokenization.  Drifting any of these can change the
        # conditioning tensors even when the frozen WAVs and transcripts match.
        "numpy": "1.26.4",
        "transformers": "4.57.3",
        "tokenizers": "0.22.2",
        "miniaudio": "1.71",
        "librosa": "0.11.0",
        "soundfile": "0.14.0",
        "soxr": "1.1.0",
        # Exact API generations may apply the profile's effects chain before
        # exposing the default audio artifact.
        "pedalboard": "0.9.24",
    }
)
MLX_AUDIO_QWEN_DTYPE_BACKPORT_VERSION = MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS["mlx-audio"]
# Covers the BF16 dtype correction, reusable ICL conditioning, batch size two,
# independent per-row sampling keys, immutable reference snapshots, and the
# canonical ordered 24-kHz multi-sample prompt path. Any numerical change must
# bump it.
MLX_QWEN_DTYPE_PATCH_REVISION = "bf16-b2-icl-v3"
MLX_QWEN_TTS_LOCAL_NUMERICAL_SOURCE_PATHS = (
    # Request validation and exact-route mapping determine the frozen inputs
    # that reach the numerical pipeline. Model routes own the guarded
    # load/unload/cache-deletion lifecycle.
    "backend/models.py",
    "backend/routes/generations.py",
    "backend/routes/llm.py",
    "backend/routes/models.py",
    "backend/routes/tasks.py",
    "backend/routes/transcription.py",
    # Backend selection/loading and every local waveform/conditioning owner on
    # the exact Qwen path. The lifecycle guard and serial queue are included
    # because cancellation must retain ownership until executor inference ends.
    # Persistence and byte-serving modules remain excluded because they do not
    # transform or concurrently access the shared model.
    "backend/backends/__init__.py",
    "backend/backends/base.py",
    "backend/backends/mlx_backend.py",
    "backend/backends/mlx_qwen_optimizations.py",
    "backend/backends/mlx_tts_lifecycle.py",
    "backend/backends/qwen_llm_backend.py",
    "backend/services/exact_chunk_checkpoints.py",
    "backend/services/effects_processing.py",
    "backend/services/generation.py",
    "backend/services/profiles.py",
    "backend/services/task_queue.py",
    "backend/services/tts.py",
    "backend/utils/audio.py",
    "backend/utils/cache.py",
    "backend/utils/chunked_tts.py",
    "backend/utils/disk_reservations.py",
    "backend/utils/effects.py",
)
MLX_QWEN_TTS_NUMERICAL_SOURCE_FINGERPRINTS: Mapping[str, str] = MappingProxyType(
    {
        # AST fingerprints ignore comments, whitespace, and source locations,
        # while changing whenever executable source structure changes.  CI
        # verifies these embedded values before PyInstaller removes sources.
        "voicebox-mlx": "a2e85c2f49f79794eb2ed2c976f222a6a29da4f670cc8567ccb46a7fcc26e5d9",
        "mlx-audio-qwen3-tts": "ea2aaa5de132f381a1a817c8d4bd01c7fb00c431f1323606ef1f9b4394700a7e",
    }
)
MLX_QWEN_TTS_PINNED_MODELS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "1.7B": (
            "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            "a6eb4f68e4b056f1215157bb696209bc82a6db48",
        ),
        "0.6B": (
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "1eccf1cb2519b5a4e8a95b5f0544f3303568164f",
        ),
    }
)


def _python_ast_fingerprint(sources: Mapping[str, Path]) -> str:
    digest = sha256()
    for label, source_path in sorted(sources.items()):
        source = source_path.read_text(encoding="utf-8")
        canonical_ast = ast.dump(
            ast.parse(source, filename=str(source_path)),
            annotate_fields=True,
            include_attributes=False,
        )
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_ast.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def get_current_mlx_qwen_tts_source_fingerprints() -> Mapping[str, str] | None:
    """Fingerprint executable numerical sources without importing MLX/Metal.

    PyInstaller does not retain Python sources in a normal filesystem tree.
    Frozen builds therefore use the embedded, CI-verified fingerprints that
    were part of the executable build. Source installations recompute both
    sides so a locally edited backport or site-package cannot advertise the
    reviewed exact-generation identity.
    """
    if getattr(sys, "frozen", False):
        return MLX_QWEN_TTS_NUMERICAL_SOURCE_FINGERPRINTS

    repository_root = Path(__file__).resolve().parents[2]
    local_sources = {
        relative_path: repository_root / relative_path for relative_path in MLX_QWEN_TTS_LOCAL_NUMERICAL_SOURCE_PATHS
    }
    try:
        qwen_root = Path(metadata.distribution("mlx-audio").locate_file("mlx_audio/tts/models/qwen3_tts"))
        qwen_sources = {source.relative_to(qwen_root).as_posix(): source for source in qwen_root.rglob("*.py")}
        if not qwen_sources:
            return None
        return MappingProxyType(
            {
                "voicebox-mlx": _python_ast_fingerprint(local_sources),
                "mlx-audio-qwen3-tts": _python_ast_fingerprint(qwen_sources),
            }
        )
    except (OSError, SyntaxError, ValueError, metadata.PackageNotFoundError):
        return None


def build_mlx_qwen_tts_implementation_revision(
    *,
    runtime_packages: Mapping[str, str] = MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS,
    patch_revision: str = MLX_QWEN_DTYPE_PATCH_REVISION,
    model_revisions: Mapping[str, tuple[str, str]] = MLX_QWEN_TTS_PINNED_MODELS,
    source_fingerprints: Mapping[str, str] = MLX_QWEN_TTS_NUMERICAL_SOURCE_FINGERPRINTS,
) -> str:
    """Build a compact identity covering code, patch, repositories, and weights."""
    mlx_audio_version = runtime_packages.get("mlx-audio")
    if not mlx_audio_version:
        raise ValueError("MLX Qwen TTS runtime pins must include mlx-audio")
    canonical_specs = "\n".join(
        (
            f"patch={patch_revision}",
            *(f"runtime:{package}={version}" for package, version in sorted(runtime_packages.items())),
            *(
                f"source:{source_name}={fingerprint}"
                for source_name, fingerprint in sorted(source_fingerprints.items())
            ),
            *(f"{size}={repo}@{revision}" for size, (repo, revision) in sorted(model_revisions.items())),
        )
    )
    model_fingerprint = sha256(canonical_specs.encode("utf-8")).hexdigest()
    identity = f"qwen3-mlx-audio-{mlx_audio_version}-{patch_revision}-runtime-sha256-{model_fingerprint}"
    if len(identity) > 128:
        raise ValueError("MLX Qwen TTS implementation identity exceeds the API's 128-character limit")
    return identity


MLX_QWEN_TTS_IMPLEMENTATION_REVISION = build_mlx_qwen_tts_implementation_revision()


def get_mlx_qwen_tts_model_spec(model_size: str) -> tuple[str, str]:
    """Return the immutable Hugging Face repository and commit for a model size."""
    try:
        return MLX_QWEN_TTS_PINNED_MODELS[model_size]
    except KeyError as exc:
        raise ValueError(f"Unsupported pinned MLX Qwen TTS model size: {model_size}") from exc


def get_installed_mlx_audio_version() -> str | None:
    """Return the installed distribution version without importing mlx-audio."""
    try:
        return metadata.version("mlx-audio")
    except metadata.PackageNotFoundError:
        return None


def get_mlx_qwen_tts_implementation_revision() -> str | None:
    """Return the exact guarded numerical revision, or fail closed."""
    for package, expected_version in MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS.items():
        try:
            installed_version = metadata.version(package)
        except metadata.PackageNotFoundError:
            return None
        if installed_version != expected_version:
            return None
    current_sources = get_current_mlx_qwen_tts_source_fingerprints()
    if current_sources != MLX_QWEN_TTS_NUMERICAL_SOURCE_FINGERPRINTS:
        return None
    return MLX_QWEN_TTS_IMPLEMENTATION_REVISION
