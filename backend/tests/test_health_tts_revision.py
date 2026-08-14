"""Health-contract coverage for exact-resume TTS runtime identity."""

import asyncio
from types import SimpleNamespace

import huggingface_hub

from backend.backends.mlx_runtime import MLX_QWEN_TTS_IMPLEMENTATION_REVISION
from backend.routes import health as health_route


def test_health_exposes_mlx_qwen_runtime_revision(monkeypatch):
    backend = SimpleNamespace(is_loaded=lambda: False)
    monkeypatch.setattr(health_route.tts, "get_tts_model", lambda: backend)
    monkeypatch.setattr(health_route, "get_backend_type", lambda: "mlx")
    monkeypatch.setattr(health_route, "is_amd_gpu_windows", lambda: False)
    monkeypatch.setattr(health_route.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(health_route.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: SimpleNamespace(repos=()))

    from backend import backends

    monkeypatch.setattr(
        backends,
        "get_tts_implementation_revision",
        lambda: MLX_QWEN_TTS_IMPLEMENTATION_REVISION,
    )

    response = asyncio.run(health_route.health())

    assert response.status == "healthy"
    assert response.tts_implementation_revision == MLX_QWEN_TTS_IMPLEMENTATION_REVISION
