"""Cloned voice synthesis must never fall back to an unrelated default voice."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from backend.backends.chatterbox_backend import ChatterboxTTSBackend
from backend.backends.chatterbox_turbo_backend import ChatterboxTurboTTSBackend


@pytest.mark.parametrize("backend_class", [ChatterboxTTSBackend, ChatterboxTurboTTSBackend])
@pytest.mark.parametrize("reference", ["missing", "directory", "absent"])
def test_invalid_reference_fails_before_loading_or_generating(tmp_path, backend_class, reference):
    backend = backend_class()
    backend.load_model = AsyncMock()
    backend.model = Mock()
    backend.model.generate.side_effect = AssertionError("Must not generate a default voice")
    prompt = (
        {}
        if reference == "absent"
        else {"ref_audio": str(tmp_path if reference == "directory" else tmp_path / "missing.wav")}
    )

    with pytest.raises(FileNotFoundError, match="Reference audio"):
        asyncio.run(backend.generate("Hello", prompt))

    backend.load_model.assert_not_called()
    backend.model.generate.assert_not_called()
