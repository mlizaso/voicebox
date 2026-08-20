"""Exact and linear-memory chunk concatenation."""

from __future__ import annotations

import numpy as np
import pytest

from backend.utils.chunked_tts import concatenate_audio_chunks


def _legacy_concatenate(chunks, sample_rate, crossfade_ms):
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result = np.array(chunks[0], dtype=np.float32, copy=True)
    for chunk in chunks[1:]:
        if len(chunk) == 0:
            continue
        overlap = min(crossfade_samples, len(result), len(chunk))
        if overlap > 0:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            result[-overlap:] = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
            result = np.concatenate([result, chunk[overlap:]])
        else:
            result = np.concatenate([result, chunk])
    return result


@pytest.mark.parametrize("crossfade_ms", [0, 1, 50, 500])
def test_preallocated_concatenation_is_byte_identical_to_legacy(crossfade_ms):
    rng = np.random.default_rng(20260815)
    chunks = [rng.standard_normal(length).astype(np.float32) for length in (17, 0, 1001, 3, 2400, 1, 0, 997)]

    expected = _legacy_concatenate(chunks, 24_000, crossfade_ms)
    actual = concatenate_audio_chunks(chunks, 24_000, crossfade_ms)

    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
