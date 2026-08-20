"""Bounded metadata probing for portable generation/story audio."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

PORTABLE_AUDIO_MAX_CHANNELS = 8
PORTABLE_AUDIO_MAX_SAMPLE_RATE = 192_000
PORTABLE_AUDIO_MAX_DURATION_SECONDS = 24 * 60 * 60


def probe_audio_metadata(path: str | Path) -> tuple[float, int, int]:
    """Return duration, channel count, and sample rate without decoding PCM."""
    try:
        audio_info = sf.info(str(path))
        return (
            audio_info.frames / audio_info.samplerate if audio_info.samplerate else 0.0,
            int(audio_info.channels),
            int(audio_info.samplerate),
        )
    except (RuntimeError, TypeError, ValueError):
        # libsndfile does not understand every accepted container (notably
        # some AAC/M4A/WebM files). audioread only probes headers here; no PCM
        # frames are retained.
        import audioread

        with audioread.audio_open(str(path)) as decoder:
            return (
                float(decoder.duration),
                int(decoder.channels),
                int(decoder.samplerate),
            )
