"""
Chunked TTS generation utilities.

Splits long text into sentence-boundary chunks, generates audio per-chunk
via any TTSBackend, and concatenates with crossfade.  All logic is
engine-agnostic — it wraps the standard ``TTSBackend.generate()`` interface.

Short text (≤ max_chunk_chars) uses the single-shot fast path with zero
overhead.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import tempfile
from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np

from .. import config
from ..backends.mlx_tts_lifecycle import run_tts_operation_cancellation_safe
from .disk_reservations import DiskSpaceReservationError, reserve_disk_space

if TYPE_CHECKING:
    from ..services.exact_chunk_checkpoints import ExactChunkCheckpointSession

logger = logging.getLogger("voicebox.chunked-tts")

# Default chunk size in characters.  Can be overridden per-request via
# the ``max_chunk_chars`` field on GenerationRequest.
DEFAULT_MAX_CHUNK_CHARS = 800
MAX_RUNAWAY_RETRIES = 2
MIN_RUNAWAY_RETRY_CHARS = 100
MAX_GENERATED_AUDIO_DURATION_SECONDS = 24 * 60 * 60
MAX_GENERATED_AUDIO_SAMPLE_RATE = 192_000
GENERATED_AUDIO_MIN_FREE_BYTES = 1024**3
_SEED_MODULUS = 1 << 32
_FLOAT32_BYTES = np.dtype(np.float32).itemsize

# Common abbreviations that should NOT be treated as sentence endings.
# Lowercase for case-insensitive matching.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "ave",
        "blvd",
        "inc",
        "ltd",
        "corp",
        "dept",
        "est",
        "approx",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "u.s",
        "u.s.a",
        "u.k",
    }
)

# Paralinguistic tags used by Chatterbox Turbo.  The splitter must never
# cut inside one of these.
_PARA_TAG_RE = re.compile(r"\[[^\]]*\]")


class GeneratedAudioLimitError(ValueError):
    """Generated audio exceeds the bounded long-form output contract."""


class GeneratedAudioStorageError(RuntimeError):
    """Long-form generation cannot preserve the application's disk reserve."""


class GeneratedAudioEmptyError(RuntimeError):
    """TTS accepted text but returned no audio frames for it."""


def _raise_empty_generated_audio(text: str) -> None:
    compact_text = " ".join(text.split())
    preview = compact_text if len(compact_text) <= 120 else compact_text[:117] + "..."
    raise GeneratedAudioEmptyError(f"TTS returned no audio frames for text {preview!r}")


def is_disk_backed_audio(audio: object) -> bool:
    """Return whether *audio* owns Voicebox's anonymous long-form storage."""
    return isinstance(audio, np.memmap) and getattr(audio, "_voicebox_disk_backed_audio", False) is True


def release_disk_backed_audio(audio: object) -> None:
    """Deterministically close an anonymous long-form mapping, if any.

    The temporary file is already unlinked on POSIX and is deleted by closing
    its handle on platforms that cannot unlink an open file. Calling this more
    than once is harmless.
    """
    if not is_disk_backed_audio(audio):
        return
    temporary_file = getattr(audio, "_voicebox_temporary_file", None)
    with suppress(Exception):
        audio.flush()
    mapping = getattr(audio, "_mmap", None)
    if mapping is not None:
        with suppress(Exception):
            mapping.close()
    if temporary_file is not None:
        with suppress(Exception):
            temporary_file.close()
    with suppress(Exception):
        delattr(audio, "_voicebox_temporary_file")
    audio._voicebox_disk_backed_audio = False


def _validate_generated_audio_sample_rate(sample_rate: int) -> None:
    if type(sample_rate) is not int or sample_rate <= 0:
        raise GeneratedAudioLimitError("TTS returned an invalid sample rate")
    if sample_rate > MAX_GENERATED_AUDIO_SAMPLE_RATE:
        raise GeneratedAudioLimitError(f"TTS sample rate exceeds the {MAX_GENERATED_AUDIO_SAMPLE_RATE} Hz limit")


class _DiskBackedChunkAccumulator:
    """Append float32 chunks with exact legacy crossfades into anonymous disk.

    Only the current model result and at most one crossfade window remain in
    memory. ``finish`` transfers the temporary-file ownership to an
    ``np.memmap``; ``close`` cleans an unfinished accumulator on error or
    cancellation.
    """

    def __init__(self, sample_rate: int, crossfade_ms: int) -> None:
        _validate_generated_audio_sample_rate(sample_rate)
        if type(crossfade_ms) is not int or crossfade_ms < 0:
            raise GeneratedAudioLimitError("Crossfade duration must be a non-negative integer")
        self.sample_rate = sample_rate
        self.crossfade_samples = int(sample_rate * crossfade_ms / 1000)
        self.max_frames = sample_rate * MAX_GENERATED_AUDIO_DURATION_SECONDS
        cache_directory = config.get_cache_dir()
        temporary_file = None
        try:
            temporary_file = tempfile.TemporaryFile(  # noqa: SIM115 - ownership transfers to the returned memmap
                mode="w+b",
                buffering=0,
                prefix=".generation-audio-",
                dir=cache_directory,
            )
            file_stat = os.fstat(temporary_file.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise GeneratedAudioStorageError("Generation scratch storage is not a regular file")
            if os.name == "posix":
                os.fchmod(temporary_file.fileno(), 0o600)
        except GeneratedAudioStorageError:
            if temporary_file is not None:
                temporary_file.close()
            raise
        except OSError as exc:
            if temporary_file is not None:
                temporary_file.close()
            raise GeneratedAudioStorageError("Could not allocate private generation scratch storage") from exc
        self._file = temporary_file
        self._cache_directory = cache_directory
        self._frames = 0
        self._closed = False

    @property
    def frames(self) -> int:
        return self._frames

    def _check_growth(self, additional_frames: int) -> None:
        if additional_frames < 0 or self._frames + additional_frames > self.max_frames:
            raise GeneratedAudioLimitError(
                f"Generated audio exceeds the {MAX_GENERATED_AUDIO_DURATION_SECONDS // 3600}-hour duration limit"
            )

    def _write_array(self, audio: np.ndarray) -> None:
        if not audio.size:
            return
        contiguous = np.ascontiguousarray(audio, dtype=np.float32)
        view = memoryview(contiguous).cast("B")
        while view:
            try:
                written = os.write(self._file.fileno(), view)
            except OSError as exc:
                raise GeneratedAudioStorageError("Could not write generated audio scratch data") from exc
            if written <= 0:
                raise GeneratedAudioStorageError("Short write while storing generated audio")
            view = view[written:]

    def _read_tail(self) -> np.ndarray:
        tail_frames = min(self.crossfade_samples, self._frames)
        if tail_frames == 0:
            return np.empty(0, dtype=np.float32)
        byte_count = tail_frames * _FLOAT32_BYTES
        try:
            self._file.seek(-byte_count, os.SEEK_END)
            payload = self._file.read(byte_count)
            self._file.seek(0, os.SEEK_END)
        except OSError as exc:
            raise GeneratedAudioStorageError("Could not read generated audio crossfade data") from exc
        if len(payload) != byte_count:
            raise GeneratedAudioStorageError("Generated audio scratch data was truncated")
        return np.frombuffer(payload, dtype=np.float32).copy()

    def append(self, chunk: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Generated audio accumulator is closed")
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 1:
            raise GeneratedAudioLimitError("TTS returned audio with an invalid shape")
        if len(chunk) == 0:
            return

        overlap = min(self.crossfade_samples, self._frames, len(chunk))
        additional_frames = len(chunk) - overlap
        self._check_growth(additional_frames)
        try:
            reservation = reserve_disk_space(
                self._cache_directory,
                additional_frames * _FLOAT32_BYTES,
                min_free_bytes=GENERATED_AUDIO_MIN_FREE_BYTES,
            )
        except DiskSpaceReservationError as exc:
            raise GeneratedAudioStorageError(
                "Insufficient free space for generated audio while preserving the disk reserve"
            ) from exc
        with reservation:
            try:
                if overlap > 0:
                    result_tail = self._read_tail()[-overlap:]
                    fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                    blended = result_tail * fade_out + chunk[:overlap] * fade_in
                    self._file.seek(-overlap * _FLOAT32_BYTES, os.SEEK_END)
                    self._write_array(blended)
                self._file.seek(0, os.SEEK_END)
                self._write_array(chunk[overlap:])
            except GeneratedAudioStorageError:
                raise
            except OSError as exc:
                raise GeneratedAudioStorageError("Could not append generated audio scratch data") from exc
        self._frames += additional_frames

    def finish(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Generated audio accumulator is closed")
        if self._frames == 0:
            self.close()
            return np.empty(0, dtype=np.float32)
        temporary_file = self._file
        try:
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            audio = np.memmap(
                temporary_file,
                dtype=np.float32,
                mode="r+",
                shape=(self._frames,),
            )
            audio._voicebox_disk_backed_audio = True
            audio._voicebox_temporary_file = temporary_file
        except BaseException:
            self.close()
            raise
        self._file = None
        self._closed = True
        return audio

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        temporary_file = self._file
        self._file = None
        if temporary_file is not None:
            with suppress(Exception):
                temporary_file.close()


def _offset_seed(seed: int | None, offset: int) -> int | None:
    """Derive a deterministic child seed without leaving NumPy's uint32 domain."""
    return (seed + offset) % _SEED_MODULUS if seed is not None else None


def split_text_into_chunks(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> list[str]:
    """Split *text* at natural boundaries into chunks of at most *max_chars*.

    Priority: sentence-end (``.!?`` not preceded by an abbreviation and not
    inside brackets) → clause boundary (``;:,—``) → whitespace → hard cut.

    Paralinguistic tags like ``[laugh]`` are treated as atomic and will not
    be split across chunks.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        segment = remaining[:max_chars]

        # Try to split at the last real sentence ending
        split_pos = _find_last_sentence_end(segment)
        if split_pos == -1:
            split_pos = _find_last_clause_boundary(segment)
        if split_pos == -1:
            split_pos = segment.rfind(" ")
        if split_pos == -1:
            # Absolute fallback: hard cut but avoid splitting inside a tag
            split_pos = _safe_hard_cut(segment, max_chars)

        chunk = remaining[: split_pos + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos + 1 :]

    return chunks


def _find_last_sentence_end(text: str) -> int:
    """Return the index of the last sentence-ending punctuation in *text*.

    Skips periods that follow common abbreviations (``Dr.``, ``Mr.``, etc.)
    and periods inside bracket tags (``[laugh]``). Also handles CJK full
    stops, exclamation marks, and question marks.
    """
    best = -1
    # ASCII sentence ends
    for m in re.finditer(r"[.!?](?:\s|$)", text):
        pos = m.start()
        char = text[pos]
        # Skip periods after abbreviations
        if char == ".":
            # Walk backwards to find the preceding word
            word_start = pos - 1
            while word_start >= 0 and text[word_start].isalpha():
                word_start -= 1
            word = text[word_start + 1 : pos].lower()
            if word in _ABBREVIATIONS:
                continue
            # Skip decimal numbers (digit immediately before the period)
            if word_start >= 0 and text[word_start].isdigit():
                continue
        # Skip if we're inside a bracket tag
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    # CJK sentence-ending punctuation
    for m in re.finditer(r"[\u3002\uff01\uff1f]", text):
        if m.start() > best:
            best = m.start()
    return best


def _find_last_clause_boundary(text: str) -> int:
    """Return the index of the last clause-boundary punctuation."""
    best = -1
    for m in re.finditer(r"[;:,\u2014](?:\s|$)", text):
        pos = m.start()
        # Skip if inside a bracket tag
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    return best


def _inside_bracket_tag(text: str, pos: int) -> bool:
    """Return True if *pos* falls inside a ``[...]`` tag."""
    return any(m.start() < pos < m.end() for m in _PARA_TAG_RE.finditer(text))


def _safe_hard_cut(segment: str, max_chars: int) -> int:
    """Find a hard-cut position that doesn't split a ``[tag]``."""
    cut = max_chars - 1
    # Check if the cut falls inside a bracket tag; if so, move before it
    for m in _PARA_TAG_RE.finditer(segment):
        if m.start() < cut < m.end():
            return m.start() - 1 if m.start() > 0 else cut
    return cut


def concatenate_audio_chunks(
    chunks: list[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 50,
) -> np.ndarray:
    """Concatenate audio arrays with a short crossfade to eliminate clicks.

    Each chunk is expected to be a 1-D float32 ndarray at *sample_rate* Hz.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result_samples = len(chunks[0])
    overlaps: list[int | None] = []
    for chunk in chunks[1:]:
        if len(chunk) == 0:
            overlaps.append(None)
            continue
        overlap = min(crossfade_samples, result_samples, len(chunk))
        overlaps.append(overlap)
        result_samples += len(chunk) - overlap

    # Allocate the final array once. Repeated np.concatenate copied the entire
    # accumulated book prefix for every logical chunk (quadratic work and a
    # near-final-size transient allocation on each iteration).
    result = np.empty(result_samples, dtype=np.float32)
    first = np.asarray(chunks[0])
    cursor = len(first)
    result[:cursor] = first

    for chunk, overlap in zip(chunks[1:], overlaps, strict=True):
        if overlap is None:
            continue
        if overlap > 0:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            result[cursor - overlap : cursor] = result[cursor - overlap : cursor] * fade_out + chunk[:overlap] * fade_in
        tail = chunk[overlap:]
        result[cursor : cursor + len(tail)] = tail
        cursor += len(tail)

    return result


async def generate_text_batch(
    backend,
    texts: list[str],
    voice_prompt: dict,
    language: str = "en",
    seeds: list[int | None] | None = None,
    instruct: str | None = None,
    crossfade_ms: int = 50,
    trim_fn=None,
    runaway_detector=None,
) -> list[tuple[np.ndarray, int]]:
    """Generate exactly two independent texts in one model-level batch.

    The returned list is positional: item ``i`` always belongs to ``texts[i]``.
    Backends without a batch implementation retain the legacy serial behavior;
    real inference failures are never hidden by a serial retry.  Each logical
    unit keeps its own seed so a saved audiobook manifest remains reproducible.
    """
    if len(texts) != 2:
        raise ValueError("model-level batching requires exactly two texts")
    if seeds is None:
        seeds = [None, None]
    if len(seeds) != len(texts):
        raise ValueError("model-level batching requires one seed per text")

    async def generate_serial() -> list[tuple[np.ndarray, int]]:
        serial_results = []
        for text, seed in zip(texts, seeds, strict=True):
            audio, sample_rate = await run_tts_operation_cancellation_safe(
                backend,
                backend.generate(
                    text,
                    voice_prompt,
                    language,
                    seed,
                    instruct,
                ),
            )
            serial_results.append((np.asarray(audio, dtype=np.float32), sample_rate))
        return serial_results

    generate_batch = getattr(backend, "generate_batch", None)
    if not callable(generate_batch):
        results = await generate_serial()
    else:
        try:
            results = await run_tts_operation_cancellation_safe(
                backend,
                generate_batch(
                    texts,
                    voice_prompt,
                    language=language,
                    seeds=seeds,
                    instruct=instruct,
                ),
            )
        except NotImplementedError:
            results = await generate_serial()

    if len(results) != len(texts):
        noun = "result" if len(results) == 1 else "results"
        raise RuntimeError(f"TTS batch returned {len(results)} {noun} for {len(texts)} texts")

    processed: list[tuple[np.ndarray, int]] = []
    sample_rate: int | None = None
    pending_disk_audio: np.ndarray | None = None
    try:
        for text, seed, result in zip(texts, seeds, results, strict=True):
            try:
                audio, item_sample_rate = result
            except (TypeError, ValueError) as exc:
                raise RuntimeError("TTS batch returned an invalid audio result") from exc
            audio = np.asarray(audio, dtype=np.float32)
            _validate_generated_audio_sample_rate(item_sample_rate)
            if audio.ndim != 1:
                raise GeneratedAudioLimitError("TTS returned audio with an invalid shape")
            if len(audio) == 0:
                _raise_empty_generated_audio(text)
            if len(audio) > item_sample_rate * MAX_GENERATED_AUDIO_DURATION_SECONDS:
                raise GeneratedAudioLimitError(
                    f"Generated audio exceeds the {MAX_GENERATED_AUDIO_DURATION_SECONDS // 3600}-hour duration limit"
                )
            if runaway_detector is not None and runaway_detector(audio, item_sample_rate):
                retry_max_chars = max(MIN_RUNAWAY_RETRY_CHARS, len(text) // 2)
                logger.warning(
                    "Detected unstable batched TTS output for %d chars; retrying serially",
                    len(text),
                )
                audio, item_sample_rate = await generate_chunked(
                    backend,
                    text,
                    voice_prompt,
                    language=language,
                    seed=seed,
                    instruct=instruct,
                    max_chunk_chars=retry_max_chars,
                    crossfade_ms=crossfade_ms,
                    trim_fn=trim_fn,
                    runaway_detector=runaway_detector,
                )
            elif trim_fn is not None:
                audio = trim_fn(audio, item_sample_rate)
            if is_disk_backed_audio(audio):
                pending_disk_audio = audio
            else:
                audio = np.asarray(audio, dtype=np.float32)
            _validate_generated_audio_sample_rate(item_sample_rate)
            if audio.ndim != 1:
                raise GeneratedAudioLimitError("TTS returned audio with an invalid shape")
            if len(audio) == 0:
                _raise_empty_generated_audio(text)
            if len(audio) > item_sample_rate * MAX_GENERATED_AUDIO_DURATION_SECONDS:
                raise GeneratedAudioLimitError(
                    f"Generated audio exceeds the {MAX_GENERATED_AUDIO_DURATION_SECONDS // 3600}-hour duration limit"
                )
            if sample_rate is None:
                sample_rate = item_sample_rate
            elif item_sample_rate != sample_rate:
                raise RuntimeError(
                    f"TTS batch returned inconsistent sample rates: {sample_rate} and {item_sample_rate}"
                )
            processed.append((audio, item_sample_rate))
            pending_disk_audio = None
    except BaseException:
        release_disk_backed_audio(pending_disk_audio)
        for audio, _item_sample_rate in processed:
            release_disk_backed_audio(audio)
        raise

    return processed


async def generate_chunked(
    backend,
    text: str,
    voice_prompt: dict,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    crossfade_ms: int = 50,
    trim_fn=None,
    runaway_detector=None,
    checkpoint_session: ExactChunkCheckpointSession | None = None,
) -> tuple[np.ndarray, int]:
    """Generate audio with automatic chunking for long text.

    For text shorter than *max_chunk_chars* this is a thin wrapper around
    ``backend.generate()`` with zero overhead.

    For longer text the input is split at natural sentence boundaries,
    each chunk is generated independently, optionally trimmed (useful for
    Chatterbox engines that hallucinate trailing noise), and the results
    are concatenated with a crossfade (or hard cut if *crossfade_ms* is 0).

    Parameters
    ----------
    backend : TTSBackend
        Any backend implementing the ``generate()`` protocol.
    text : str
        Input text (may be arbitrarily long).
    voice_prompt, language, seed, instruct
        Forwarded to ``backend.generate()`` verbatim.
    max_chunk_chars : int
        Maximum characters per chunk (default 800).
    crossfade_ms : int
        Crossfade duration in milliseconds between chunks.  0 for a hard
        cut with no overlap (default 50).
    trim_fn : callable | None
        Optional ``(audio, sample_rate) -> audio`` post-processing
        function applied to each chunk before concatenation (e.g.
        ``trim_tts_output`` for Chatterbox engines).
    runaway_detector : callable | None
        Optional ``(audio, sample_rate) -> bool`` detector. When it flags
        unstable output, the affected text is split in half and retried.
    checkpoint_session : ExactChunkCheckpointSession | None
        Exact-singleton-only durable storage. Each post-trim logical chunk is
        fsynced before the next model call and reused after interruption.

    Returns
    -------
    (audio, sample_rate) : Tuple[np.ndarray, int]
        Multi-chunk or runaway-retried output is an anonymous disk-backed
        ``np.memmap``. Its consumer must call
        :func:`release_disk_backed_audio` in ``finally``; a stable single-shot
        output retains the backend's ordinary ndarray fast path.
    """

    async def generate_one(
        chunk_text: str,
        chunk_seed: int | None,
        retry_depth: int = 0,
    ) -> tuple[np.ndarray, int]:
        chunk_audio, chunk_sr = await run_tts_operation_cancellation_safe(
            backend,
            backend.generate(
                chunk_text,
                voice_prompt,
                language,
                chunk_seed,
                instruct,
            ),
        )
        chunk_audio = np.asarray(chunk_audio, dtype=np.float32)
        _validate_generated_audio_sample_rate(chunk_sr)
        if chunk_audio.ndim != 1:
            raise GeneratedAudioLimitError("TTS returned audio with an invalid shape")
        if len(chunk_audio) == 0:
            _raise_empty_generated_audio(chunk_text)

        if runaway_detector is not None and runaway_detector(chunk_audio, chunk_sr):
            if retry_depth >= MAX_RUNAWAY_RETRIES or len(chunk_text) <= MIN_RUNAWAY_RETRY_CHARS:
                raise RuntimeError("TTS output remained unstable after retrying smaller text chunks")

            retry_max_chars = max(MIN_RUNAWAY_RETRY_CHARS, len(chunk_text) // 2)
            retry_chunks = split_text_into_chunks(chunk_text, retry_max_chars)
            if len(retry_chunks) <= 1:
                raise RuntimeError("Unable to split unstable TTS output for retry")

            logger.warning(
                "Detected unstable TTS output for %d chars; retrying as %d smaller chunks",
                len(chunk_text),
                len(retry_chunks),
            )
            retry_accumulator: _DiskBackedChunkAccumulator | None = None
            retry_sample_rate: int | None = None
            try:
                for i, retry_text in enumerate(retry_chunks):
                    retry_seed = _offset_seed(
                        chunk_seed,
                        ((retry_depth + 1) * 1000) + i,
                    )
                    audio, item_sample_rate = await generate_one(
                        retry_text,
                        retry_seed,
                        retry_depth + 1,
                    )
                    try:
                        _validate_generated_audio_sample_rate(item_sample_rate)
                        if retry_sample_rate is None:
                            retry_sample_rate = item_sample_rate
                            retry_accumulator = _DiskBackedChunkAccumulator(
                                retry_sample_rate,
                                crossfade_ms,
                            )
                        elif item_sample_rate != retry_sample_rate:
                            raise RuntimeError(
                                f"TTS returned inconsistent sample rates: {retry_sample_rate} and {item_sample_rate}"
                            )
                        assert retry_accumulator is not None
                        retry_accumulator.append(audio)
                    finally:
                        release_disk_backed_audio(audio)

                assert retry_accumulator is not None
                assert retry_sample_rate is not None
                return retry_accumulator.finish(), retry_sample_rate
            finally:
                if retry_accumulator is not None:
                    retry_accumulator.close()

        if trim_fn is not None:
            chunk_audio = trim_fn(chunk_audio, chunk_sr)
        chunk_audio = np.asarray(chunk_audio, dtype=np.float32)
        if chunk_audio.ndim != 1:
            raise GeneratedAudioLimitError("TTS returned audio with an invalid shape")
        if len(chunk_audio) == 0:
            _raise_empty_generated_audio(chunk_text)
        return chunk_audio, chunk_sr

    async def load_or_generate_logical_chunk(
        logical_index: int,
        chunk_text: str,
        chunk_seed: int | None,
    ) -> tuple[np.ndarray, int]:
        if checkpoint_session is not None:
            if chunk_seed is None:
                raise ValueError("Exact chunk checkpoints require an explicit uint32 seed")
            checkpoint = checkpoint_session.load(
                logical_index=logical_index,
                text=chunk_text,
                seed=chunk_seed,
            )
            if checkpoint is not None:
                logger.info("Reusing durable exact checkpoint for logical chunk %d", logical_index + 1)
                return checkpoint.audio, checkpoint.sample_rate

        chunk_audio, chunk_sr = await generate_one(chunk_text, chunk_seed)
        if checkpoint_session is not None:
            assert chunk_seed is not None
            checkpoint_session.save(
                logical_index=logical_index,
                text=chunk_text,
                seed=chunk_seed,
                audio=chunk_audio,
                sample_rate=chunk_sr,
            )
        return chunk_audio, chunk_sr

    chunks = split_text_into_chunks(text, max_chunk_chars)

    if len(chunks) <= 1:
        # Short text — single-shot fast path
        audio, sample_rate = await load_or_generate_logical_chunk(0, text, seed)
        _validate_generated_audio_sample_rate(sample_rate)
        if audio.ndim != 1:
            raise GeneratedAudioLimitError("TTS returned audio with an invalid shape")
        if len(audio) > sample_rate * MAX_GENERATED_AUDIO_DURATION_SECONDS:
            raise GeneratedAudioLimitError(
                f"Generated audio exceeds the {MAX_GENERATED_AUDIO_DURATION_SECONDS // 3600}-hour duration limit"
            )
        return audio, sample_rate

    # Long text — chunked generation
    logger.info(
        "Splitting %d chars into %d chunks (max %d chars each)",
        len(text),
        len(chunks),
        max_chunk_chars,
    )
    accumulator: _DiskBackedChunkAccumulator | None = None
    sample_rate: int | None = None
    try:
        for i, chunk_text in enumerate(chunks):
            logger.info(
                "Generating chunk %d/%d (%d chars)",
                i + 1,
                len(chunks),
                len(chunk_text),
            )
            # Vary the seed per chunk to avoid correlated RNG artefacts,
            # but keep it deterministic so the same (text, seed) pair
            # always produces the same output.
            chunk_seed = _offset_seed(seed, i)

            chunk_audio, chunk_sr = await load_or_generate_logical_chunk(
                i,
                chunk_text,
                chunk_seed,
            )
            _validate_generated_audio_sample_rate(chunk_sr)
            if sample_rate is None:
                sample_rate = chunk_sr
                accumulator = _DiskBackedChunkAccumulator(sample_rate, crossfade_ms)
            elif chunk_sr != sample_rate:
                raise RuntimeError(f"TTS returned inconsistent sample rates: {sample_rate} and {chunk_sr}")
            assert accumulator is not None
            try:
                accumulator.append(chunk_audio)
            finally:
                release_disk_backed_audio(chunk_audio)

        assert accumulator is not None
        assert sample_rate is not None
        audio = accumulator.finish()
        return audio, sample_rate
    finally:
        if accumulator is not None:
            accumulator.close()
