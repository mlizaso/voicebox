"""Bounded, disk-backed audio-effects processing.

The effects API accepts imported audio as well as short TTS generations.  A
legal imported generation can be many hours long, so decoding it into one
NumPy array (and then making another processed array and an in-memory WAV)
is not safe.  This module keeps decode, resample, DSP, and encode work
incremental while preserving Pedalboard state across blocks.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import stat
import tempfile
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import audioread
import numpy as np
import soundfile as sf
import soxr

from .. import config
from ..utils.disk_reservations import (
    DiskSpaceReservation,
    DiskSpaceReservationError,
    reserve_disk_space,
)

EFFECTS_OUTPUT_SAMPLE_RATE = 24_000
EFFECTS_DECODE_BLOCK_FRAMES = 65_536
# Large blocks reduce Python/Pedalboard overhead and, importantly, give the
# phase-vocoder used by PitchShift generous continuous context without making
# memory usage depend on the duration of the source.
EFFECTS_PROCESS_BLOCK_FRAMES = 1_048_576
# PitchShift has roughly one second of algorithmic look-ahead at 24 kHz. Its
# stateful API returns fewer samples on the first call and produces a damaged
# transition when a new input array is supplied. Give each bounded one-shot
# window ample context, then join only a short interior overlap.
EFFECTS_PITCH_GUARD_SECONDS = 4.096
EFFECTS_PITCH_OVERLAP_SECONDS = 4096 / 24_000
EFFECTS_MAX_DURATION_SECONDS = 24 * 60 * 60
EFFECTS_MAX_CHANNELS = 8
EFFECTS_MAX_SAMPLE_RATE = 192_000
EFFECTS_MAX_SOURCE_BYTES = 8 * 1024**3
EFFECTS_MAX_COMPRESSED_SNAPSHOT_BYTES = 512 * 1024**2
EFFECTS_MAX_DECODED_SAMPLE_VALUES = EFFECTS_MAX_DURATION_SECONDS * 48_000 * 2
EFFECTS_MIN_FREE_BYTES = 1024**3
EFFECTS_EXPORT_ROOT_NAME = "effects-processing-v1"
EFFECTS_MAX_STALE_ENTRIES = 1000
EFFECTS_MAX_ACTIVE_DIRECTORIES = 16

logger = logging.getLogger(__name__)


class EffectsProcessingError(ValueError):
    """The selected audio cannot be processed safely."""


class EffectsProcessingLimitError(EffectsProcessingError):
    """The selected audio exceeds a bounded processing contract."""


class EffectsProcessingStorageError(RuntimeError):
    """Processing cannot preserve the application's disk-space reserve."""


class EffectsProcessingBusyError(RuntimeError):
    """Another effects render already owns the bounded worker slot."""


class _EffectsProcessingCancelledError(RuntimeError):
    """Internal cooperative cancellation signal raised in a worker thread."""


@dataclass(frozen=True)
class EffectsPreview:
    """A private preview WAV whose response owner must clean it."""

    path: Path
    temporary_directory: Path

    def cleanup(self) -> None:
        _cleanup_effects_directory(self.temporary_directory)


@dataclass(frozen=True)
class GeneratedAudioFile:
    """A private generated WAV whose HTTP response owns its cleanup."""

    path: Path
    temporary_directory: Path

    def cleanup(self) -> None:
        _cleanup_effects_directory(self.temporary_directory)


@dataclass(frozen=True)
class _ProbedSource:
    frames: int
    sample_rate: int
    channels: int
    output_frames: int
    decoder: str
    snapshot_path: Path | None


_effects_processing_lock = threading.Lock()
_effects_directory_state_lock = threading.Lock()
_active_effects_directories: set[Path] = set()


def _effects_root() -> Path:
    """Return the private, managed effects scratch directory."""
    root = config.get_cache_dir() / EFFECTS_EXPORT_ROOT_NAME
    with suppress(FileExistsError):
        root.mkdir(mode=0o700)
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise EffectsProcessingStorageError("Effects scratch storage is unavailable") from exc
    is_junction = getattr(root, "is_junction", None)
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
        or (is_junction is not None and is_junction())
    ):
        raise EffectsProcessingStorageError("Effects scratch storage is not a real directory")
    if os.name == "posix":
        os.chmod(root, 0o700, follow_symlinks=False)
    return root


def _remove_effects_directory(directory: Path) -> bool:
    """Remove one owned scratch entry without following a replacement link."""
    try:
        entry_stat = directory.lstat()
    except FileNotFoundError:
        return True
    is_junction = getattr(directory, "is_junction", None)
    if is_junction is not None and is_junction():
        directory.rmdir()
        return True
    if stat.S_ISDIR(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode):
        shutil.rmtree(directory)
        return True
    if stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode):
        directory.unlink(missing_ok=True)
        return True
    return False


def _cleanup_stale_effects_directories_locked(root: Path) -> tuple[int, int, bool]:
    removed = 0
    refused = 0
    scanned = 0
    truncated = False
    try:
        for entry in root.iterdir():
            scanned += 1
            if scanned > EFFECTS_MAX_STALE_ENTRIES:
                truncated = True
                break
            if not entry.name.startswith("job-") or entry in _active_effects_directories:
                continue
            try:
                if _remove_effects_directory(entry):
                    removed += 1
                else:
                    refused += 1
            except OSError:
                refused += 1
    except OSError as exc:
        raise EffectsProcessingStorageError("Could not inspect effects scratch storage") from exc
    return removed, refused, truncated


def cleanup_abandoned_effects_processing() -> tuple[int, int, bool]:
    """Reclaim crash-abandoned preview and decoder scratch."""
    with _effects_directory_state_lock:
        return _cleanup_stale_effects_directories_locked(_effects_root())


def _allocate_effects_directory() -> Path:
    with _effects_directory_state_lock:
        root = _effects_root()
        removed, refused, truncated = _cleanup_stale_effects_directories_locked(root)
        if removed:
            logger.info("Removed %d abandoned effects processing director%s", removed, "y" if removed == 1 else "ies")
        if refused or truncated:
            logger.warning(
                "Retained unsafe or excess effects scratch (refused=%d, truncated=%s)",
                refused,
                truncated,
            )
        if len(_active_effects_directories) >= EFFECTS_MAX_ACTIVE_DIRECTORIES:
            raise EffectsProcessingBusyError(
                f"Too many effects responses are active (max {EFFECTS_MAX_ACTIVE_DIRECTORIES}); retry later"
            )
        directory = Path(tempfile.mkdtemp(prefix="job-", dir=root))
        _active_effects_directories.add(directory)
        return directory


def _cleanup_effects_directory(directory: Path) -> None:
    with _effects_directory_state_lock:
        _active_effects_directories.discard(directory)
        try:
            if not _remove_effects_directory(directory):
                logger.warning("Refused to remove an unsafe effects scratch entry")
        except OSError:
            logger.warning("Could not remove a private effects scratch directory")


@asynccontextmanager
async def _hold_effects_processing():
    """Admit one CPU/disk-heavy render without an unbounded waiter queue."""
    if not _effects_processing_lock.acquire(blocking=False):
        raise EffectsProcessingBusyError("Another effects render is already running; retry when it finishes")
    try:
        yield
    finally:
        _effects_processing_lock.release()


def _check_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise _EffectsProcessingCancelledError


async def _run_worker_cancellation_safe(function, /, *args, **kwargs):
    """Signal and drain the real worker before propagating cancellation."""
    cancel_event = threading.Event()
    operation = asyncio.create_task(asyncio.to_thread(function, *args, cancel_event=cancel_event, **kwargs))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError as cancellation:
        cancel_event.set()
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if not operation.cancelled():
            with suppress(BaseException):
                operation.result()
        raise cancellation


def _source_identity(source_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )


def _open_source_path(path: Path, flags: int) -> int:
    """Open a managed source component-by-component without following links."""
    root = Path(os.path.abspath(config.get_data_dir()))
    absolute = Path(os.path.abspath(path))
    try:
        relative = absolute.relative_to(root)
    except ValueError:
        return os.open(path, flags)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        return os.open(path, flags)

    if os.name != "posix" or os.open not in os.supports_dir_fd:
        current = root
        for component in relative.parts[:-1]:
            current /= component
            component_stat = current.lstat()
            is_junction = getattr(current, "is_junction", None)
            if (
                not stat.S_ISDIR(component_stat.st_mode)
                or stat.S_ISLNK(component_stat.st_mode)
                or (is_junction is not None and is_junction())
            ):
                raise OSError("Managed effects source parent is unsafe")
        return os.open(path, flags)

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    current_descriptor = os.open(root, directory_flags)
    try:
        for component in relative.parts[:-1]:
            next_descriptor = os.open(component, directory_flags, dir_fd=current_descriptor)
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        return os.open(relative.name, flags, dir_fd=current_descriptor)
    finally:
        os.close(current_descriptor)


def _open_source(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        lexical_stat = path.lstat()
        is_junction = getattr(path, "is_junction", None)
        if stat.S_ISLNK(lexical_stat.st_mode) or (is_junction is not None and is_junction()):
            raise OSError("Effects source is a link")
        descriptor = _open_source_path(path, flags)
    except OSError as exc:
        raise EffectsProcessingError("Source audio is unavailable or unsafe") from exc
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_nlink != 1:
            raise EffectsProcessingError("Source audio is not a regular file")
        if source_stat.st_size <= 0:
            raise EffectsProcessingError("Source audio is empty")
        if source_stat.st_size > EFFECTS_MAX_SOURCE_BYTES:
            raise EffectsProcessingLimitError(
                f"Source audio exceeds the {EFFECTS_MAX_SOURCE_BYTES // 1024**3} GiB file-size limit"
            )
        return descriptor, source_stat
    except BaseException:
        os.close(descriptor)
        raise


def _validate_source_shape(*, frames: int, sample_rate: int, channels: int) -> int:
    if frames <= 0 or sample_rate <= 0 or channels <= 0:
        raise EffectsProcessingError("Source audio has invalid duration metadata")
    if sample_rate > EFFECTS_MAX_SAMPLE_RATE:
        raise EffectsProcessingLimitError(f"Source audio exceeds the {EFFECTS_MAX_SAMPLE_RATE} Hz sample-rate limit")
    if channels > EFFECTS_MAX_CHANNELS:
        raise EffectsProcessingLimitError(f"Source audio exceeds the {EFFECTS_MAX_CHANNELS}-channel limit")
    if frames * channels > EFFECTS_MAX_DECODED_SAMPLE_VALUES:
        raise EffectsProcessingLimitError("Source audio exceeds the decoded-sample limit")
    duration = frames / sample_rate
    if not math.isfinite(duration) or duration <= 0:
        raise EffectsProcessingError("Source audio has invalid duration metadata")
    if duration > EFFECTS_MAX_DURATION_SECONDS:
        raise EffectsProcessingLimitError(
            f"Source audio exceeds the {EFFECTS_MAX_DURATION_SECONDS // 3600}-hour duration limit"
        )
    # libsoxr rounds positive half-frame results up.
    output_frames = (frames * EFFECTS_OUTPUT_SAMPLE_RATE + sample_rate // 2) // sample_rate
    if output_frames <= 0:
        raise EffectsProcessingError("Source audio is too short to process")
    return output_frames


def _copy_source_snapshot(
    source_descriptor: int,
    source_stat: os.stat_result,
    source_path: Path,
    scratch_directory: Path,
    cancel_event: threading.Event,
) -> Path:
    if source_stat.st_size > EFFECTS_MAX_COMPRESSED_SNAPSHOT_BYTES:
        raise EffectsProcessingLimitError("Compressed source audio exceeds the 512 MiB decoder-snapshot limit")
    try:
        reservation = reserve_disk_space(
            scratch_directory,
            source_stat.st_size,
            min_free_bytes=EFFECTS_MIN_FREE_BYTES,
        )
    except DiskSpaceReservationError as exc:
        raise EffectsProcessingStorageError(
            "Not enough disk space to snapshot compressed source audio while preserving the 1 GiB reserve"
        ) from exc
    suffix = source_path.suffix.lower()
    if not (2 <= len(suffix) <= 12 and suffix[1:].isalnum()):
        suffix = ".audio"
    snapshot = scratch_directory / f"source{suffix}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        output_descriptor = os.open(snapshot, flags, 0o600)
        copied = 0
        try:
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            while True:
                _check_cancelled(cancel_event)
                payload = os.read(source_descriptor, 1024 * 1024)
                if not payload:
                    break
                copied += len(payload)
                if copied > EFFECTS_MAX_COMPRESSED_SNAPSHOT_BYTES or copied > source_stat.st_size:
                    raise EffectsProcessingLimitError("Source audio changed or exceeded its snapshot limit")
                view = memoryview(payload)
                while view:
                    written = os.write(output_descriptor, view)
                    if written <= 0:
                        raise OSError("short write while snapshotting effects source")
                    view = view[written:]
            if copied != source_stat.st_size:
                raise EffectsProcessingError("Source audio changed while being snapshotted")
            os.fsync(output_descriptor)
        finally:
            os.close(output_descriptor)
    finally:
        reservation.release()
    return snapshot


def _probe_source(
    source_descriptor: int,
    source_stat: os.stat_result,
    source_path: Path,
    scratch_directory: Path,
    cancel_event: threading.Event,
) -> _ProbedSource:
    """Probe without decoding when libsndfile supports the source."""
    _check_cancelled(cancel_event)
    try:
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        with sf.SoundFile(source_descriptor, mode="r", closefd=False) as source:
            frames = int(source.frames)
            sample_rate = int(source.samplerate)
            channels = int(source.channels)
        return _ProbedSource(
            frames=frames,
            sample_rate=sample_rate,
            channels=channels,
            output_frames=_validate_source_shape(frames=frames, sample_rate=sample_rate, channels=channels),
            decoder="soundfile",
            snapshot_path=None,
        )
    except EffectsProcessingError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        pass

    snapshot = _copy_source_snapshot(
        source_descriptor,
        source_stat,
        source_path,
        scratch_directory,
        cancel_event,
    )
    try:
        with audioread.audio_open(str(snapshot)) as source:
            sample_rate = int(source.samplerate)
            channels = int(source.channels)
            frames = 0
            frame_bytes = channels * 2
            if frame_bytes <= 0:
                raise EffectsProcessingError("Source audio has invalid channel metadata")
            for payload in source:
                _check_cancelled(cancel_event)
                if len(payload) % frame_bytes:
                    raise EffectsProcessingError("Source audio contains a malformed decoded frame")
                frames += len(payload) // frame_bytes
                _validate_source_shape(frames=frames, sample_rate=sample_rate, channels=channels)
    except EffectsProcessingError:
        raise
    except Exception as exc:
        raise EffectsProcessingError("Source audio could not be decoded") from exc
    return _ProbedSource(
        frames=frames,
        sample_rate=sample_rate,
        channels=channels,
        output_frames=_validate_source_shape(frames=frames, sample_rate=sample_rate, channels=channels),
        decoder="audioread",
        snapshot_path=snapshot,
    )


def _mono_block(block: np.ndarray, channels: int) -> np.ndarray:
    if block.ndim != 2 or block.shape[1] != channels:
        raise EffectsProcessingError("Source audio changed while being processed")
    if not np.isfinite(block).all():
        raise EffectsProcessingError("Source audio contains non-finite samples")
    return block.mean(axis=1, dtype=np.float32)


def _decoded_mono_blocks(
    probed: _ProbedSource,
    source_descriptor: int,
    cancel_event: threading.Event,
) -> Iterator[np.ndarray]:
    if probed.decoder == "soundfile":
        try:
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            with sf.SoundFile(source_descriptor, mode="r", closefd=False) as source:
                if (
                    int(source.frames) != probed.frames
                    or int(source.samplerate) != probed.sample_rate
                    or int(source.channels) != probed.channels
                ):
                    raise EffectsProcessingError("Source audio changed while being processed")
                while True:
                    _check_cancelled(cancel_event)
                    block = source.read(EFFECTS_DECODE_BLOCK_FRAMES, dtype="float32", always_2d=True)
                    if not len(block):
                        break
                    yield _mono_block(block, probed.channels)
        except EffectsProcessingError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise EffectsProcessingError("Source audio could not be decoded") from exc
        return

    assert probed.snapshot_path is not None
    decoded_frames = 0
    try:
        with audioread.audio_open(str(probed.snapshot_path)) as source:
            if int(source.samplerate) != probed.sample_rate or int(source.channels) != probed.channels:
                raise EffectsProcessingError("Source audio changed while being processed")
            frame_bytes = probed.channels * 2
            for payload in source:
                _check_cancelled(cancel_event)
                if len(payload) % frame_bytes:
                    raise EffectsProcessingError("Source audio contains a malformed decoded frame")
                block = np.frombuffer(payload, dtype="<i2").reshape(-1, probed.channels)
                decoded_frames += len(block)
                yield _mono_block(block.astype(np.float32) / 32768.0, probed.channels)
    except EffectsProcessingError:
        raise
    except Exception as exc:
        raise EffectsProcessingError("Source audio could not be decoded") from exc
    if decoded_frames != probed.frames:
        raise EffectsProcessingError("Source audio produced an inconsistent decoded duration")


def _resampled_blocks(
    probed: _ProbedSource,
    source_descriptor: int,
    cancel_event: threading.Event,
) -> Iterator[np.ndarray]:
    blocks = _decoded_mono_blocks(probed, source_descriptor, cancel_event)
    if probed.sample_rate == EFFECTS_OUTPUT_SAMPLE_RATE:
        yield from blocks
        return
    resampler = soxr.ResampleStream(
        probed.sample_rate,
        EFFECTS_OUTPUT_SAMPLE_RATE,
        1,
        dtype="float32",
        quality="HQ",
    )
    for block in blocks:
        _check_cancelled(cancel_event)
        output = resampler.resample_chunk(block, last=False)
        if len(output):
            yield np.asarray(output, dtype=np.float32)
    output = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
    if len(output):
        yield np.asarray(output, dtype=np.float32)


def _coalesced_blocks(blocks: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
    buffer = np.empty(EFFECTS_PROCESS_BLOCK_FRAMES, dtype=np.float32)
    filled = 0
    for block in blocks:
        offset = 0
        while offset < len(block):
            count = min(len(block) - offset, len(buffer) - filled)
            buffer[filled : filled + count] = block[offset : offset + count]
            filled += count
            offset += count
            if filled == len(buffer):
                yield buffer.copy()
                filled = 0
    if filled:
        yield buffer[:filled].copy()


def _reserve_output_capacity(
    directory: Path,
    output_frames: int,
    *,
    scratch_copies: int = 0,
    include_output: bool = True,
) -> DiskSpaceReservation:
    # PCM16 mono plus conservative room for WAV/WAVEX headers.
    output_bytes = output_frames * 2 + 65_536 if include_output else 0
    required_bytes = output_bytes + output_frames * scratch_copies * np.dtype(np.float32).itemsize
    try:
        return reserve_disk_space(
            directory,
            required_bytes,
            min_free_bytes=EFFECTS_MIN_FREE_BYTES,
        )
    except DiskSpaceReservationError as exc:
        raise EffectsProcessingStorageError(
            "Not enough disk space to process the audio "
            f"({required_bytes / 1024**3:.1f} GiB required plus a 1 GiB reserve)"
        ) from exc


def _write_processed_block(output: sf.SoundFile, block: np.ndarray, remaining_frames: int) -> int:
    if block.ndim == 2:
        if block.shape[0] != 1:
            raise EffectsProcessingError("Effects chain produced an invalid channel layout")
        block = block[0]
    if block.ndim != 1:
        raise EffectsProcessingError("Effects chain produced invalid audio")
    count = min(len(block), remaining_frames)
    if count and not np.isfinite(block[:count]).all():
        raise EffectsProcessingError("Effects chain produced non-finite audio")
    if count:
        output.write(np.asarray(block[:count], dtype=np.float32))
    return count


def _mono_processed_block(block: np.ndarray) -> np.ndarray:
    """Return one finite mono DSP result without copying normal float32 data."""
    block = np.asarray(block)
    if block.ndim == 2:
        if block.shape[0] != 1:
            raise EffectsProcessingError("Effects chain produced an invalid channel layout")
        block = block[0]
    if block.ndim != 1:
        raise EffectsProcessingError("Effects chain produced invalid audio")
    block = np.asarray(block, dtype=np.float32)
    if not np.isfinite(block).all():
        raise EffectsProcessingError("Effects chain produced non-finite audio")
    return block


def _open_private_scratch_file(directory: Path, name: str) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return os.open(directory / name, flags, 0o600)


def _write_all(descriptor: int, payload: memoryview) -> None:
    while payload:
        written = os.write(descriptor, payload)
        if written <= 0:
            raise OSError("short write while storing effects scratch audio")
        payload = payload[written:]


def _write_raw_frames(descriptor: int, start: int, block: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(block, dtype=np.float32)
    os.lseek(descriptor, start * np.dtype(np.float32).itemsize, os.SEEK_SET)
    _write_all(descriptor, memoryview(contiguous).cast("B"))


def _read_raw_frames(descriptor: int, start: int, count: int) -> np.ndarray:
    if count < 0 or start < 0:
        raise EffectsProcessingError("Effects scratch read has invalid bounds")
    byte_count = count * np.dtype(np.float32).itemsize
    os.lseek(descriptor, start * np.dtype(np.float32).itemsize, os.SEEK_SET)
    payload = bytearray(byte_count)
    view = memoryview(payload)
    offset = 0
    while offset < byte_count:
        chunk = os.read(descriptor, byte_count - offset)
        if not chunk:
            raise EffectsProcessingError("Effects scratch audio ended unexpectedly")
        view[offset : offset + len(chunk)] = chunk
        offset += len(chunk)
    return np.frombuffer(payload, dtype=np.float32)


def _prepare_raw_destination(descriptor: int) -> None:
    entry_stat = os.fstat(descriptor)
    if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
        raise EffectsProcessingStorageError("Effects scratch output is not a private regular file")
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)


def _spool_float32_blocks(
    blocks: Iterator[np.ndarray],
    descriptor: int,
    expected_frames: int,
    cancel_event: threading.Event,
) -> None:
    _prepare_raw_destination(descriptor)
    frames = 0
    for block in blocks:
        _check_cancelled(cancel_event)
        block = np.asarray(block, dtype=np.float32)
        if block.ndim != 1 or not np.isfinite(block).all():
            raise EffectsProcessingError("Source audio contains invalid samples")
        if frames + len(block) > expected_frames:
            raise EffectsProcessingError("Source audio exceeded its expected duration")
        _write_raw_frames(descriptor, frames, block)
        frames += len(block)
    if frames != expected_frames:
        raise EffectsProcessingError("Source audio produced an inconsistent duration")


def _effect_segments(effects_chain: list[dict]) -> list[tuple[str, list[dict]]]:
    """Split pitch shifters from stateful neighbours without reordering DSP."""
    segments: list[tuple[str, list[dict]]] = []
    streaming: list[dict] = []
    for effect in effects_chain:
        if not effect.get("enabled", True):
            continue
        if effect["type"] == "pitch_shift":
            if streaming:
                segments.append(("stream", streaming))
                streaming = []
            segments.append(("pitch", [effect]))
        else:
            streaming.append(effect)
    if streaming:
        segments.append(("stream", streaming))
    return segments


def _process_streaming_raw_stage(
    source_descriptor: int,
    destination_descriptor: int,
    frames: int,
    sample_rate: int,
    effects_chain: list[dict],
    cancel_event: threading.Event,
) -> None:
    """Run a non-pitch board continuously and retain exactly ``frames``."""
    from ..utils.effects import build_pedalboard

    _prepare_raw_destination(destination_descriptor)
    board = build_pedalboard(effects_chain)
    input_frames = 0
    output_frames = 0
    while input_frames < frames:
        _check_cancelled(cancel_event)
        count = min(EFFECTS_PROCESS_BLOCK_FRAMES, frames - input_frames)
        block = _read_raw_frames(source_descriptor, input_frames, count)
        input_frames += count
        if input_frames == frames:
            # Keep the final real input and bounded flush context in one call.
            # Some buffered plugins cannot reconstruct their tail when silence
            # arrives in a separate Pedalboard.process invocation.
            flush_frames = max(EFFECTS_DECODE_BLOCK_FRAMES, math.ceil(sample_rate * EFFECTS_PITCH_GUARD_SECONDS))
            padded = np.zeros(count + flush_frames, dtype=np.float32)
            padded[:count] = block
            block = padded
        processed = _mono_processed_block(board(block[np.newaxis, :], sample_rate, reset=False))
        writable = min(len(processed), frames - output_frames)
        if writable:
            _write_raw_frames(destination_descriptor, output_frames, processed[:writable])
            output_frames += writable
    if output_frames != frames:
        raise EffectsProcessingError("Effects chain retained audio that could not be flushed")


def _adaptive_crossfade(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, float]:
    """Join two equal-time pitch windows and return the chosen mono polarity."""
    if len(previous) != len(current) or not len(previous):
        raise EffectsProcessingError("Pitch-shift overlap has invalid bounds")
    previous_64 = previous.astype(np.float64)
    current_64 = current.astype(np.float64)
    previous_energy = float(np.dot(previous_64, previous_64))
    current_energy = float(np.dot(current_64, current_64))
    correlation = 1.0
    if previous_energy > 0 and current_energy > 0:
        correlation = float(np.dot(previous_64, current_64) / math.sqrt(previous_energy * current_energy))
        correlation = min(1.0, max(-1.0, correlation))
    # Independent phase-vocoder windows can choose opposite waveform
    # polarity. Mono polarity is acoustically neutral, while crossfading two
    # anti-correlated windows creates an audible periodic level notch.
    polarity = -1.0 if correlation < 0 else 1.0
    if polarity < 0:
        current = -current
        correlation = -correlation
    progress = np.linspace(0.0, 1.0, len(previous), endpoint=False, dtype=np.float32)
    previous_weight = 1.0 - progress
    denominator = np.sqrt(
        previous_weight**2 + progress**2 + 2.0 * correlation * previous_weight * progress,
    )
    blended = np.asarray(
        (previous * previous_weight + current * progress) / denominator,
        dtype=np.float32,
    )
    # Global correlation cannot capture phase drift within the overlap. Match
    # its slowly varying energy envelope as well, so a residual local phase
    # cancellation cannot create a brief audible volume notch.
    smoothing_frames = min(len(previous), max(64, len(previous) // 8))
    smoothing_kernel = np.full(smoothing_frames, 1.0 / smoothing_frames, dtype=np.float64)
    target_power = np.convolve(
        previous_weight * previous_64**2 + progress * current.astype(np.float64) ** 2,
        smoothing_kernel,
        mode="same",
    )
    blended_power = np.convolve(
        blended.astype(np.float64) ** 2,
        smoothing_kernel,
        mode="same",
    )
    envelope_gain = np.sqrt(target_power / np.maximum(blended_power, np.finfo(np.float64).tiny))
    blended *= np.minimum(envelope_gain, 2.0).astype(np.float32)
    return blended, polarity


def _process_pitch_raw_stage(
    source_descriptor: int,
    destination_descriptor: int,
    frames: int,
    sample_rate: int,
    effect: dict,
    cancel_event: threading.Event,
) -> None:
    """Pitch-shift bounded guarded windows and crossfade their valid interiors."""
    from ..utils.effects import build_pedalboard

    _prepare_raw_destination(destination_descriptor)
    overlap = max(1, math.ceil(sample_rate * EFFECTS_PITCH_OVERLAP_SECONDS))
    overlap = min(overlap, max(1, EFFECTS_PROCESS_BLOCK_FRAMES // 8))
    guard = max(overlap, math.ceil(sample_rate * EFFECTS_PITCH_GUARD_SECONDS))
    written_end = 0

    for core_start in range(0, frames, EFFECTS_PROCESS_BLOCK_FRAMES):
        _check_cancelled(cancel_event)
        core_end = min(frames, core_start + EFFECTS_PROCESS_BLOCK_FRAMES)
        region_start = max(0, core_start - overlap)
        region_end = min(frames, core_end + overlap)
        input_start = max(0, region_start - guard)
        input_end = min(frames, region_end + guard)
        source = _read_raw_frames(source_descriptor, input_start, input_end - input_start)
        board = build_pedalboard([effect])
        processed = _mono_processed_block(board(source[np.newaxis, :], sample_rate))
        if len(processed) != len(source):
            raise EffectsProcessingError("Pitch shift produced an inconsistent duration")
        region = processed[region_start - input_start : region_end - input_start]

        if written_end == 0:
            _write_raw_frames(destination_descriptor, region_start, region)
            written_end = region_end
            continue
        overlap_frames = written_end - region_start
        if overlap_frames <= 0 or overlap_frames > len(region):
            raise EffectsProcessingError("Pitch-shift windows did not overlap safely")
        previous = _read_raw_frames(destination_descriptor, region_start, overlap_frames)
        blended, polarity = _adaptive_crossfade(previous, region[:overlap_frames])
        if polarity < 0:
            region = -region
        _write_raw_frames(destination_descriptor, region_start, blended)
        if overlap_frames < len(region):
            _write_raw_frames(
                destination_descriptor,
                region_start + overlap_frames,
                region[overlap_frames:],
            )
        written_end = region_end

    if written_end != frames:
        raise EffectsProcessingError("Pitch shift produced an inconsistent duration")


def _normalize_raw_in_place(
    source_descriptor: int,
    scratch_descriptor: int,
    frames: int,
    cancel_event: threading.Event,
) -> None:
    """Normalize a raw float32 stage before effects with legacy-identical math."""
    squared_file = None
    squared = None
    try:
        os.ftruncate(scratch_descriptor, frames * np.dtype(np.float32).itemsize)
        squared_file = os.fdopen(os.dup(scratch_descriptor), "r+b", closefd=True)
        squared = np.memmap(
            squared_file,
            dtype=np.float32,
            mode="r+",
            shape=(frames,),
        )
        for start in range(0, frames, EFFECTS_PROCESS_BLOCK_FRAMES):
            _check_cancelled(cancel_event)
            end = min(start + EFFECTS_PROCESS_BLOCK_FRAMES, frames)
            block = _read_raw_frames(source_descriptor, start, end - start)
            if not np.isfinite(block).all():
                raise EffectsProcessingError("Generated audio contains non-finite samples")
            np.square(block, out=squared[start:end])
        squared.flush()

        # This mirrors normalize_audio's contiguous float32 reduction rather
        # than summing blocks in float64, which would change samples.
        rms = np.sqrt(np.mean(squared))
        target_rms = 10 ** (-20.0 / 20.0)
        gain = target_rms / rms if rms > 0 else None
        for start in range(0, frames, EFFECTS_PROCESS_BLOCK_FRAMES):
            _check_cancelled(cancel_event)
            count = min(EFFECTS_PROCESS_BLOCK_FRAMES, frames - start)
            block = _read_raw_frames(source_descriptor, start, count)
            if gain is not None:
                np.multiply(block, gain, out=block)
            np.clip(block, -0.85, 0.85, out=block)
            _write_raw_frames(source_descriptor, start, block)
    finally:
        if squared is not None:
            with suppress(Exception):
                squared.flush()
            mapping = getattr(squared, "_mmap", None)
            if mapping is not None:
                with suppress(Exception):
                    mapping.close()
        if squared_file is not None:
            with suppress(Exception):
                squared_file.close()


def _encode_raw_wav(
    source_descriptor: int,
    output_descriptor: int,
    frames: int,
    sample_rate: int,
    cancel_event: threading.Event,
) -> None:
    os.lseek(output_descriptor, 0, os.SEEK_SET)
    os.ftruncate(output_descriptor, 0)
    with sf.SoundFile(
        output_descriptor,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        format="WAV",
        subtype="PCM_16",
        closefd=False,
    ) as output:
        for start in range(0, frames, EFFECTS_PROCESS_BLOCK_FRAMES):
            _check_cancelled(cancel_event)
            block = _read_raw_frames(
                source_descriptor,
                start,
                min(EFFECTS_PROCESS_BLOCK_FRAMES, frames - start),
            )
            _write_processed_block(output, block, frames - start)
        output.flush()
    os.fsync(output_descriptor)


def _render_disk_backed_chain_to_fd(
    blocks: Iterator[np.ndarray],
    output_descriptor: int,
    scratch_directory: Path,
    frames: int,
    sample_rate: int,
    effects_chain: list[dict],
    normalize: bool,
    cancel_event: threading.Event,
) -> None:
    """Run a long chain through two bounded scratch descriptors."""
    segments = _effect_segments(effects_chain)
    first_descriptor = _open_private_scratch_file(scratch_directory, "stage-a.float32")
    second_descriptor: int | None = None
    try:
        _spool_float32_blocks(blocks, first_descriptor, frames, cancel_event)
        current_descriptor = first_descriptor
        if normalize or segments:
            second_descriptor = _open_private_scratch_file(scratch_directory, "stage-b.float32")
        if normalize:
            assert second_descriptor is not None
            _normalize_raw_in_place(
                first_descriptor,
                second_descriptor,
                frames,
                cancel_event,
            )
        for kind, segment in segments:
            assert second_descriptor is not None
            destination_descriptor = second_descriptor if current_descriptor == first_descriptor else first_descriptor
            if kind == "pitch":
                _process_pitch_raw_stage(
                    current_descriptor,
                    destination_descriptor,
                    frames,
                    sample_rate,
                    segment[0],
                    cancel_event,
                )
            else:
                _process_streaming_raw_stage(
                    current_descriptor,
                    destination_descriptor,
                    frames,
                    sample_rate,
                    segment,
                    cancel_event,
                )
            current_descriptor = destination_descriptor
        _encode_raw_wav(
            current_descriptor,
            output_descriptor,
            frames,
            sample_rate,
            cancel_event,
        )
    finally:
        os.close(first_descriptor)
        if second_descriptor is not None:
            os.close(second_descriptor)


def _render_streaming_blocks_to_wav(
    blocks: Iterator[np.ndarray],
    output_descriptor: int,
    frames: int,
    sample_rate: int,
    effects_chain: list[dict],
    cancel_event: threading.Event,
) -> None:
    """Stream a chain without PitchShift directly into its final WAV."""
    from ..utils.effects import build_pedalboard

    board = build_pedalboard(effects_chain) if effects_chain else None
    iterator = iter(blocks)
    try:
        block = next(iterator)
    except StopIteration as exc:
        raise EffectsProcessingError("Source audio produced no decodable samples") from exc

    input_frames = 0
    output_frames = 0
    os.lseek(output_descriptor, 0, os.SEEK_SET)
    os.ftruncate(output_descriptor, 0)
    with sf.SoundFile(
        output_descriptor,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        format="WAV",
        subtype="PCM_16",
        closefd=False,
    ) as output:
        while True:
            _check_cancelled(cancel_event)
            try:
                following = next(iterator)
            except StopIteration:
                following = None
            real_frames = len(block)
            input_frames += real_frames
            if following is None and board is not None:
                flush_frames = max(
                    EFFECTS_DECODE_BLOCK_FRAMES,
                    math.ceil(sample_rate * EFFECTS_PITCH_GUARD_SECONDS),
                )
                padded = np.zeros(real_frames + flush_frames, dtype=np.float32)
                padded[:real_frames] = block
                block = padded
            processed = board(block[np.newaxis, :], sample_rate, reset=False) if board is not None else block
            output_frames += _write_processed_block(
                output,
                processed,
                frames - output_frames,
            )
            if following is None:
                break
            block = following

        if input_frames != frames:
            raise EffectsProcessingError("Source audio produced an inconsistent duration")
        if output_frames != frames:
            raise EffectsProcessingError("Effects chain retained audio that could not be flushed")
        output.flush()
    os.fsync(output_descriptor)


def _render_effects_audio_to_fd(
    source_path: Path,
    output_descriptor: int,
    output_directory: Path,
    scratch_directory: Path,
    effects_chain: list[dict],
    *,
    cancel_event: threading.Event,
) -> None:
    """Decode, process, and encode with duration-independent working memory."""
    from ..utils.effects import build_pedalboard

    output_stat = os.fstat(output_descriptor)
    if not stat.S_ISREG(output_stat.st_mode) or output_stat.st_nlink != 1:
        raise EffectsProcessingStorageError("Effects output is not a private regular file")
    try:
        if output_stat.st_dev != output_directory.stat().st_dev:
            raise EffectsProcessingStorageError("Effects output directory is on a different filesystem")
    except OSError as exc:
        raise EffectsProcessingStorageError("Could not identify the effects output filesystem") from exc
    source_descriptor, source_stat = _open_source(source_path)
    initial_identity = _source_identity(source_stat)
    output_reservation: DiskSpaceReservation | None = None
    scratch_reservation: DiskSpaceReservation | None = None
    try:
        probed = _probe_source(
            source_descriptor,
            source_stat,
            source_path,
            scratch_directory,
            cancel_event,
        )
        enabled_chain = [effect for effect in effects_chain if effect.get("enabled", True)]
        use_disk_backed_pitch = probed.output_frames > EFFECTS_PROCESS_BLOCK_FRAMES and any(
            effect["type"] == "pitch_shift" for effect in enabled_chain
        )
        output_reservation = _reserve_output_capacity(output_directory, probed.output_frames)
        if use_disk_backed_pitch:
            scratch_reservation = _reserve_output_capacity(
                scratch_directory,
                probed.output_frames,
                scratch_copies=2,
                include_output=False,
            )

        blocks = _coalesced_blocks(_resampled_blocks(probed, source_descriptor, cancel_event))
        if probed.output_frames <= EFFECTS_PROCESS_BLOCK_FRAMES:
            # Preserve Pedalboard's historical one-shot behavior for normal
            # TTS clips, including PitchShift's phase-vocoder padding.
            small_blocks = list(blocks)
            if not small_blocks:
                raise EffectsProcessingError("Source audio produced no decodable samples")
            block = np.concatenate(small_blocks) if len(small_blocks) > 1 else small_blocks[0]
            if len(block) != probed.output_frames:
                raise EffectsProcessingError("Source audio produced an inconsistent resampled duration")
            _check_cancelled(cancel_event)
            board = build_pedalboard(effects_chain) if enabled_chain else None
            processed = board(block[np.newaxis, :], EFFECTS_OUTPUT_SAMPLE_RATE) if board else block
            os.lseek(output_descriptor, 0, os.SEEK_SET)
            os.ftruncate(output_descriptor, 0)
            with sf.SoundFile(
                output_descriptor,
                mode="w",
                samplerate=EFFECTS_OUTPUT_SAMPLE_RATE,
                channels=1,
                format="WAV",
                subtype="PCM_16",
                closefd=False,
            ) as output:
                written = _write_processed_block(output, processed, probed.output_frames)
                if written != probed.output_frames:
                    raise EffectsProcessingError("Effects chain produced an inconsistent duration")
                output.flush()
            os.fsync(output_descriptor)
        elif use_disk_backed_pitch:
            _render_disk_backed_chain_to_fd(
                blocks,
                output_descriptor,
                scratch_directory,
                probed.output_frames,
                EFFECTS_OUTPUT_SAMPLE_RATE,
                effects_chain,
                False,
                cancel_event,
            )
        else:
            _render_streaming_blocks_to_wav(
                blocks,
                output_descriptor,
                probed.output_frames,
                EFFECTS_OUTPUT_SAMPLE_RATE,
                enabled_chain,
                cancel_event,
            )
        if _source_identity(os.fstat(source_descriptor)) != initial_identity:
            raise EffectsProcessingError("Source audio changed while being processed")
    finally:
        if scratch_reservation is not None:
            scratch_reservation.release()
        if output_reservation is not None:
            output_reservation.release()
        os.close(source_descriptor)


def _generated_audio_blocks(
    audio: np.ndarray,
    cancel_event: threading.Event,
) -> Iterator[np.ndarray]:
    for start in range(0, len(audio), EFFECTS_PROCESS_BLOCK_FRAMES):
        _check_cancelled(cancel_event)
        block = np.asarray(
            audio[start : start + EFFECTS_PROCESS_BLOCK_FRAMES],
            dtype=np.float32,
        )
        if not np.isfinite(block).all():
            raise EffectsProcessingError("Generated audio contains non-finite samples")
        yield block


def _validate_generated_audio_contract(audio: np.ndarray, sample_rate: int) -> int:
    if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.dtype != np.float32:
        raise EffectsProcessingError("Generated audio must be a one-dimensional float32 array")
    if type(sample_rate) is not int:
        raise EffectsProcessingError("Generated audio sample rate must be an integer")
    frames = len(audio)
    _validate_source_shape(frames=frames, sample_rate=sample_rate, channels=1)
    return frames


def _render_reserved_generated_audio_to_fd(
    audio: np.ndarray,
    sample_rate: int,
    output_descriptor: int,
    scratch_directory: Path,
    effects_chain: list[dict],
    normalize: bool,
    frames: int,
    enabled_chain: list[dict],
    use_disk_pipeline: bool,
    cancel_event: threading.Event,
) -> None:
    """Render after the complete future disk allocation has been reserved."""
    from ..utils.audio import normalize_audio
    from ..utils.effects import apply_effects

    if frames <= EFFECTS_PROCESS_BLOCK_FRAMES:
        _check_cancelled(cancel_event)
        if normalize:
            try:
                audio = normalize_audio(audio)
            except OSError as exc:
                raise EffectsProcessingStorageError("Could not allocate generated-audio normalization storage") from exc
        processed = apply_effects(audio, sample_rate, effects_chain) if effects_chain else audio
        processed = _mono_processed_block(processed)
        if len(processed) != frames:
            raise EffectsProcessingError("Effects chain produced an inconsistent duration")
        os.lseek(output_descriptor, 0, os.SEEK_SET)
        os.ftruncate(output_descriptor, 0)
        with sf.SoundFile(
            output_descriptor,
            mode="w",
            samplerate=sample_rate,
            channels=1,
            format="WAV",
            subtype="PCM_16",
            closefd=False,
        ) as output:
            if _write_processed_block(output, processed, frames) != frames:
                raise EffectsProcessingError("Effects chain produced an inconsistent duration")
            output.flush()
        os.fsync(output_descriptor)
        return

    if use_disk_pipeline:
        _render_disk_backed_chain_to_fd(
            _generated_audio_blocks(audio, cancel_event),
            output_descriptor,
            scratch_directory,
            frames,
            sample_rate,
            effects_chain,
            normalize,
            cancel_event,
        )
        return

    blocks = _generated_audio_blocks(audio, cancel_event)
    _render_streaming_blocks_to_wav(
        blocks,
        output_descriptor,
        frames,
        sample_rate,
        enabled_chain,
        cancel_event,
    )


def _render_generated_audio_to_fd(
    audio: np.ndarray,
    sample_rate: int,
    output_descriptor: int,
    output_directory: Path,
    scratch_directory: Path,
    effects_chain: list[dict],
    normalize: bool,
    *,
    cancel_event: threading.Event,
) -> None:
    """Post-process one generated array without constructing WAV bytes."""
    output_stat = os.fstat(output_descriptor)
    if not stat.S_ISREG(output_stat.st_mode) or output_stat.st_nlink != 1:
        raise EffectsProcessingStorageError("Generated audio output is not a private regular file")
    try:
        if output_stat.st_dev != output_directory.stat().st_dev:
            raise EffectsProcessingStorageError("Generated audio output directory is on a different filesystem")
    except OSError as exc:
        raise EffectsProcessingStorageError("Could not identify the generated audio output filesystem") from exc
    frames = _validate_generated_audio_contract(audio, sample_rate)
    enabled_chain = [effect for effect in effects_chain if effect.get("enabled", True)]
    use_disk_pipeline = frames > EFFECTS_PROCESS_BLOCK_FRAMES and (
        normalize or any(effect["type"] == "pitch_shift" for effect in enabled_chain)
    )
    output_reservation = _reserve_output_capacity(output_directory, frames)
    scratch_reservation: DiskSpaceReservation | None = None
    try:
        if use_disk_pipeline:
            scratch_reservation = _reserve_output_capacity(
                scratch_directory,
                frames,
                scratch_copies=2,
                include_output=False,
            )
        _render_reserved_generated_audio_to_fd(
            audio,
            sample_rate,
            output_descriptor,
            scratch_directory,
            effects_chain,
            normalize,
            frames,
            enabled_chain,
            use_disk_pipeline,
            cancel_event,
        )
    finally:
        if scratch_reservation is not None:
            scratch_reservation.release()
        output_reservation.release()


async def render_effects_to_descriptor(
    source_path: Path,
    output_descriptor: int,
    output_directory: Path,
    effects_chain: list[dict],
) -> None:
    """Render to a caller-owned durable descriptor and clean all scratch."""
    from ..utils.effects import validate_effects_chain

    validation_error = validate_effects_chain(effects_chain)
    if validation_error:
        raise EffectsProcessingError(validation_error)
    async with _hold_effects_processing():
        scratch_directory = _allocate_effects_directory()
        try:
            await _run_worker_cancellation_safe(
                _render_effects_audio_to_fd,
                source_path,
                output_descriptor,
                output_directory,
                scratch_directory,
                effects_chain,
            )
        finally:
            _cleanup_effects_directory(scratch_directory)


async def render_generated_audio_to_descriptor(
    audio: np.ndarray,
    sample_rate: int,
    output_descriptor: int,
    output_directory: Path,
    effects_chain: list[dict],
    *,
    normalize: bool = False,
) -> None:
    """Render generated audio into a caller-owned regular-file descriptor.

    Long arrays are consumed blockwise (including disk-backed ``np.memmap``
    outputs from chunked TTS). Cancellation signals and drains the real worker
    before returning, so the caller can safely reconcile or remove its
    journaled output inode.
    """
    from ..utils.effects import validate_effects_chain

    validation_error = validate_effects_chain(effects_chain)
    if validation_error:
        raise EffectsProcessingError(validation_error)
    if type(normalize) is not bool:
        raise EffectsProcessingError("normalize must be a boolean")
    _validate_generated_audio_contract(audio, sample_rate)
    async with _hold_effects_processing():
        scratch_directory = _allocate_effects_directory()
        try:
            await _run_worker_cancellation_safe(
                _render_generated_audio_to_fd,
                audio,
                sample_rate,
                output_descriptor,
                output_directory,
                scratch_directory,
                effects_chain,
                normalize,
            )
        finally:
            _cleanup_effects_directory(scratch_directory)


async def create_generated_audio_response_file(
    audio: np.ndarray,
    sample_rate: int,
    effects_chain: list[dict],
    normalize: bool,
) -> GeneratedAudioFile:
    """Encode generated mono float32 audio into a private response-owned WAV."""
    from ..utils.effects import validate_effects_chain

    validation_error = validate_effects_chain(effects_chain)
    if validation_error:
        raise EffectsProcessingError(validation_error)
    if type(normalize) is not bool:
        raise EffectsProcessingError("normalize must be a boolean")
    # Shape/rate checks are allocation-free and reject invalid jobs before they
    # consume the single heavy-render admission slot.
    _validate_generated_audio_contract(audio, sample_rate)
    async with _hold_effects_processing():
        scratch_directory = _allocate_effects_directory()
        output_path = scratch_directory / "speech.wav"
        descriptor: int | None = None
        try:
            descriptor = _open_private_scratch_file(scratch_directory, output_path.name)
            await _run_worker_cancellation_safe(
                _render_generated_audio_to_fd,
                audio,
                sample_rate,
                descriptor,
                scratch_directory,
                scratch_directory,
                effects_chain,
                normalize,
            )
            os.close(descriptor)
            descriptor = None
            return GeneratedAudioFile(
                path=output_path,
                temporary_directory=scratch_directory,
            )
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            _cleanup_effects_directory(scratch_directory)
            raise


async def create_effects_preview(source_path: Path, effects_chain: list[dict]) -> EffectsPreview:
    """Create a private streamed preview without timeline-sized RAM."""
    from ..utils.effects import validate_effects_chain

    validation_error = validate_effects_chain(effects_chain)
    if validation_error:
        raise EffectsProcessingError(validation_error)
    async with _hold_effects_processing():
        scratch_directory = _allocate_effects_directory()
        output_path = scratch_directory / "preview.wav"
        descriptor: int | None = None
        try:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(output_path, flags, 0o600)
            await _run_worker_cancellation_safe(
                _render_effects_audio_to_fd,
                source_path,
                descriptor,
                scratch_directory,
                scratch_directory,
                effects_chain,
            )
            os.close(descriptor)
            descriptor = None
            return EffectsPreview(path=output_path, temporary_directory=scratch_directory)
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            _cleanup_effects_directory(scratch_directory)
            raise
