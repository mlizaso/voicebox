"""
Unified TTS generation orchestration.

Replaces the three near-identical closures (_run_generation, _run_retry,
_run_regenerate) that lived in main.py with a single ``run_generation()``
function parameterized by *mode*.

Mode differences:
  - "generate"   : full pipeline -- save clean version, optionally apply
                    effects and create a processed version.
  - "retry"      : re-runs a failed generation with the same seed.
                    No effects, no version creation.
  - "regenerate" : re-runs with seed=None for variation.  Creates a new
                    version with an auto-incremented "take-N" label.
"""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .. import config
from ..database import Generation as DBGeneration, get_db
from ..utils.disk_reservations import DiskSpaceReservationError, reserve_disk_space
from ..utils.tasks import get_task_manager
from . import deletion_journal, history, profiles

logger = logging.getLogger(__name__)
SYNC_AUDIO_MAX_DURATION_SECONDS = 10 * 60
GENERATION_AUDIO_PUBLICATION_MIN_FREE_BYTES = 1024**3


@dataclass(frozen=True)
class ExactBatchGenerationSpec:
    """Frozen persistence and post-processing contract for one batch unit."""

    generation_id: str
    profile_id: str
    text: str
    language: str
    engine: str
    model_size: str
    seed: int | None
    normalize: bool
    effects_chain: list | None
    instruct: str | None
    crossfade_ms: int
    expected_voice_binding_sha256: str | None = None
    exact_voice_snapshot: dict | None = None
    expected_tts_implementation_revision: str | None = None


async def run_exact_generation_batch(
    specs: list[ExactBatchGenerationSpec],
) -> None:
    """Run two independently persisted audiobook units in one model call."""
    if len(specs) != 2:
        raise ValueError("exact generation batches require exactly two units")
    first = specs[0]
    homogeneous_fields = (
        "profile_id",
        "language",
        "engine",
        "model_size",
        "instruct",
        "expected_voice_binding_sha256",
        "exact_voice_snapshot",
        "expected_tts_implementation_revision",
    )
    if any(getattr(spec, field) != getattr(first, field) for spec in specs[1:] for field in homogeneous_fields):
        raise ValueError("exact generation batch units must share one model and voice")

    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        get_tts_backend_for_engine,
    )
    from ..backends.mlx_tts_lifecycle import (
        loaded_tts_backend_for_request,
        run_blocking_operation_cancellation_safe,
        run_tts_operation_cancellation_safe,
    )
    from ..utils.audio import has_tts_runaway, normalize_audio, trim_tts_output
    from ..utils.chunked_tts import generate_text_batch, release_disk_backed_audio

    task_manager = get_task_manager()
    bg_db = next(get_db())
    completed_ids: set[str] = set()
    results: list[tuple[object, int]] = []

    async def update_active(status: str, *, error: str | None = None) -> None:
        for spec in specs:
            if spec.generation_id not in completed_ids:
                await history.update_generation_status(
                    spec.generation_id,
                    status,
                    bg_db,
                    error=error,
                )

    try:
        if (
            first.expected_voice_binding_sha256 is None
            or first.exact_voice_snapshot is None
            or first.expected_tts_implementation_revision is None
        ):
            raise ValueError("Exact model batches require a frozen voice and runtime revision")
        profiles._require_exact_tts_revision(first.expected_tts_implementation_revision)
        tts_model = get_tts_backend_for_engine(first.engine)
        if not tts_model.is_loaded():
            await update_active("loading_model")
        async with loaded_tts_backend_for_request(
            first.engine,
            first.model_size,
        ) as tts_model:
            voice_prompt = await run_tts_operation_cancellation_safe(
                tts_model,
                profiles.create_exact_voice_prompt_from_snapshot(
                    first.exact_voice_snapshot,
                    expected_voice_binding_sha256=first.expected_voice_binding_sha256,
                    expected_tts_implementation_revision=(first.expected_tts_implementation_revision),
                    engine=first.engine,
                ),
            )
            profiles._require_exact_tts_revision(first.expected_tts_implementation_revision)
            await update_active("generating")

            trim_fn = trim_tts_output if engine_needs_trim(first.engine) else None
            runaway_detector = has_tts_runaway if engine_retries_runaway(first.engine) else None
            results = await generate_text_batch(
                tts_model,
                [spec.text for spec in specs],
                voice_prompt,
                language=first.language,
                seeds=[spec.seed for spec in specs],
                instruct=first.instruct,
                crossfade_ms=first.crossfade_ms,
                trim_fn=trim_fn,
                runaway_detector=runaway_detector,
            )

        for spec, (audio, sample_rate) in zip(specs, results, strict=True):
            if spec.normalize:
                audio = await run_blocking_operation_cancellation_safe(normalize_audio, audio)
            duration = len(audio) / sample_rate
            final_path = await _save_generate(
                generation_id=spec.generation_id,
                audio=audio,
                sample_rate=sample_rate,
                effects_chain=spec.effects_chain,
                db=bg_db,
            )
            _durably_sync_exact_generation_audio(
                spec.generation_id,
                final_path,
            )
            await history.update_generation_status(
                generation_id=spec.generation_id,
                status="completed",
                db=bg_db,
                audio_path=final_path,
                duration=duration,
            )
            completed_ids.add(spec.generation_id)
            _notify_speak_end(spec.generation_id, status="completed")

    except asyncio.CancelledError:
        await update_active("failed", error="Generation batch cancelled")
        for spec in specs:
            if spec.generation_id not in completed_ids:
                _notify_speak_end(spec.generation_id, status="cancelled")
    except Exception as exc:
        traceback.print_exc()
        await update_active("failed", error=str(exc))
        for spec in specs:
            if spec.generation_id not in completed_ids:
                _notify_speak_end(spec.generation_id, status="failed")
    finally:
        for audio, _sample_rate in results:
            release_disk_backed_audio(audio)
        for spec in specs:
            task_manager.complete_generation(spec.generation_id)
        bg_db.close()


async def run_generation(
    *,
    generation_id: str,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: int | None,
    normalize: bool = False,
    effects_chain: list | None = None,
    instruct: str | None = None,
    mode: Literal["generate", "retry", "regenerate"],
    max_chunk_chars: int | None = None,
    crossfade_ms: int | None = None,
    version_id: str | None = None,
    expected_voice_binding_sha256: str | None = None,
    exact_voice_snapshot: dict | None = None,
    expected_tts_implementation_revision: str | None = None,
    exact_request_sha256: str | None = None,
) -> None:
    """Execute TTS inference and persist the result.

    This is the single entry point for all background generation work.
    It is designed to be enqueued via ``services.task_queue.enqueue_generation``.
    """
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        get_tts_backend_for_engine,
    )
    from ..backends.mlx_tts_lifecycle import (
        loaded_tts_backend_for_request,
        run_blocking_operation_cancellation_safe,
        run_tts_operation_cancellation_safe,
    )
    from ..utils.audio import has_tts_runaway, normalize_audio, trim_tts_output
    from ..utils.chunked_tts import (
        DEFAULT_MAX_CHUNK_CHARS,
        generate_chunked,
        release_disk_backed_audio,
        split_text_into_chunks,
    )

    task_manager = get_task_manager()
    bg_db = next(get_db())

    checkpoint_session = None
    audio = None
    try:
        if expected_voice_binding_sha256 is not None:
            if exact_voice_snapshot is None or expected_tts_implementation_revision is None:
                raise ValueError("Exact generation requires a frozen voice and runtime revision")
            profiles._require_exact_tts_revision(expected_tts_implementation_revision)
        tts_model = get_tts_backend_for_engine(engine)

        if not tts_model.is_loaded():
            await history.update_generation_status(generation_id, "loading_model", bg_db)

        async with loaded_tts_backend_for_request(engine, model_size) as tts_model:
            if exact_voice_snapshot is not None:
                voice_prompt = await run_tts_operation_cancellation_safe(
                    tts_model,
                    profiles.create_exact_voice_prompt_from_snapshot(
                        exact_voice_snapshot,
                        expected_voice_binding_sha256=expected_voice_binding_sha256,
                        expected_tts_implementation_revision=(expected_tts_implementation_revision),
                        engine=engine,
                    ),
                )
                profiles._require_exact_tts_revision(expected_tts_implementation_revision)
            else:
                voice_prompt = await run_tts_operation_cancellation_safe(
                    tts_model,
                    profiles.create_voice_prompt_for_profile(
                        profile_id,
                        bg_db,
                        use_cache=True,
                        engine=engine,
                    ),
                )

            if exact_request_sha256 is not None:
                if (
                    mode != "generate"
                    or engine != "qwen"
                    or seed is None
                    or expected_voice_binding_sha256 is None
                    or exact_voice_snapshot is None
                    or expected_tts_implementation_revision is None
                ):
                    raise ValueError("Chunk checkpoints require a seeded exact singleton Qwen request")
                effective_max_chunk_chars = max_chunk_chars or DEFAULT_MAX_CHUNK_CHARS
                if len(split_text_into_chunks(text, effective_max_chunk_chars)) > 1:
                    from .exact_chunk_checkpoints import ExactChunkCheckpointSession

                    # Construct this only after the immutable prompt helper has copied,
                    # hashed, and verified the exact voice bytes used by inference. A
                    # one-chunk request cannot recover any completed prefix, so it
                    # deliberately avoids checkpoint disk I/O.
                    checkpoint_session = ExactChunkCheckpointSession(exact_request_sha256)

            await history.update_generation_status(generation_id, "generating", bg_db)
            trim_fn = trim_tts_output if engine_needs_trim(engine) else None
            runaway_detector = has_tts_runaway if engine_retries_runaway(engine) else None

            gen_kwargs: dict = dict(
                language=language,
                seed=seed if mode != "regenerate" else None,
                instruct=instruct,
                trim_fn=trim_fn,
                runaway_detector=runaway_detector,
                checkpoint_session=checkpoint_session,
            )
            if max_chunk_chars is not None:
                gen_kwargs["max_chunk_chars"] = max_chunk_chars
            if crossfade_ms is not None:
                gen_kwargs["crossfade_ms"] = crossfade_ms

            audio, sample_rate = await generate_chunked(
                tts_model,
                text,
                voice_prompt,
                **gen_kwargs,
            )

        # --- Normalize (generate and regenerate always; retry skips) -----
        if normalize or mode == "regenerate":
            audio = await run_blocking_operation_cancellation_safe(normalize_audio, audio)

        duration = len(audio) / sample_rate

        # --- Persist audio and update status -----------------------------
        if mode == "generate":
            final_path = await _save_generate(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                effects_chain=effects_chain,
                db=bg_db,
            )
        elif mode == "retry":
            final_path = await _save_retry(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                duration=duration,
                db=bg_db,
            )
        elif mode == "regenerate":
            final_path = await _save_regenerate(
                generation_id=generation_id,
                version_id=version_id,
                audio=audio,
                sample_rate=sample_rate,
                db=bg_db,
            )

        if exact_request_sha256 is not None:
            _durably_sync_exact_generation_audio(generation_id, final_path)

        completed = await history.update_generation_status(
            generation_id=generation_id,
            status="completed",
            db=bg_db,
            audio_path=final_path,
            duration=duration,
        )
        if completed is None:
            raise RuntimeError("Generation row disappeared before completion was persisted")
        if checkpoint_session is not None:
            # Cleanup is deliberately after WAV/version durability and the
            # completed-status DB commit. Failures are nonfatal and later
            # bounded cleanup may remove only abandoned temporary files.
            checkpoint_session.complete(bg_db)

    except asyncio.CancelledError:
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error="Generation cancelled",
        )
        _notify_speak_end(generation_id, status="cancelled")
    except Exception as e:
        traceback.print_exc()
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error=str(e),
        )
        _notify_speak_end(generation_id, status="failed")
    else:
        _notify_speak_end(generation_id, status="completed")
    finally:
        release_disk_backed_audio(audio)
        task_manager.complete_generation(generation_id)
        bg_db.close()


def _notify_speak_end(generation_id: str, *, status: str) -> None:
    """Publish a speak-end event; the frontend ignores unknown ids."""
    try:
        from ..mcp_server import events as mcp_events

        mcp_events.publish(
            "speak-end",
            {"generation_id": generation_id, "status": status},
        )
    except Exception:
        # Never let event pub/sub break generation completion.
        pass


def _durably_sync_exact_generation_audio(generation_id: str, final_path: str) -> None:
    """Fsync exact output artifacts before their checkpoints are removed."""
    resolved_final_path = config.resolve_storage_path(final_path)
    if resolved_final_path is None:
        raise OSError("Completed exact generation has no final audio path")
    paths = {
        config.get_generations_dir() / f"{generation_id}.wav",
        resolved_final_path,
    }
    parent_paths: set = set()
    for path in paths:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise OSError(f"Exact generation output is not a regular file: {path}")
            os.fsync(fd)
        finally:
            os.close(fd)
        parent_paths.add(path.parent)

    if os.name != "nt":
        for parent in parent_paths:
            flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                flags |= os.O_DIRECTORY
            fd = os.open(parent, flags)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


@dataclass(frozen=True)
class _GenerationAudioPublication:
    """One fully populated WAV awaiting durable database ownership."""

    storage_path: str
    intent: deletion_journal.DeletionIntent


def _populate_generation_audio_payload(
    descriptor: int,
    audio,
    sample_rate: int,
    intent: deletion_journal.DeletionIntent,
) -> None:
    """Encode WAV bytes directly into an already journaled managed inode."""
    import soundfile as sf

    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not deletion_journal.entry_matches_intent(intent, before)
    ):
        raise OSError("Journaled generation staging file identity changed")

    # The anonymous float32 accumulator has already consumed its disk space.
    # Preserve the same reserve while creating the durable PCM16 WAV rather
    # than discovering ENOSPC after hours of successful inference.
    required_bytes = len(audio) * 2 + 65_536
    try:
        reservation = reserve_disk_space(
            config.get_generations_dir(),
            required_bytes,
            min_free_bytes=GENERATION_AUDIO_PUBLICATION_MIN_FREE_BYTES,
        )
    except DiskSpaceReservationError as exc:
        raise OSError("Insufficient free space to publish generated audio while preserving the disk reserve") from exc
    with reservation:
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        # libsndfile accepts a seekable binary stream. Keeping the original
        # descriptor open prevents a path-based encoder from replacing the inode
        # recorded by the durable publication intent.
        with open(descriptor, "r+b", closefd=False) as output:
            sf.write(output, audio, sample_rate, format="WAV")
            output.flush()
        os.fsync(descriptor)

    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or not deletion_journal.entry_matches_intent(intent, after)
    ):
        raise OSError("Journaled generation WAV identity changed during encoding")


def _reconcile_generation_audio_publication(
    publication: _GenerationAudioPublication,
    db,
) -> None:
    """Resolve an interrupted publish from a fresh view of durable ownership."""
    try:
        with deletion_journal.durable_reconciliation_session(db) as durable_db:
            deletion_journal.reconcile_deletion_intent(
                publication.intent,
                durable_db,
            )
    except BaseException:
        logger.error(
            "Retaining one generation-audio publication intent for startup recovery",
            exc_info=True,
        )


def _rollback_and_reconcile_generation_audio(
    publication: _GenerationAudioPublication,
    db,
) -> None:
    try:
        db.rollback()
    except BaseException:
        logger.error("Generation-audio ownership rollback failed", exc_info=True)
    _reconcile_generation_audio_publication(publication, db)


def _finish_generation_audio_publication(
    publication: _GenerationAudioPublication,
) -> None:
    try:
        deletion_journal.finish_deletion_intent(publication.intent)
    except OSError:
        logger.warning("Deferred cleanup of a committed generation-audio publication intent")


def _rollback_and_reconcile_retry_audio(
    publication: _GenerationAudioPublication,
    staged_predecessors: list[history._StagedGenerationAudio],
    db,
) -> None:
    """Resolve both sides of a retry replacement from durable DB ownership."""
    try:
        db.rollback()
    except BaseException:
        logger.error("Retry-audio ownership rollback failed", exc_info=True)
    try:
        with deletion_journal.durable_reconciliation_session(db) as durable_db:
            deletion_journal.reconcile_deletion_intent(
                publication.intent,
                durable_db,
            )
            history._reconcile_staged_generation_audio(
                staged_predecessors,
                durable_db,
            )
    except BaseException:
        logger.error(
            "Retaining retry-audio intents for startup recovery",
            exc_info=True,
        )


def _retry_audio_commit_succeeded(
    *,
    generation_id: str,
    publication: _GenerationAudioPublication,
    duration: float | None,
    db,
) -> bool:
    """Resolve a retry commit acknowledgement from durable row state."""
    try:
        with deletion_journal.durable_reconciliation_session(db) as durable_db:
            durable_generation = durable_db.query(DBGeneration).filter_by(id=generation_id).one_or_none()
            if durable_generation is None or durable_generation.audio_path != publication.storage_path:
                return False
            return duration is None or durable_generation.duration == duration
    except BaseException:
        logger.error("Could not determine durable retry-audio ownership", exc_info=True)
        return False


async def _populate_processed_generation_audio_payload(
    descriptor: int,
    audio,
    sample_rate: int,
    effects_chain: list,
    intent: deletion_journal.DeletionIntent,
) -> None:
    """Render effects directly into the journaled publication inode."""
    from . import effects_processing

    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not deletion_journal.entry_matches_intent(intent, before)
    ):
        raise OSError("Journaled generation staging file identity changed")
    await effects_processing.render_generated_audio_to_descriptor(
        audio,
        sample_rate,
        descriptor,
        config.get_generations_dir(),
        effects_chain,
        normalize=False,
    )
    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or not deletion_journal.entry_matches_intent(intent, after)
    ):
        raise OSError("Journaled generation WAV identity changed during effects rendering")


async def _publish_generation_audio(
    *,
    target: Path,
    audio,
    sample_rate: int,
    db,
    effects_chain: list | None = None,
) -> _GenerationAudioPublication:
    """Stage, fsync, and atomically expose one WAV behind a durable intent."""
    generations_dir = config.get_generations_dir()
    if target.parent != generations_dir or target.name != Path(target.name).name:
        raise ValueError("Generation audio target is outside managed storage")

    target_relative = Path("generations") / target.name
    staged_relative = Path("generations") / f".voicebox-delete-publish-{uuid.uuid4().hex}.wav"
    staged_path = config.get_data_dir() / staged_relative
    descriptor: int | None = None
    publication: _GenerationAudioPublication | None = None
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(staged_path, flags, 0o600)
        os.fsync(descriptor)
        staged_stat = os.fstat(descriptor)
        intent = deletion_journal.prepare_deletion_intent(
            kind=deletion_journal.GENERATION_AUDIO_PUBLICATION,
            original=target_relative,
            staged=staged_relative,
            entry_stat=staged_stat,
        )
        publication = _GenerationAudioPublication(
            storage_path=target_relative.as_posix(),
            intent=intent,
        )

        if effects_chain is None:
            from ..backends.mlx_tts_lifecycle import run_blocking_operation_cancellation_safe

            await run_blocking_operation_cancellation_safe(
                _populate_generation_audio_payload,
                descriptor,
                audio,
                sample_rate,
                intent,
            )
        else:
            await _populate_processed_generation_audio_payload(
                descriptor,
                audio,
                sample_rate,
                effects_chain,
                intent,
            )
        staged_stat = deletion_journal.managed_entry_stat(staged_relative)
        if staged_stat is None or not deletion_journal.entry_matches_intent(intent, staged_stat):
            raise OSError("Journaled generation WAV was replaced before publication")
        os.close(descriptor)
        descriptor = None

        deletion_journal.replace_managed_entry(staged_relative, target_relative)
        published_stat = deletion_journal.managed_entry_stat(target_relative)
        if published_stat is None or not deletion_journal.entry_matches_intent(intent, published_stat):
            raise OSError("Published generation WAV identity changed")
        # Apply the configured private/shared mode only after the complete WAV
        # is public. The rename already fsynced its directory entry.
        publication = _GenerationAudioPublication(
            storage_path=config.to_storage_path(target),
            intent=intent,
        )
        return publication
    except BaseException:
        if publication is not None:
            _rollback_and_reconcile_generation_audio(publication, db)
        else:
            try:
                deletion_journal.discard_managed_entry(staged_relative)
            except BaseException:
                logger.error("Deferred cleanup of an unpublished generation WAV", exc_info=True)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


async def _save_generate(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    effects_chain: list | None,
    db,
) -> str:
    """Save clean version and optionally an effects-processed version.

    Returns the final audio path (processed if effects were applied,
    otherwise clean).
    """
    from . import versions as versions_mod

    has_effects = effects_chain and any(e.get("enabled", True) for e in effects_chain)
    clean_audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    clean_publication = await _publish_generation_audio(
        target=clean_audio_path,
        audio=audio,
        sample_rate=sample_rate,
        db=db,
    )

    try:
        versions_mod.create_version(
            generation_id=generation_id,
            label="original",
            audio_path=clean_publication.storage_path,
            db=db,
            effects_chain=None,
            is_default=not has_effects,
        )
    except BaseException:
        _rollback_and_reconcile_generation_audio(clean_publication, db)
        raise
    _finish_generation_audio_publication(clean_publication)

    final_audio_path = str(clean_audio_path)

    if has_effects:
        from ..utils.effects import validate_effects_chain

        assert effects_chain is not None

        error_msg = validate_effects_chain(effects_chain)
        if error_msg:
            import logging

            logging.getLogger(__name__).warning("invalid effects chain, skipping: %s", error_msg)
            versions_mod.set_default_version(versions_mod.list_versions(generation_id, db)[0].id, db)
        else:
            processed_path = config.get_generations_dir() / f"{generation_id}_processed.wav"
            processed_publication = await _publish_generation_audio(
                target=processed_path,
                audio=audio,
                sample_rate=sample_rate,
                db=db,
                effects_chain=effects_chain,
            )
            final_audio_path = str(processed_path)
            try:
                versions_mod.create_version(
                    generation_id=generation_id,
                    label="version-2",
                    audio_path=processed_publication.storage_path,
                    db=db,
                    effects_chain=effects_chain,
                    is_default=True,
                )
            except BaseException:
                _rollback_and_reconcile_generation_audio(processed_publication, db)
                raise
            _finish_generation_audio_publication(processed_publication)

    return config.to_storage_path(final_audio_path)


async def _save_retry(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    db,
    duration: float | None = None,
) -> str:
    """Save retry output -- single file, no versions.

    Returns the audio path.
    """
    db.rollback()
    db.expire_all()
    generation = db.query(DBGeneration).populate_existing().filter_by(id=generation_id).one_or_none()
    if generation is None:
        raise RuntimeError("Generation row disappeared before retry audio was persisted")
    predecessor_path = generation.audio_path

    # Never replace an owned predecessor in place: publication recovery uses
    # database ownership to choose old versus new bytes after an ambiguous
    # commit, which requires the two inodes to have distinct managed names.
    audio_path = config.get_generations_dir() / f"{generation_id}_retry_{uuid.uuid4().hex}.wav"
    publication = await _publish_generation_audio(
        target=audio_path,
        audio=audio,
        sample_rate=sample_rate,
        db=db,
    )
    staged_predecessors: list[history._StagedGenerationAudio] = []
    try:
        if predecessor_path:
            from . import versions as versions_mod

            unreferenced = versions_mod._unreferenced_managed_audio_paths(
                [predecessor_path],
                db,
                deleting_version_ids=set(),
                changing_generation_ids={generation_id},
            )
            for stored_path in unreferenced:
                staged = history._stage_managed_generation_audio(stored_path)
                if staged is not None:
                    staged_predecessors.append(staged)
        generation.audio_path = publication.storage_path
        if duration is not None:
            generation.duration = duration
        db.commit()
    except BaseException:
        _rollback_and_reconcile_retry_audio(
            publication,
            staged_predecessors,
            db,
        )
        if not _retry_audio_commit_succeeded(
            generation_id=generation_id,
            publication=publication,
            duration=duration,
            db=db,
        ):
            raise
        # The driver reported failure after the row commit became durable.
        # Returning success prevents the outer generation handler from
        # overwriting a valid long retry as failed and needlessly rerunning it.
        return publication.storage_path
    _finish_generation_audio_publication(publication)
    history._discard_staged_generation_audio(staged_predecessors)
    return publication.storage_path


async def generate_audio_sync(
    *,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: int | None = None,
    instruct: str | None = None,
    normalize: bool = True,
    max_chunk_chars: int | None = None,
    crossfade_ms: int | None = None,
) -> bytes:
    """Run a TTS generation synchronously and return the resulting wav bytes.

    Unlike :func:`run_generation`, this path does not touch the
    ``generations`` table, enqueue work, or write anything to the
    generations directory. It's used by ``POST /profiles/{id}/speak``
    when the caller passes ``persist=false`` — they just want the audio
    back in the HTTP response without polluting their history.

    Loads the engine model on demand, runs ``generate_chunked`` and optional
    normalization. This legacy bytes-returning API has an explicit ten-minute
    response cap; HTTP streaming uses a private disk-backed response file and
    therefore supports the full long-form generation contract.
    """
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
    )
    from ..backends.mlx_tts_lifecycle import (
        loaded_tts_backend_for_request,
        run_blocking_operation_cancellation_safe,
        run_tts_operation_cancellation_safe,
    )
    from ..utils.audio import has_tts_runaway, normalize_audio, trim_tts_output
    from ..utils.chunked_tts import generate_chunked, release_disk_backed_audio
    from . import tts

    bg_db = next(get_db())
    audio = None
    try:
        async with loaded_tts_backend_for_request(engine, model_size) as tts_model:
            voice_prompt = await run_tts_operation_cancellation_safe(
                tts_model,
                profiles.create_voice_prompt_for_profile(
                    profile_id,
                    bg_db,
                    use_cache=True,
                    engine=engine,
                ),
            )

            trim_fn = trim_tts_output if engine_needs_trim(engine) else None
            runaway_detector = has_tts_runaway if engine_retries_runaway(engine) else None

            gen_kwargs: dict = dict(
                language=language,
                seed=seed,
                instruct=instruct,
                trim_fn=trim_fn,
                runaway_detector=runaway_detector,
            )
            if max_chunk_chars is not None:
                gen_kwargs["max_chunk_chars"] = max_chunk_chars
            if crossfade_ms is not None:
                gen_kwargs["crossfade_ms"] = crossfade_ms

            audio, sample_rate = await generate_chunked(
                tts_model,
                text,
                voice_prompt,
                **gen_kwargs,
            )
        if len(audio) > sample_rate * SYNC_AUDIO_MAX_DURATION_SECONDS:
            raise ValueError(
                "In-memory audio responses are limited to "
                f"{SYNC_AUDIO_MAX_DURATION_SECONDS // 60} minutes; use the streaming endpoint"
            )
        if normalize:
            audio = await run_blocking_operation_cancellation_safe(normalize_audio, audio)
        return await run_blocking_operation_cancellation_safe(
            tts.audio_to_wav_bytes,
            audio,
            sample_rate,
        )
    finally:
        release_disk_backed_audio(audio)
        bg_db.close()


async def _save_regenerate(
    *,
    generation_id: str,
    version_id: str | None,
    audio,
    sample_rate: int,
    db,
) -> str:
    """Save regeneration output as a new version with auto-label.

    Returns the audio path.
    """
    import uuid as _uuid

    from ..database import GenerationVersion as DBGenerationVersion
    from . import versions as versions_mod

    suffix = _uuid.uuid4().hex[:8]
    audio_path = config.get_generations_dir() / f"{generation_id}_{suffix}.wav"
    publication = await _publish_generation_audio(
        target=audio_path,
        audio=audio,
        sample_rate=sample_rate,
        db=db,
    )

    # Count via DB query rather than list length to avoid TOCTOU race
    try:
        count = db.query(DBGenerationVersion).filter_by(generation_id=generation_id).count()
        label = f"take-{count + 1}"
        versions_mod.create_version(
            generation_id=generation_id,
            label=label,
            audio_path=publication.storage_path,
            db=db,
            effects_chain=None,
            is_default=True,
        )
    except BaseException:
        _rollback_and_reconcile_generation_audio(publication, db)
        raise
    _finish_generation_audio_publication(publication)

    return publication.storage_path
