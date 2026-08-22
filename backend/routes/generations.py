"""TTS generation endpoints."""

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import stat
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import config, models
from ..backends.mlx_tts_lifecycle import run_blocking_operation_cancellation_safe
from ..database import Generation as DBGeneration, VoiceProfile as DBVoiceProfile, get_db
from ..services import deletion_journal, effects_processing, history, personality, profiles
from ..services.generation import (
    ExactBatchGenerationSpec,
    run_exact_generation_batch,
    run_generation,
)
from ..services.task_queue import (
    GenerationQueueFullError,
    cancel_generation as cancel_generation_job,
    enqueue_generation,
    enqueue_generation_batch,
    generation_job_ids,
    generation_job_is_active,
    run_queued_generation,
)
from ..utils.audio_metadata import (
    PORTABLE_AUDIO_MAX_CHANNELS,
    PORTABLE_AUDIO_MAX_DURATION_SECONDS,
    PORTABLE_AUDIO_MAX_SAMPLE_RATE,
    probe_audio_metadata,
)
from ..utils.responses import CleanupFileResponse
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)

router = APIRouter()

IMPORTED_AUDIO_PROFILE_NAME = "Imported Audio"
IMPORT_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
IMPORT_AUDIO_MAX_BYTES = 200 * 1024 * 1024  # 200 MB


async def _enqueue_generation_or_restore(
    *,
    generation_id: str,
    generation_coro,
    task_manager,
    db: Session,
    restore_status: str,
    restore_error: str | None,
) -> None:
    """Close rejected work and restore a durable non-phantom row state."""
    try:
        enqueue_generation(generation_id, generation_coro)
        return
    except BaseException as enqueue_error:
        generation_coro.close()
        task_manager.complete_generation(generation_id)
        try:
            await history.update_generation_status(
                generation_id,
                restore_status,
                db,
                error=restore_error,
                clear_error=restore_error is None,
            )
        except BaseException:
            logger.exception("Could not restore a generation row after queue rejection")
        if isinstance(enqueue_error, ValueError):
            raise HTTPException(
                status_code=409,
                detail="Generation work with this ID is still draining; retry after it leaves the queue",
            ) from enqueue_error
        if isinstance(enqueue_error, GenerationQueueFullError):
            raise HTTPException(
                status_code=503,
                detail="Generation queue is full; retry after queued work finishes",
            ) from enqueue_error
        raise


def _fsync_directory(path: Path) -> None:
    """Flush a directory where supported; keep Windows path operations usable."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except (NotImplementedError, OSError):
        if os.name == "nt":
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def _get_or_create_import_profile(db: Session) -> DBVoiceProfile:
    """Singleton profile every imported audio clip points at — keeps the
    Generation FK happy without making profile_id nullable across the schema."""
    row = db.query(DBVoiceProfile).filter(DBVoiceProfile.name == IMPORTED_AUDIO_PROFILE_NAME).first()
    if row is not None:
        return row
    row = DBVoiceProfile(
        id=str(uuid.uuid4()),
        name=IMPORTED_AUDIO_PROFILE_NAME,
        description="External audio imported into a story timeline.",
        language="en",
        voice_type="import",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _durable_import_audio_response(
    db: Session,
    *,
    generation_id: str,
    expected_fields: dict[str, object],
    publish_intent: deletion_journal.DeletionIntent,
) -> models.GenerationResponse | None:
    """Return a direct import only after independently proving row and inode."""
    with deletion_journal.durable_reconciliation_session(db) as durable_db:
        row = durable_db.query(DBGeneration).filter_by(id=generation_id).one_or_none()
        if row is None:
            return None
        if any(getattr(row, field) != value for field, value in expected_fields.items()):
            raise RuntimeError("Imported generation ID is owned by different durable data")
        target_stat = deletion_journal.managed_entry_stat(publish_intent.original)
        if (
            target_stat is None
            or not deletion_journal.entry_matches_intent(publish_intent, target_stat)
            or not stat.S_ISREG(target_stat.st_mode)
            or target_stat.st_nlink != 1
        ):
            raise RuntimeError("Imported generation audio identity changed after commit")
        return models.GenerationResponse.model_validate(row)


def _finish_import_audio_intent(
    intent: deletion_journal.DeletionIntent,
) -> bool:
    """Retire a committed import journal without revoking durable success."""
    try:
        deletion_journal.finish_deletion_intent(intent)
    except Exception:
        logger.warning(
            "Deferred cleanup of a committed direct-import intent",
            exc_info=True,
        )
        return False
    return True


def _resolve_generation_engine(data: models.GenerationRequest, profile) -> str:
    return data.engine or getattr(profile, "default_engine", None) or getattr(profile, "preset_engine", None) or "qwen"


def _require_tts_implementation_revision(
    data: models.GenerationRequest,
    *,
    required: bool = False,
    engine: str | None = None,
    model_size: str | None = None,
) -> None:
    """Reject a request whose frozen TTS implementation is not this server."""
    expected = data.tts_implementation_revision
    if expected is None:
        if required:
            raise HTTPException(
                status_code=422,
                detail="tts_implementation_revision is required for exact generation",
            )
        return

    from ..backends import get_tts_implementation_revision

    actual = get_tts_implementation_revision()
    if actual != expected:
        raise HTTPException(
            status_code=409,
            detail=(
                f"TTS implementation revision mismatch: requested {expected!r}, "
                f"running {actual!r}; restart the matching Voicebox backend"
            ),
        )
    if engine is None:
        return
    if engine != "qwen":
        raise HTTPException(
            status_code=422,
            detail="tts_implementation_revision is only valid for the pinned Qwen MLX engine",
        )
    from ..backends.mlx_runtime import get_mlx_qwen_tts_model_spec

    try:
        get_mlx_qwen_tts_model_spec(model_size or "1.7B")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _require_exact_seed(data: models.GenerationRequest) -> None:
    """Exact generation must never hide a random seed behind a stable contract."""
    if data.seed is None:
        raise HTTPException(
            status_code=422,
            detail="An explicit seed is required for exact generation",
        )


async def _generate_speech_impl(
    data: models.GenerationRequest,
    db: Session,
    *,
    exact: bool,
):
    """Generate speech from text using a voice profile."""
    _require_tts_implementation_revision(data, required=exact)
    if exact:
        _require_exact_seed(data)
    task_manager = get_task_manager()
    generation_id = str(uuid.uuid4())

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    from ..backends import engine_has_model_sizes

    engine = _resolve_generation_engine(data, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    model_size = (data.model_size or "1.7B") if engine_has_model_sizes(engine) else None
    _require_tts_implementation_revision(data, engine=engine, model_size=model_size)

    voice_binding_sha256 = None
    exact_voice_snapshot = None
    if exact:
        if (getattr(profile, "voice_type", None) or "cloned") != "cloned":
            raise HTTPException(
                status_code=422,
                detail="Exact pinned Qwen generation requires a cloned profile with reference audio",
            )
        if data.personality:
            raise HTTPException(
                status_code=422,
                detail="Exact generation cannot rewrite text with personality mode",
            )
        try:
            exact_voice_snapshot = profiles.freeze_exact_voice_profile(
                data.profile_id,
                db,
                engine=engine,
            )
            voice_binding_sha256 = exact_voice_snapshot["voice_binding_sha256"]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    effects_chain_config = None
    if data.effects_chain is not None:
        effects_chain_config = [effect.model_dump() for effect in data.effects_chain]
    else:
        profile_obj = db.query(DBVoiceProfile).filter_by(id=data.profile_id).first()
        if profile_obj and profile_obj.effects_chain:
            with contextlib.suppress(Exception):
                effects_chain_config = json.loads(profile_obj.effects_chain)
    exact_effects_json = _canonical_exact_effects_json(effects_chain_config) if exact else None
    exact_voice_snapshot_json = (
        _canonical_exact_voice_snapshot_json(exact_voice_snapshot) if exact_voice_snapshot is not None else None
    )
    exact_request_sha256 = _exact_request_sha256(data, voice_binding_sha256, effects_chain_config) if exact else None
    exact_envelope_sha256 = (
        _exact_envelope_sha256(
            [generation_id],
            [exact_request_sha256],
        )
        if exact_request_sha256
        else None
    )

    text = data.text
    source = "manual"
    profile_personality = getattr(profile, "personality", None)
    if data.personality and profile_personality:
        # Personality inference is serialized with every other local model
        # operation. Release the profile read transaction before waiting, then
        # let create_generation acquire a fresh short transaction afterward.
        db.close()
        try:
            llm_result = await personality.rewrite_as_profile(profile_personality, data.text)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        text = llm_result.text.strip()
        if not text:
            raise HTTPException(status_code=500, detail="LLM produced empty output; nothing to speak.")
        source = "personality_speak"

    generation = await history.create_generation(
        profile_id=data.profile_id,
        text=text,
        language=data.language,
        audio_path="",
        duration=0,
        seed=data.seed,
        db=db,
        instruct=data.instruct,
        generation_id=generation_id,
        status="generating",
        engine=engine,
        model_size=model_size if engine_has_model_sizes(engine) else None,
        source=source,
        exact_request_sha256=exact_request_sha256,
        exact_envelope_sha256=exact_envelope_sha256,
        exact_effects_json=exact_effects_json,
        exact_voice_snapshot_json=exact_voice_snapshot_json,
        voice_binding_sha256=voice_binding_sha256,
    )

    task_manager.start_generation(
        task_id=generation_id,
        profile_id=data.profile_id,
        text=text,
    )

    generation_coro = run_generation(
        generation_id=generation_id,
        profile_id=data.profile_id,
        text=text,
        language=data.language,
        engine=engine,
        model_size=model_size,
        seed=data.seed,
        normalize=data.normalize,
        effects_chain=effects_chain_config,
        instruct=data.instruct,
        mode="generate",
        max_chunk_chars=data.max_chunk_chars,
        crossfade_ms=data.crossfade_ms,
        expected_voice_binding_sha256=voice_binding_sha256,
        exact_voice_snapshot=exact_voice_snapshot,
        expected_tts_implementation_revision=(data.tts_implementation_revision if exact else None),
        exact_request_sha256=exact_request_sha256,
    )
    await _enqueue_generation_or_restore(
        generation_id=generation_id,
        generation_coro=generation_coro,
        task_manager=task_manager,
        db=db,
        restore_status="failed",
        restore_error="Generation queue rejected the request",
    )

    return generation


@router.post("/generate", response_model=models.GenerationResponse)
async def generate_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    return await _generate_speech_impl(data, db, exact=False)


@router.post("/generate/exact", response_model=models.GenerationResponse)
async def generate_speech_exact(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Generate only when this server proves the caller's frozen TTS revision."""
    _require_tts_implementation_revision(data, required=True)
    _require_exact_seed(data)
    return await _generate_speech_impl(data, db, exact=True)


def _matching_existing_batch(
    data: models.ExactBatchGenerationRequest,
    db: Session,
) -> list[models.GenerationResponse] | None:
    """Return an already accepted caller-ID batch, or reject an ID collision."""
    generation_ids = [str(item.generation_id) for item in data.items]
    rows = db.query(DBGeneration).filter(DBGeneration.id.in_(generation_ids)).all()
    if not rows:
        return None
    if len(rows) != len(generation_ids):
        raise HTTPException(
            status_code=409,
            detail="Exact batch IDs partially exist; use a new deterministic attempt ID pair",
        )

    rows_by_id = {row.id: row for row in rows}
    ordered_rows = [rows_by_id[generation_id] for generation_id in generation_ids]
    resolved_effects = []
    resolved_snapshots = []
    expected_request_hashes = []
    for item, row in zip(data.items, ordered_rows, strict=True):
        try:
            effects = json.loads(row.exact_effects_json)
        except (TypeError, json.JSONDecodeError):
            raise HTTPException(
                status_code=409,
                detail=(f"Exact batch generation ID {item.generation_id} has no frozen effects contract"),
            ) from None
        if effects is not None and not isinstance(effects, list):
            raise HTTPException(
                status_code=409,
                detail=f"Exact batch generation ID {item.generation_id} has invalid frozen effects",
            )
        resolved_effects.append(effects)
        try:
            snapshot = json.loads(row.exact_voice_snapshot_json)
        except (TypeError, json.JSONDecodeError):
            raise HTTPException(
                status_code=409,
                detail=(f"Exact batch generation ID {item.generation_id} has no frozen voice snapshot"),
            ) from None
        if not isinstance(snapshot, dict) or snapshot.get("voice_binding_sha256") != row.voice_binding_sha256:
            raise HTTPException(
                status_code=409,
                detail=f"Exact batch generation ID {item.generation_id} has invalid voice snapshot",
            )
        resolved_snapshots.append(snapshot)
        expected_request_hashes.append(
            _exact_request_sha256(
                item.request,
                row.voice_binding_sha256,
                effects,
            )
        )
    expected_envelope_sha256 = _exact_envelope_sha256(
        generation_ids,
        expected_request_hashes,
    )
    for item, row, expected_request_sha256, effects, snapshot in zip(
        data.items,
        ordered_rows,
        expected_request_hashes,
        resolved_effects,
        resolved_snapshots,
        strict=True,
    ):
        request = item.request
        expected = (
            request.profile_id,
            request.text,
            request.language,
            request.seed,
            request.instruct,
            request.engine or "qwen",
            request.model_size or "1.7B",
            expected_request_sha256,
            expected_envelope_sha256,
            _canonical_exact_effects_json(effects),
            _canonical_exact_voice_snapshot_json(snapshot),
        )
        actual = (
            row.profile_id,
            row.text,
            row.language,
            row.seed,
            row.instruct,
            row.engine,
            row.model_size,
            row.exact_request_sha256,
            row.exact_envelope_sha256,
            row.exact_effects_json,
            row.exact_voice_snapshot_json,
        )
        if actual != expected:
            raise HTTPException(
                status_code=409,
                detail=f"Exact batch generation ID {item.generation_id} belongs to another request",
            )
    return [models.GenerationResponse.model_validate(row) for row in ordered_rows]


def _fail_unowned_exact_batch(
    existing: list[models.GenerationResponse],
    db: Session,
    *,
    error: str,
) -> list[models.GenerationResponse]:
    """Make a durably accepted exact envelope terminal if no queue owns it."""
    active_statuses = {"generating", "loading_model"}
    active_ids = [row.id for row in existing if row.status in active_statuses]
    if not active_ids or any(generation_job_is_active(row.id) for row in existing):
        return existing

    rows = db.query(DBGeneration).filter(DBGeneration.id.in_(active_ids)).all()
    for row in rows:
        if row.status in active_statuses:
            row.status = "failed"
            row.error = error
    db.commit()
    for generation_id in active_ids:
        get_task_manager().complete_generation(generation_id)
    refreshed = db.query(DBGeneration).filter(DBGeneration.id.in_([row.id for row in existing])).all()
    refreshed_by_id = {row.id: row for row in refreshed}
    return [models.GenerationResponse.model_validate(refreshed_by_id[row.id]) for row in existing]


def _exact_request_sha256(
    request: models.GenerationRequest,
    voice_binding_sha256: str | None,
    resolved_effects_chain: list | None,
) -> str:
    """Canonical numerical identity for exact checkpoint reuse.

    ``profile_id`` owns the API/database row, but a recreated profile UUID must
    not invalidate completed inference when its immutable voice binding and
    every numerical request parameter are unchanged.
    """
    payload = json.dumps(
        {
            "request": request.model_dump(mode="json", exclude={"profile_id"}),
            "voice_binding_sha256": voice_binding_sha256,
            "resolved_effects_chain": resolved_effects_chain,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_exact_effects_json(effects_chain: list | None) -> str:
    return json.dumps(
        effects_chain,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_exact_voice_snapshot_json(snapshot: dict) -> str:
    return json.dumps(
        snapshot,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _exact_envelope_sha256(
    generation_ids: list[str],
    exact_request_sha256s: list[str],
) -> str:
    """Bind idempotent rows to their ordered serial-or-batch execution shape."""
    if len(generation_ids) != len(exact_request_sha256s):
        raise ValueError("Exact envelope IDs and requests must have equal length")
    if len(generation_ids) not in (1, 2):
        raise ValueError("Exact envelope supports one or two generation units")
    payload = json.dumps(
        {
            "algorithm": ("serial-chunked-v1" if len(generation_ids) == 1 else "qwen-model-batch2-v1"),
            "items": [
                {
                    "index": index,
                    "generation_id": generation_id,
                    "exact_request_sha256": request_sha256,
                }
                for index, (generation_id, request_sha256) in enumerate(
                    zip(generation_ids, exact_request_sha256s, strict=True)
                )
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@router.post(
    "/generate/batch/exact",
    response_model=list[models.GenerationResponse],
)
async def generate_speech_batch_exact(
    data: models.ExactBatchGenerationRequest,
    db: Session = Depends(get_db),
):
    """Accept one idempotent unit, or batch two units in one Qwen model call."""
    generation_ids = [str(item.generation_id) for item in data.items]
    if len(set(generation_ids)) != len(generation_ids):
        raise HTTPException(status_code=422, detail="Exact batch generation IDs must be unique")

    # Reject a stale/mismatched process before any profile or history mutation.
    for item in data.items:
        _require_tts_implementation_revision(item.request, required=True)
        _require_exact_seed(item.request)

    existing = _matching_existing_batch(data, db)
    if existing is not None:
        return _fail_unowned_exact_batch(
            existing,
            db,
            error="Exact generation was accepted without a live queue owner",
        )

    requests = [item.request for item in data.items]
    if len(requests) == 2 and any(request.instruct is not None for request in requests):
        raise HTTPException(
            status_code=422,
            detail="Two-item exact generation does not support instruction-conditioned batching",
        )
    profile_ids = {request.profile_id for request in requests}
    if len(profile_ids) != 1:
        raise HTTPException(status_code=422, detail="Exact batch units must use one voice profile")
    if any(request.personality for request in requests):
        raise HTTPException(
            status_code=422,
            detail="Exact batch generation cannot rewrite audiobook text with personality mode",
        )

    profile_id = requests[0].profile_id
    profile = await profiles.get_profile(profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    if (getattr(profile, "voice_type", None) or "cloned") != "cloned":
        raise HTTPException(
            status_code=422,
            detail="Exact pinned Qwen generation requires a cloned profile with reference audio",
        )
    from ..backends import engine_has_model_sizes
    from ..utils.chunked_tts import split_text_into_chunks

    engines = [_resolve_generation_engine(request, profile) for request in requests]
    for request, engine in zip(requests, engines, strict=True):
        try:
            profiles.validate_profile_engine(profile, engine)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        model_size = (request.model_size or "1.7B") if engine_has_model_sizes(engine) else None
        _require_tts_implementation_revision(
            request,
            engine=engine,
            model_size=model_size,
        )

    model_sizes = [
        (request.model_size or "1.7B") if engine_has_model_sizes(engine) else None
        for request, engine in zip(requests, engines, strict=True)
    ]
    common_contracts = {
        (
            request.profile_id,
            request.language,
            engine,
            model_size,
            request.instruct,
            request.max_chunk_chars,
            request.crossfade_ms,
        )
        for request, engine, model_size in zip(requests, engines, model_sizes, strict=True)
    }
    if len(common_contracts) != 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "Exact batch units must share profile, language, engine, model, instruction, chunk size, and crossfade"
            ),
        )
    try:
        exact_voice_snapshot = profiles.freeze_exact_voice_profile(
            profile_id,
            db,
            engine=engines[0],
        )
        voice_binding_sha256 = exact_voice_snapshot["voice_binding_sha256"]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    exact_voice_snapshot_json = _canonical_exact_voice_snapshot_json(exact_voice_snapshot)
    if len(requests) == 2 and any(
        len(split_text_into_chunks(request.text, request.max_chunk_chars)) != 1 for request in requests
    ):
        raise HTTPException(
            status_code=422,
            detail="Each exact batch unit must fit in one model chunk",
        )

    profile_row = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    default_effects = None
    if profile_row and profile_row.effects_chain:
        import json as _json

        with contextlib.suppress(Exception):
            default_effects = _json.loads(profile_row.effects_chain)

    effects_configs = [
        [effect.model_dump() for effect in request.effects_chain]
        if request.effects_chain is not None
        else default_effects
        for request in requests
    ]
    exact_request_hashes = [
        _exact_request_sha256(request, voice_binding_sha256, effects)
        for request, effects in zip(requests, effects_configs, strict=True)
    ]
    exact_envelope_sha256 = _exact_envelope_sha256(
        generation_ids,
        exact_request_hashes,
    )
    entries = [
        {
            "id": generation_id,
            "profile_id": request.profile_id,
            "text": request.text,
            "language": request.language,
            "audio_path": "",
            "duration": 0,
            "seed": request.seed,
            "instruct": request.instruct,
            "engine": engine,
            "model_size": model_size,
            "status": "generating",
            "source": "manual",
            "exact_request_sha256": exact_request_sha256,
            "exact_envelope_sha256": exact_envelope_sha256,
            "exact_effects_json": _canonical_exact_effects_json(effects),
            "exact_voice_snapshot_json": exact_voice_snapshot_json,
            "voice_binding_sha256": voice_binding_sha256,
        }
        for generation_id, request, engine, model_size, exact_request_sha256, effects in zip(
            generation_ids,
            requests,
            engines,
            model_sizes,
            exact_request_hashes,
            effects_configs,
            strict=True,
        )
    ]
    try:
        responses = await history.create_generation_batch(entries, db)
    except IntegrityError:
        # Two identical retries can arrive together after a client loses the POST response.
        # The unique caller IDs make one transaction win; the loser reattaches those exact rows.
        existing = _matching_existing_batch(data, db)
        if existing is not None:
            return existing
        raise HTTPException(
            status_code=409,
            detail="Exact batch caller IDs raced with another request",
        ) from None
    except BaseException:
        # SQLite (and a failing connection/proxy) can make COMMIT's outcome
        # ambiguous: rows may be durable even though commit() raised. This
        # request never reached enqueue, so independently force matching active
        # rows terminal. A retry can then advance to its next deterministic
        # attempt instead of polling a phantom generation for the stall timeout.
        try:
            with deletion_journal.durable_reconciliation_session(db) as durable_db:
                durable_existing = _matching_existing_batch(data, durable_db)
                if durable_existing is not None:
                    _fail_unowned_exact_batch(
                        durable_existing,
                        durable_db,
                        error="Exact generation acceptance failed before queue ownership",
                    )
        except BaseException:
            logger.exception("Could not reconcile ambiguous exact batch acceptance")
        raise

    task_manager = get_task_manager()
    for generation_id, request in zip(generation_ids, requests, strict=True):
        task_manager.start_generation(
            task_id=generation_id,
            profile_id=request.profile_id,
            text=request.text,
        )

    specs = [
        ExactBatchGenerationSpec(
            generation_id=generation_id,
            profile_id=request.profile_id,
            text=request.text,
            language=request.language,
            engine=engine,
            model_size=model_size or "1.7B",
            seed=request.seed,
            normalize=request.normalize,
            effects_chain=effects,
            instruct=request.instruct,
            crossfade_ms=request.crossfade_ms,
            expected_voice_binding_sha256=voice_binding_sha256,
            exact_voice_snapshot=exact_voice_snapshot,
            expected_tts_implementation_revision=request.tts_implementation_revision,
        )
        for generation_id, request, engine, model_size, effects in zip(
            generation_ids,
            requests,
            engines,
            model_sizes,
            effects_configs,
            strict=True,
        )
    ]
    if len(specs) == 2:
        generation_coro = run_exact_generation_batch(specs)
    else:
        spec = specs[0]
        generation_coro = run_generation(
            generation_id=spec.generation_id,
            profile_id=spec.profile_id,
            text=spec.text,
            language=spec.language,
            engine=spec.engine,
            model_size=spec.model_size,
            seed=spec.seed,
            normalize=spec.normalize,
            effects_chain=spec.effects_chain,
            instruct=spec.instruct,
            mode="generate",
            max_chunk_chars=requests[0].max_chunk_chars,
            crossfade_ms=spec.crossfade_ms,
            expected_voice_binding_sha256=spec.expected_voice_binding_sha256,
            exact_voice_snapshot=spec.exact_voice_snapshot,
            expected_tts_implementation_revision=(spec.expected_tts_implementation_revision),
            exact_request_sha256=exact_request_hashes[0],
        )
    try:
        enqueue_generation_batch(tuple(generation_ids), generation_coro)
    except BaseException as enqueue_error:
        generation_coro.close()
        for generation_id in generation_ids:
            task_manager.complete_generation(generation_id)
            await history.update_generation_status(
                generation_id,
                "failed",
                db,
                error="Generation queue rejected exact batch",
            )
        if isinstance(enqueue_error, GenerationQueueFullError):
            raise HTTPException(
                status_code=503,
                detail="Generation queue is full; retry after queued work finishes",
            ) from enqueue_error
        raise
    return responses


@router.post("/generate/{generation_id}/retry", response_model=models.GenerationResponse)
async def retry_generation(generation_id: str, db: Session = Depends(get_db)):
    """Retry a failed generation using the same parameters."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if gen.exact_request_sha256:
        raise HTTPException(
            status_code=409,
            detail=(
                "Exact generations cannot use generic retry; submit the frozen request "
                "with a new deterministic attempt ID"
            ),
        )

    if generation_job_is_active(generation_id):
        raise HTTPException(
            status_code=409,
            detail="Cancelled generation work is still draining; retry after it leaves the queue",
        )

    if (gen.status or "completed") != "failed":
        raise HTTPException(status_code=400, detail="Only failed generations can be retried")

    previous_error = gen.error
    # Keep any previously committed artifact owned until the replacement WAV
    # is durably published. Cancellation or a backend crash before that point
    # must not orphan an otherwise valid retry result.
    accepted = await history.update_generation_status(
        generation_id,
        "generating",
        db,
        clear_error=True,
    )
    if accepted is None:
        raise HTTPException(status_code=404, detail="Generation not found")

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=accepted.profile_id,
        text=accepted.text,
    )

    generation_coro = run_generation(
        generation_id=generation_id,
        profile_id=accepted.profile_id,
        text=accepted.text,
        language=accepted.language,
        engine=accepted.engine or "qwen",
        model_size=accepted.model_size or "1.7B",
        seed=accepted.seed,
        instruct=accepted.instruct,
        mode="retry",
    )
    await _enqueue_generation_or_restore(
        generation_id=generation_id,
        generation_coro=generation_coro,
        task_manager=task_manager,
        db=db,
        restore_status="failed",
        restore_error=previous_error,
    )

    return accepted


@router.post(
    "/generate/{generation_id}/regenerate",
    response_model=models.GenerationResponse,
)
async def regenerate_generation(generation_id: str, db: Session = Depends(get_db)):
    """Re-run TTS with the same parameters and save the result as a new version."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if gen.exact_request_sha256:
        raise HTTPException(
            status_code=409,
            detail=(
                "Exact generations cannot use generic regenerate; submit a new frozen "
                "request and deterministic attempt ID"
            ),
        )
    if generation_job_is_active(generation_id):
        raise HTTPException(
            status_code=409,
            detail="Generation work with this ID is still draining; retry after it leaves the queue",
        )
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation must be completed to regenerate")

    previous_error = gen.error
    accepted = await history.update_generation_status(
        generation_id,
        "generating",
        db,
        clear_error=True,
    )
    if accepted is None:
        raise HTTPException(status_code=404, detail="Generation not found")

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=accepted.profile_id,
        text=accepted.text,
    )

    version_id = str(uuid.uuid4())

    generation_coro = run_generation(
        generation_id=generation_id,
        profile_id=accepted.profile_id,
        text=accepted.text,
        language=accepted.language,
        engine=accepted.engine or "qwen",
        model_size=accepted.model_size or "1.7B",
        seed=accepted.seed,
        instruct=accepted.instruct,
        mode="regenerate",
        version_id=version_id,
    )
    await _enqueue_generation_or_restore(
        generation_id=generation_id,
        generation_coro=generation_coro,
        task_manager=task_manager,
        db=db,
        restore_status="completed",
        restore_error=previous_error,
    )

    return accepted


@router.post("/generate/{generation_id}/cancel")
async def cancel_generation(generation_id: str, db: Session = Depends(get_db)):
    """Cancel a queued or running generation."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    if (gen.status or "completed") not in ("loading_model", "generating"):
        raise HTTPException(status_code=400, detail="Only active generations can be cancelled")

    owned_generation_ids = generation_job_ids(generation_id)
    cancellation_state = cancel_generation_job(generation_id)
    if cancellation_state is None:
        # Row says active but the worker is no longer tracking it — the gen
        # coroutine exited without writing a terminal status (most often a
        # SQLite lock racing with the failed-status write inside the worker's
        # exception handler). Fail the row here so the user can move on.
        task_manager = get_task_manager()
        task_manager.complete_generation(generation_id)
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=db,
            error="Generation orphaned by worker",
        )
        return {"message": "Orphaned generation cleared"}

    if cancellation_state == "queued":
        task_manager = get_task_manager()
        for owned_generation_id in owned_generation_ids:
            task_manager.complete_generation(owned_generation_id)
            await history.update_generation_status(
                generation_id=owned_generation_id,
                status="failed",
                db=db,
                error="Generation batch cancelled",
            )
        return {"message": "Queued generation job cancelled"}

    return {"message": "Generation cancellation requested"}


@router.get("/generate/{generation_id}/status")
async def get_generation_status(generation_id: str, db: Session = Depends(get_db)):
    """SSE endpoint that streams generation status updates."""
    import json

    # Streaming-response dependencies otherwise live until disconnect. Release
    # this request's transaction before returning and use one short session per
    # poll, so many audiobook EventSources cannot exhaust SQLAlchemy's pool.
    close_request_db = getattr(db, "close", None)
    if callable(close_request_db):
        close_request_db()

    async def event_stream():
        try:
            while True:
                poll_dependency = get_db()
                poll_db = next(poll_dependency)
                try:
                    gen = poll_db.query(DBGeneration).filter_by(id=generation_id).first()
                    if gen is None:
                        payload = None
                        terminal = True
                    else:
                        status = gen.status or "completed"
                        payload = {
                            "id": gen.id,
                            "status": status,
                            "duration": gen.duration,
                            "error": gen.error,
                            # Agent-originated sources ("mcp", "rest") skip
                            # main-window autoplay; the floating pill owns it.
                            "source": gen.source,
                        }
                        terminal = status in ("completed", "failed")
                finally:
                    poll_dependency.close()

                if payload is None:
                    yield f"data: {json.dumps({'status': 'not_found', 'id': generation_id})}\n\n"
                    return

                yield f"data: {json.dumps(payload)}\n\n"

                if terminal:
                    return

                await asyncio.sleep(1)
        except (BrokenPipeError, ConnectionResetError, asyncio.CancelledError):
            logger.debug("SSE client disconnected for generation %s", generation_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_speech_impl(
    data: models.GenerationRequest,
    db: Session,
    *,
    exact: bool,
):
    """Generate speech and stream the WAV audio directly without saving to disk."""
    _require_tts_implementation_revision(data, required=exact)
    if exact:
        _require_exact_seed(data)
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        ensure_model_cached_or_raise,
    )
    from ..backends.mlx_tts_lifecycle import (
        loaded_tts_backend_for_request,
        run_tts_operation_cancellation_safe,
    )
    from ..utils.chunked_tts import (
        GeneratedAudioEmptyError,
        GeneratedAudioLimitError,
        GeneratedAudioStorageError,
        release_disk_backed_audio,
    )

    profile = await profiles.get_profile(data.profile_id, db)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    effects_chain_config = None
    if data.effects_chain is not None:
        effects_chain_config = [effect.model_dump() for effect in data.effects_chain]
    elif profile.effects_chain:
        import json as _json

        try:
            effects_chain_config = _json.loads(profile.effects_chain)
        except Exception:
            effects_chain_config = None
    if effects_chain_config:
        from ..utils.effects import validate_effects_chain

        effects_error = validate_effects_chain(effects_chain_config)
        if effects_error:
            raise HTTPException(status_code=400, detail=effects_error)

    voice_binding_sha256 = None
    exact_voice_snapshot = None
    if exact:
        if (getattr(profile, "voice_type", None) or "cloned") != "cloned":
            raise HTTPException(
                status_code=422,
                detail="Exact pinned Qwen generation requires a cloned profile with reference audio",
            )
        if data.personality:
            raise HTTPException(
                status_code=422,
                detail="Exact generation cannot rewrite text with personality mode",
            )
    engine = _resolve_generation_engine(data, profile)
    try:
        profiles.validate_profile_engine(profile, engine)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    model_size = data.model_size or "1.7B"
    _require_tts_implementation_revision(data, engine=engine, model_size=model_size)
    if exact:
        try:
            exact_voice_snapshot = profiles.freeze_exact_voice_profile(
                data.profile_id,
                db,
                engine=engine,
            )
            voice_binding_sha256 = exact_voice_snapshot["voice_binding_sha256"]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # A foreground request may wait behind a long durable audiobook job. End
    # its dependency-session transaction before queueing so the bounded
    # waiter set does not retain SQLite connections or read transactions.
    close_request_db = getattr(db, "close", None)
    if callable(close_request_db):
        close_request_db()

    async def _generate_stream_file():
        audio = None
        try:
            await ensure_model_cached_or_raise(engine, model_size)
            async with loaded_tts_backend_for_request(engine, model_size) as tts_model:
                if voice_binding_sha256 is not None:
                    try:
                        voice_prompt = await run_tts_operation_cancellation_safe(
                            tts_model,
                            profiles.create_exact_voice_prompt_from_snapshot(
                                exact_voice_snapshot,
                                expected_voice_binding_sha256=voice_binding_sha256,
                                expected_tts_implementation_revision=data.tts_implementation_revision,
                                engine=engine,
                            ),
                        )
                    except (RuntimeError, ValueError) as exc:
                        raise HTTPException(status_code=409, detail=str(exc)) from exc
                else:
                    prompt_db_dependency = get_db()
                    prompt_db = next(prompt_db_dependency)
                    try:
                        voice_prompt = await run_tts_operation_cancellation_safe(
                            tts_model,
                            profiles.create_voice_prompt_for_profile(
                                data.profile_id,
                                prompt_db,
                                engine=engine,
                            ),
                        )
                    finally:
                        prompt_db_dependency.close()

                from ..utils.chunked_tts import generate_chunked

                trim_fn = None
                runaway_detector = None
                if engine_needs_trim(engine):
                    from ..utils.audio import trim_tts_output

                    trim_fn = trim_tts_output
                if engine_retries_runaway(engine):
                    from ..utils.audio import has_tts_runaway

                    runaway_detector = has_tts_runaway

                audio, sample_rate = await generate_chunked(
                    tts_model,
                    data.text,
                    voice_prompt,
                    language=data.language,
                    seed=data.seed,
                    instruct=data.instruct,
                    max_chunk_chars=data.max_chunk_chars,
                    crossfade_ms=data.crossfade_ms,
                    trim_fn=trim_fn,
                    runaway_detector=runaway_detector,
                )

            return await effects_processing.create_generated_audio_response_file(
                audio,
                sample_rate,
                effects_chain_config or [],
                data.normalize,
            )
        finally:
            release_disk_backed_audio(audio)

    generated_file = None
    handed_to_response = False
    try:
        generated_file = await run_queued_generation(
            f"stream-{uuid.uuid4()}",
            _generate_stream_file(),
            discard_result=lambda value: value.cleanup(),
        )
        response = CleanupFileResponse(
            generated_file.path,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="speech.wav"',
                "Cache-Control": "no-cache, no-store",
            },
            cleanup=generated_file.cleanup,
        )
        handed_to_response = True
        return response
    except GenerationQueueFullError as exc:
        raise HTTPException(
            status_code=503,
            detail="Generation queue is full; retry after queued work finishes",
            headers={"Retry-After": "1"},
        ) from exc
    except GeneratedAudioEmptyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GeneratedAudioLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except GeneratedAudioStorageError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    except effects_processing.EffectsProcessingLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except effects_processing.EffectsProcessingStorageError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    except effects_processing.EffectsProcessingBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
            headers={"Retry-After": "1"},
        ) from exc
    except effects_processing.EffectsProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if generated_file is not None and not handed_to_response:
            generated_file.cleanup()


@router.post("/generate/stream")
async def stream_speech(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    return await _stream_speech_impl(data, db, exact=False)


@router.post("/generate/stream/exact")
async def stream_speech_exact(
    data: models.GenerationRequest,
    db: Session = Depends(get_db),
):
    """Stream only when this server proves the caller's frozen TTS revision."""
    _require_tts_implementation_revision(data, required=True)
    _require_exact_seed(data)
    return await _stream_speech_impl(data, db, exact=True)


@router.post("/generate/import", response_model=models.GenerationResponse)
async def import_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Register an external audio file as a generation row.

    Designed for the story timeline so users can drop in music or other
    non-TTS audio. The row points at a singleton "Imported Audio" profile
    so the existing generation/story plumbing keeps working unchanged."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMPORT_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {sorted(IMPORT_AUDIO_EXTENSIONS)}",
        )

    generation_id = str(uuid.uuid4())
    generations_dir = config.get_generations_dir()
    target = generations_dir / f"{generation_id}{suffix}"
    pending = generations_dir / f".voicebox-delete-import-{generation_id}.part{suffix}"
    target_relative = Path("generations") / target.name
    pending_relative = Path("generations") / pending.name
    total = 0
    descriptor: int | None = None
    published = False
    publish_intent: deletion_journal.DeletionIntent | None = None
    expected_generation_fields: dict[str, object] | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(pending, flags, 0o600)
        publish_intent = deletion_journal.prepare_deletion_intent(
            kind=deletion_journal.GENERATION_AUDIO,
            original=target_relative,
            staged=pending_relative,
            entry_stat=os.fstat(descriptor),
            owner_id=generation_id,
        )
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > IMPORT_AUDIO_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(f"File exceeds {IMPORT_AUDIO_MAX_BYTES // (1024 * 1024)} MB limit."),
                )
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while importing audio")
                view = view[written:]
        if total == 0:
            raise HTTPException(status_code=400, detail="Empty audio file.")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None

        try:
            # Probe only bounded container metadata off the event loop. The
            # portable story/effects contract is enforced before publication.
            duration, audio_channels, audio_sample_rate = await run_blocking_operation_cancellation_safe(
                probe_audio_metadata,
                pending,
            )
        except Exception as decode_err:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode audio: {decode_err}",
            ) from decode_err
        if not math.isfinite(duration) or duration <= 0:
            raise HTTPException(status_code=400, detail="Audio duration is invalid.")
        if duration > PORTABLE_AUDIO_MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=413,
                detail=(f"Audio duration is too long (max {PORTABLE_AUDIO_MAX_DURATION_SECONDS // 3600} hours)."),
            )
        if not 1 <= audio_channels <= PORTABLE_AUDIO_MAX_CHANNELS:
            raise HTTPException(
                status_code=413,
                detail=f"Audio exceeds the {PORTABLE_AUDIO_MAX_CHANNELS}-channel limit.",
            )
        if not 1 <= audio_sample_rate <= PORTABLE_AUDIO_MAX_SAMPLE_RATE:
            raise HTTPException(
                status_code=413,
                detail=f"Audio exceeds the {PORTABLE_AUDIO_MAX_SAMPLE_RATE} Hz sample-rate limit.",
            )
        deletion_journal.rename_managed_entry(pending_relative, target_relative)
        published = True

        profile = _get_or_create_import_profile(db)
        display_name = Path(file.filename or "Imported audio").stem or "Imported audio"
        stored_audio_path = config.to_storage_path(target)
        expected_generation_fields = {
            "id": generation_id,
            "profile_id": profile.id,
            "text": display_name,
            "language": "en",
            "audio_path": stored_audio_path,
            "duration": duration,
            "seed": None,
            "instruct": None,
            "engine": "import",
            "model_size": None,
            "status": "completed",
            "error": None,
            "is_favorited": False,
            "source": "import",
            "exact_request_sha256": None,
            "exact_envelope_sha256": None,
            "exact_effects_json": None,
            "exact_voice_snapshot_json": None,
            "voice_binding_sha256": None,
        }
        result = await history.create_generation(
            profile_id=profile.id,
            text=display_name,
            language="en",
            audio_path=stored_audio_path,
            duration=duration,
            seed=None,
            db=db,
            generation_id=generation_id,
            status="completed",
            engine="import",
            model_size=None,
            source="import",
        )
        if _finish_import_audio_intent(publish_intent):
            publish_intent = None
        return result
    except BaseException as operation_error:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
        durable_response: models.GenerationResponse | None = None
        if published and publish_intent is not None and expected_generation_fields is not None:
            try:
                durable_response = _durable_import_audio_response(
                    db,
                    generation_id=generation_id,
                    expected_fields=expected_generation_fields,
                    publish_intent=publish_intent,
                )
            except Exception:
                logger.warning(
                    "Could not prove an ambiguous direct audio import",
                    exc_info=True,
                )
        if durable_response is not None:
            if _finish_import_audio_intent(publish_intent):
                publish_intent = None
            return durable_response

        keep_published: bool | None = False
        if published:
            keep_published = None
            try:
                with deletion_journal.durable_reconciliation_session(db) as owner_db:
                    row = owner_db.query(DBGeneration).filter_by(id=generation_id).one_or_none()
                    keep_published = row is not None and (
                        config.managed_storage_relative_path(row.audio_path)
                        == config.managed_storage_relative_path(target)
                    )
            except Exception:
                # An indeterminate database outcome must not delete audio that
                # may already be durably referenced by a committed row. Keep
                # its journal as well so startup can decide from a fresh DB.
                pass
        if publish_intent is not None and keep_published is False:
            try:
                target_stat = deletion_journal.managed_entry_stat(target_relative)
                pending_stat = deletion_journal.managed_entry_stat(pending_relative)
                if target_stat is not None and pending_stat is not None:
                    raise RuntimeError("Both imported audio paths exist")
                if target_stat is not None:
                    if not deletion_journal.entry_matches_intent(publish_intent, target_stat):
                        raise RuntimeError("Published imported audio identity changed")
                    deletion_journal.discard_managed_entry(target_relative)
                elif pending_stat is not None:
                    if not deletion_journal.entry_matches_intent(publish_intent, pending_stat):
                        raise RuntimeError("Pending imported audio identity changed")
                    deletion_journal.discard_managed_entry(pending_relative)
                deletion_journal.finish_deletion_intent(publish_intent)
                publish_intent = None
            except BaseException:
                # Retain the durable intent so startup can finish the cleanup.
                pass
        elif publish_intent is not None and keep_published is True:
            # A committed owner makes publication complete even when the DB
            # driver reported an outcome-ambiguous commit error.
            try:
                target_stat = deletion_journal.managed_entry_stat(target_relative)
                if (
                    target_stat is not None
                    and deletion_journal.entry_matches_intent(publish_intent, target_stat)
                    and _finish_import_audio_intent(publish_intent)
                ):
                    publish_intent = None
            except BaseException:
                # Leave the intent for startup if publication is ambiguous.
                pass
        elif not published:
            with contextlib.suppress(OSError):
                pending.unlink()
                _fsync_directory(generations_dir)
        if rollback_error is not None:
            raise rollback_error from operation_error
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if publish_intent is None:
            with contextlib.suppress(OSError):
                pending.unlink()
                _fsync_directory(generations_dir)
