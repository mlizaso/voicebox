"""Story management and bounded timeline audio export."""

import asyncio
import logging
import math
import os
import shutil
import stat
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import audioread
import numpy as np
import soundfile as sf
import soxr
from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import config
from ..database import (
    Generation as DBGeneration,
    GenerationVersion as DBGenerationVersion,
    Story as DBStory,
    StoryItem as DBStoryItem,
    VoiceProfile as DBVoiceProfile,
)
from ..models import (
    STORY_MAX_CLIP_MS,
    STORY_MAX_ITEMS,
    STORY_MAX_TIMELINE_MS,
    STORY_MAX_TRACK,
    StoryCreate,
    StoryDetailResponse,
    StoryItemBatchUpdate,
    StoryItemCreate,
    StoryItemDetail,
    StoryItemMove,
    StoryItemSplit,
    StoryItemTrim,
    StoryItemVersionUpdate,
    StoryItemVolumeUpdate,
    StoryResponse,
)
from ..utils.audio_metadata import (
    PORTABLE_AUDIO_MAX_CHANNELS,
    PORTABLE_AUDIO_MAX_SAMPLE_RATE,
)
from ..utils.disk_reservations import (
    DiskSpaceReservation,
    DiskSpaceReservationError,
    reserve_disk_space,
)
from . import deletion_journal
from .history import _get_versions_for_generation

STORY_EXPORT_SAMPLE_RATE = 24_000
STORY_EXPORT_BLOCK_FRAMES = 65_536
STORY_EXPORT_MAX_CHANNELS = PORTABLE_AUDIO_MAX_CHANNELS
STORY_EXPORT_MAX_SAMPLE_RATE = PORTABLE_AUDIO_MAX_SAMPLE_RATE
STORY_EXPORT_MAX_SOURCE_SECONDS = 48 * 60 * 60
# This admits either 48 hours of normal 48 kHz stereo media or 24 hours of
# 96 kHz stereo media while putting a hard ceiling on compressed-audio decode
# work. Generated speech is normally 24 kHz mono and is well below this bound.
STORY_EXPORT_MAX_SOURCE_SAMPLE_VALUES = 48 * 60 * 60 * 48_000 * 2
STORY_EXPORT_MIN_FREE_BYTES = 1024**3
STORY_EXPORT_MAX_OUTPUT_BYTES = 44 + (STORY_MAX_TIMELINE_MS * STORY_EXPORT_SAMPLE_RATE // 1000) * 2
STORY_EXPORT_ROOT_NAME = "story-audio-exports-v1"
STORY_EXPORT_MAX_STALE_ENTRIES = 1000

logger = logging.getLogger(__name__)


class StoryAudioExportError(ValueError):
    """A story cannot be rendered from its current audio data."""


class StoryAudioExportLimitError(StoryAudioExportError):
    """A story export exceeds a bounded resource or file-format limit."""


class StoryAudioExportBusyError(StoryAudioExportError):
    """Another story already owns the single heavy export worker."""


class _StoryAudioExportCancelledError(RuntimeError):
    """Internal cooperative-cancellation signal for the worker thread."""


@dataclass(frozen=True)
class StoryAudioExport:
    """A private rendered file that must be cleaned after the response."""

    path: Path
    temporary_directory: Path

    def cleanup(self) -> None:
        _cleanup_story_export(self.temporary_directory)


@dataclass(frozen=True)
class _StoryClip:
    path: Path
    generation_id: str
    start_time_ms: int
    trim_start_ms: int
    trim_end_ms: int
    volume: float


@dataclass(frozen=True)
class _ProbedClip:
    clip: _StoryClip
    frames: int
    sample_rate: int
    channels: int
    decoder: str
    output_frames: int
    trim_start_frames: int
    trim_end_frames: int


_story_export_lock = asyncio.Lock()
_story_export_state_lock = threading.Lock()
_active_story_export_directories: set[Path] = set()
_story_export_reservations: dict[Path, DiskSpaceReservation] = {}


def _release_story_export_reservation_locked(directory: Path) -> None:
    reservation = _story_export_reservations.pop(directory, None)
    if reservation is not None:
        reservation.release()


def _valid_timecode(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= STORY_MAX_TIMELINE_MS


def _valid_track(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= STORY_MAX_TRACK


def _story_item_count(story_id: str, db: Session) -> int:
    return db.query(func.count(DBStoryItem.id)).filter_by(story_id=story_id).scalar() or 0


def _story_export_root() -> Path:
    """Return a private, managed, non-link directory for export scratch."""
    root = config.get_cache_dir() / STORY_EXPORT_ROOT_NAME
    with suppress(FileExistsError):
        root.mkdir(mode=0o700)
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise StoryAudioExportError("Story export storage is unavailable") from exc
    is_junction = getattr(root, "is_junction", None)
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
        or (is_junction is not None and is_junction())
    ):
        raise StoryAudioExportError("Story export storage is not a real directory")
    if os.name == "posix":
        os.chmod(root, 0o700, follow_symlinks=False)
    return root


def _remove_story_export_directory(directory: Path) -> bool:
    """Remove one owned scratch entry without following a replacement link."""
    try:
        entry_stat = directory.lstat()
    except FileNotFoundError:
        return True
    is_junction = getattr(directory, "is_junction", None)
    if is_junction is not None and is_junction():
        # Windows rmdir removes the junction itself, not its target.
        directory.rmdir()
        return True
    if stat.S_ISDIR(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode):
        shutil.rmtree(directory)
        return True
    if stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode):
        directory.unlink(missing_ok=True)
        return True
    return False


def _story_export_entry_missing(directory: Path) -> bool:
    try:
        directory.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _cleanup_stale_story_exports_locked(root: Path) -> tuple[int, int, bool]:
    """Boundedly reclaim prior-process jobs while preserving live responses."""
    removed = 0
    refused = 0
    scanned = 0
    truncated = False
    try:
        entries = root.iterdir()
        for entry in entries:
            scanned += 1
            if scanned > STORY_EXPORT_MAX_STALE_ENTRIES:
                truncated = True
                break
            if not entry.name.startswith("job-") or entry in _active_story_export_directories:
                continue
            try:
                if _remove_story_export_directory(entry):
                    removed += 1
                    _release_story_export_reservation_locked(entry)
                else:
                    refused += 1
            except OSError:
                if _story_export_entry_missing(entry):
                    _release_story_export_reservation_locked(entry)
                refused += 1
    except OSError as exc:
        raise StoryAudioExportError("Could not inspect story export storage") from exc
    return removed, refused, truncated


def cleanup_abandoned_story_audio_exports() -> tuple[int, int, bool]:
    """Startup-safe cleanup for crash-abandoned private Story exports."""
    with _story_export_state_lock:
        return _cleanup_stale_story_exports_locked(_story_export_root())


def _allocate_story_export_directory() -> Path:
    with _story_export_state_lock:
        root = _story_export_root()
        removed, refused, truncated = _cleanup_stale_story_exports_locked(root)
        if removed:
            logger.info("Removed %d abandoned Story audio export(s)", removed)
        if refused or truncated:
            logger.warning(
                "Retained unsafe or excess Story export scratch (refused=%d, truncated=%s)",
                refused,
                truncated,
            )
        temporary_directory = Path(tempfile.mkdtemp(prefix="job-", dir=root))
        _active_story_export_directories.add(temporary_directory)
        return temporary_directory


def _cleanup_story_export(directory: Path) -> None:
    with _story_export_state_lock:
        _active_story_export_directories.discard(directory)
        removed = False
        try:
            removed = _remove_story_export_directory(directory)
            if not removed:
                logger.warning("Refused to remove an unsafe Story audio export entry")
        except OSError:
            if _story_export_entry_missing(directory):
                _release_story_export_reservation_locked(directory)
            logger.warning("Could not remove a private Story audio export")
        if removed:
            _release_story_export_reservation_locked(directory)


def _build_item_detail(
    item: DBStoryItem,
    generation: DBGeneration,
    profile_name: str,
    db: Session,
) -> StoryItemDetail:
    """Build a StoryItemDetail with version info from a story item and its generation."""
    versions, active_version_id = _get_versions_for_generation(generation.id, db)

    # Resolve the audio path: if version_id is set, use that version's audio
    audio_path = generation.audio_path
    if item.version_id and versions:
        for v in versions:
            if v.id == item.version_id:
                audio_path = v.audio_path
                break

    return StoryItemDetail(
        id=item.id,
        story_id=item.story_id,
        generation_id=item.generation_id,
        version_id=getattr(item, "version_id", None),
        start_time_ms=item.start_time_ms,
        track=item.track,
        trim_start_ms=getattr(item, "trim_start_ms", 0),
        trim_end_ms=getattr(item, "trim_end_ms", 0),
        created_at=item.created_at,
        profile_id=generation.profile_id,
        profile_name=profile_name,
        text=generation.text,
        language=generation.language,
        audio_path=audio_path,
        duration=generation.duration,
        seed=generation.seed,
        instruct=generation.instruct,
        engine=generation.engine,
        volume=getattr(item, "volume", 1.0),
        generation_created_at=generation.created_at,
        versions=versions,
        active_version_id=active_version_id,
    )


def _story_response(story: DBStory, db: Session) -> StoryResponse:
    """Build a Story response from one database view."""
    response = StoryResponse.model_validate(story)
    response.item_count = db.query(func.count(DBStoryItem.id)).filter_by(story_id=story.id).scalar() or 0
    return response


def _durable_created_story_response(
    db: Session,
    *,
    story_id: str,
    expected_fields: dict[str, object],
) -> StoryResponse | None:
    """Resolve a create acknowledgement against an independent durable view."""
    with deletion_journal.durable_reconciliation_session(db) as durable_db:
        durable_story = durable_db.query(DBStory).filter_by(id=story_id).one_or_none()
        if durable_story is None:
            return None
        if any(getattr(durable_story, field) != value for field, value in expected_fields.items()):
            raise RuntimeError("Story ID is owned by different durable data")
        return _story_response(durable_story, durable_db)


def _story_item_fields(item: DBStoryItem) -> dict[str, object]:
    """Capture the complete persisted Story-item contract before committing."""
    return {
        "story_id": item.story_id,
        "generation_id": item.generation_id,
        "version_id": item.version_id,
        "start_time_ms": item.start_time_ms,
        "track": item.track,
        "trim_start_ms": item.trim_start_ms,
        "trim_end_ms": item.trim_end_ms,
        "volume": item.volume,
        "created_at": item.created_at,
    }


def _durable_story_item_responses(
    db: Session,
    *,
    expected_items: list[tuple[str, dict[str, object]]],
) -> list[StoryItemDetail] | None:
    """Resolve item writes by their fresh IDs and exact persisted fields."""
    with deletion_journal.durable_reconciliation_session(db) as durable_db:
        durable_items: list[DBStoryItem] = []
        for item_id, expected_fields in expected_items:
            durable_item = durable_db.query(DBStoryItem).filter_by(id=item_id).one_or_none()
            if durable_item is None:
                return None
            if any(getattr(durable_item, field) != value for field, value in expected_fields.items()):
                raise RuntimeError("Story item ID is owned by different durable data")
            durable_items.append(durable_item)

        responses: list[StoryItemDetail] = []
        for durable_item in durable_items:
            generation = durable_db.query(DBGeneration).filter_by(id=durable_item.generation_id).one_or_none()
            if generation is None:
                raise RuntimeError("Durable Story item references a missing generation")
            profile = durable_db.query(DBVoiceProfile).filter_by(id=generation.profile_id).one_or_none()
            responses.append(
                _build_item_detail(
                    durable_item,
                    generation,
                    profile.name if profile else "Unknown",
                    durable_db,
                )
            )
        return responses


async def create_story(
    data: StoryCreate,
    db: Session,
) -> StoryResponse:
    """
    Create a new story.

    Args:
        data: Story creation data
        db: Database session

    Returns:
        Created story
    """
    story_id = str(uuid.uuid4())
    created_at = datetime.utcnow()
    updated_at = datetime.utcnow()
    expected_fields = {
        "name": data.name,
        "description": data.description,
        "created_at": created_at,
        "updated_at": updated_at,
    }
    db_story = DBStory(
        id=story_id,
        name=data.name,
        description=data.description,
        created_at=created_at,
        updated_at=updated_at,
    )

    db.add(db_story)
    try:
        db.commit()
        db.refresh(db_story)
        return _story_response(db_story, db)
    except BaseException as operation_error:
        try:
            db.rollback()
        except BaseException:
            logger.error("Story creation rollback failed", exc_info=True)
        try:
            durable_response = _durable_created_story_response(
                db,
                story_id=story_id,
                expected_fields=expected_fields,
            )
        except BaseException as reconciliation_error:
            raise operation_error from reconciliation_error
        if durable_response is not None:
            return durable_response
        raise operation_error


async def list_stories(
    db: Session,
) -> list[StoryResponse]:
    """
    List all stories.

    Args:
        db: Database session

    Returns:
        List of stories with item counts
    """
    stories = db.query(DBStory).order_by(DBStory.updated_at.desc()).all()

    if not stories:
        return []

    # Batch-fetch all story item counts in one query to avoid an N+1 pattern
    # (previously there was one COUNT query per story in the loop below).
    story_ids = [s.id for s in stories]
    count_rows = (
        db.query(DBStoryItem.story_id, func.count(DBStoryItem.id).label("cnt"))
        .filter(DBStoryItem.story_id.in_(story_ids))
        .group_by(DBStoryItem.story_id)
        .all()
    )
    item_counts = {row.story_id: row.cnt for row in count_rows}

    result = []
    for story in stories:
        response = StoryResponse.model_validate(story)
        response.item_count = item_counts.get(story.id, 0)
        result.append(response)

    return result


async def get_story(
    story_id: str,
    db: Session,
) -> StoryDetailResponse | None:
    """
    Get a story with all its items.

    Args:
        story_id: Story ID
        db: Database session

    Returns:
        Story with items or None if not found
    """
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return None

    items = (
        db.query(DBStoryItem, DBGeneration, DBVoiceProfile.name.label("profile_name"))
        .join(DBGeneration, DBStoryItem.generation_id == DBGeneration.id)
        .join(DBVoiceProfile, DBGeneration.profile_id == DBVoiceProfile.id)
        .filter(DBStoryItem.story_id == story_id)
        .order_by(DBStoryItem.start_time_ms)
        .all()
    )

    item_details = []
    for item, generation, profile_name in items:
        item_details.append(_build_item_detail(item, generation, profile_name, db))

    response = StoryDetailResponse.model_validate(story)
    response.items = item_details
    return response


async def update_story(
    story_id: str,
    data: StoryCreate,
    db: Session,
) -> StoryResponse | None:
    """
    Update a story.

    Args:
        story_id: Story ID
        data: Update data
        db: Database session

    Returns:
        Updated story or None if not found
    """
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return None

    story.name = data.name
    story.description = data.description
    story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(story)

    item_count = db.query(func.count(DBStoryItem.id)).filter(DBStoryItem.story_id == story.id).scalar()

    response = StoryResponse.model_validate(story)
    response.item_count = item_count
    return response


async def delete_story(
    story_id: str,
    db: Session,
) -> bool:
    """
    Delete a story and all its items.

    Args:
        story_id: Story ID
        db: Database session

    Returns:
        True if deleted, False if not found
    """
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return False

    # Delete all items
    db.query(DBStoryItem).filter_by(story_id=story_id).delete()

    # Delete story
    db.delete(story)
    db.commit()

    return True


async def add_item_to_story(
    story_id: str,
    data: StoryItemCreate,
    db: Session,
) -> StoryItemDetail | None:
    """
    Add a generation to a story.

    Args:
        story_id: Story ID
        data: Item creation data
        db: Database session

    Returns:
        Created item detail or None if story/generation not found
    """
    # Verify story exists
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return None

    # Verify generation exists
    generation = db.query(DBGeneration).filter_by(id=data.generation_id).first()
    if not generation:
        return None

    # Check if generation is already in story
    existing = db.query(DBStoryItem).filter_by(story_id=story_id, generation_id=data.generation_id).first()
    if existing:
        # Return existing item
        profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()
        return _build_item_detail(existing, generation, profile.name if profile else "Unknown", db)

    if _story_item_count(story_id, db) >= STORY_MAX_ITEMS:
        return None

    # Get track from data or default to 0
    track = data.track if data.track is not None else 0
    if not _valid_track(track):
        return None

    # Calculate start_time_ms if not provided
    if data.start_time_ms is not None:
        start_time_ms = data.start_time_ms
    else:
        existing_items = (
            db.query(DBStoryItem, DBGeneration)
            .join(DBGeneration, DBStoryItem.generation_id == DBGeneration.id)
            .filter(
                DBStoryItem.story_id == story_id,
                DBStoryItem.track == track,
            )
            .all()
        )

        if not existing_items:
            start_time_ms = 0
        else:
            max_end_time_ms = 0
            for item, gen in existing_items:
                try:
                    duration_ms = int(float(gen.duration) * 1000)
                except (TypeError, ValueError, OverflowError):
                    return None
                if not _valid_timecode(item.start_time_ms) or duration_ms < 0:
                    return None
                item_end_ms = item.start_time_ms + duration_ms
                max_end_time_ms = max(max_end_time_ms, item_end_ms)

            # Add 200ms gap after the last item
            start_time_ms = max_end_time_ms + 200

    if not _valid_timecode(start_time_ms):
        return None

    # Create item
    item = DBStoryItem(
        id=str(uuid.uuid4()),
        story_id=story_id,
        generation_id=data.generation_id,
        start_time_ms=start_time_ms,
        track=track,
        created_at=datetime.utcnow(),
    )

    db.add(item)

    # Update story updated_at
    story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    # Get profile name
    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()

    return _build_item_detail(item, generation, profile.name if profile else "Unknown", db)


async def move_story_item(
    story_id: str,
    item_id: str,
    data: StoryItemMove,
    db: Session,
) -> StoryItemDetail | None:
    """
    Move a story item (update position and/or track).

    Args:
        story_id: Story ID
        item_id: Story item ID
        data: New position and track data
        db: Database session

    Returns:
        Updated item detail or None if not found
    """
    # Get the item
    item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .first()
    )
    if not item:
        return None

    # Get the generation
    generation = db.query(DBGeneration).filter_by(id=item.generation_id).first()
    if not generation:
        return None

    if not _valid_timecode(data.start_time_ms) or not _valid_track(data.track):
        return None

    # Update position and track
    item.start_time_ms = data.start_time_ms
    item.track = data.track

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    # Get profile name
    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()

    return _build_item_detail(item, generation, profile.name if profile else "Unknown", db)


async def remove_item_from_story(
    story_id: str,
    item_id: str,
    db: Session,
) -> bool:
    """
    Remove a story item from a story.

    Args:
        story_id: Story ID
        item_id: Story item ID to remove
        db: Database session

    Returns:
        True if removed, False if not found
    """
    item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .first()
    )
    if not item:
        return False

    # Delete item
    db.delete(item)

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    return True


async def trim_story_item(
    story_id: str,
    item_id: str,
    data: StoryItemTrim,
    db: Session,
) -> StoryItemDetail | None:
    """
    Trim a story item (update trim_start_ms and trim_end_ms).

    Args:
        story_id: Story ID
        item_id: Story item ID
        data: Trim data (trim_start_ms, trim_end_ms)
        db: Database session

    Returns:
        Updated item detail or None if not found
    """
    # Get the item
    item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .first()
    )
    if not item:
        return None

    # Get the generation
    generation = db.query(DBGeneration).filter_by(id=item.generation_id).first()
    if not generation:
        return None

    # Validate trim values don't exceed duration. Repeat the request-model
    # bounds here because services are also called directly by integrations.
    if (
        not isinstance(data.trim_start_ms, int)
        or isinstance(data.trim_start_ms, bool)
        or not isinstance(data.trim_end_ms, int)
        or isinstance(data.trim_end_ms, bool)
        or not 0 <= data.trim_start_ms <= STORY_MAX_CLIP_MS
        or not 0 <= data.trim_end_ms <= STORY_MAX_CLIP_MS
    ):
        return None
    try:
        max_duration_ms = int(float(generation.duration) * 1000)
    except (TypeError, ValueError, OverflowError):
        return None
    if data.trim_start_ms + data.trim_end_ms >= max_duration_ms:
        return None  # Invalid trim - would result in zero or negative duration

    # Update trim values
    item.trim_start_ms = data.trim_start_ms
    item.trim_end_ms = data.trim_end_ms

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    # Get profile name
    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()

    return _build_item_detail(item, generation, profile.name if profile else "Unknown", db)


async def update_story_item_volume(
    story_id: str,
    item_id: str,
    data: StoryItemVolumeUpdate,
    db: Session,
) -> StoryItemDetail | None:
    """Update a story item's playback volume (per-clip linear gain)."""
    item = db.query(DBStoryItem).filter_by(id=item_id, story_id=story_id).first()
    if not item:
        return None
    generation = db.query(DBGeneration).filter_by(id=item.generation_id).first()
    if not generation:
        return None

    try:
        volume = float(data.volume)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(volume) or not 0.0 <= volume <= 2.0:
        return None

    item.volume = volume

    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()
    return _build_item_detail(item, generation, profile.name if profile else "Unknown", db)


async def split_story_item(
    story_id: str,
    item_id: str,
    data: StoryItemSplit,
    db: Session,
) -> list[StoryItemDetail] | None:
    """
    Split a story item at a given time, creating two clips.

    Args:
        story_id: Story ID
        item_id: Story item ID to split
        data: Split data (split_time_ms - time within clip to split at)
        db: Database session

    Returns:
        List of two updated item details (original and new) or None if not found/invalid
    """
    # Get the item with a row lock to prevent concurrent splits on the
    # same clip (e.g. from rapid double-clicks racing each other).
    item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .with_for_update()
        .first()
    )
    if not item:
        return None

    # Get the generation
    generation = db.query(DBGeneration).filter_by(id=item.generation_id).first()
    if not generation:
        return None

    if _story_item_count(story_id, db) >= STORY_MAX_ITEMS:
        return None

    # Calculate effective duration and validate split point
    current_trim_start = getattr(item, "trim_start_ms", 0)
    current_trim_end = getattr(item, "trim_end_ms", 0)
    try:
        original_duration_ms = int(float(generation.duration) * 1000)
    except (TypeError, ValueError, OverflowError):
        return None
    effective_duration_ms = original_duration_ms - current_trim_start - current_trim_end

    # Validate split_time_ms is within the effective duration
    if (
        not isinstance(data.split_time_ms, int)
        or isinstance(data.split_time_ms, bool)
        or data.split_time_ms <= 0
        or data.split_time_ms > STORY_MAX_CLIP_MS
        or data.split_time_ms >= effective_duration_ms
    ):
        return None  # Invalid split point

    # Calculate the absolute time in the original audio where we're splitting
    absolute_split_ms = current_trim_start + data.split_time_ms

    # Update original clip: trim from the end
    item.trim_end_ms = original_duration_ms - absolute_split_ms

    # Create new clip: starts after the split, trimmed from the start
    new_start_time_ms = item.start_time_ms + data.split_time_ms
    if not _valid_timecode(item.start_time_ms) or not _valid_timecode(new_start_time_ms):
        return None
    new_item_id = str(uuid.uuid4())
    new_item = DBStoryItem(
        id=new_item_id,
        story_id=story_id,
        generation_id=item.generation_id,  # Same generation, different trim
        version_id=getattr(item, "version_id", None),  # Preserve pinned version
        start_time_ms=new_start_time_ms,
        track=item.track,
        trim_start_ms=absolute_split_ms,
        trim_end_ms=current_trim_end,
        volume=getattr(item, "volume", 1.0),
        created_at=datetime.utcnow(),
    )

    db.add(new_item)

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    expected_items = [
        (item.id, _story_item_fields(item)),
        (new_item_id, _story_item_fields(new_item)),
    ]
    try:
        db.commit()
        db.refresh(item)
        db.refresh(new_item)

        # Get profile name
        profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()
        profile_name = profile.name if profile else "Unknown"

        return [
            _build_item_detail(item, generation, profile_name, db),
            _build_item_detail(new_item, generation, profile_name, db),
        ]
    except BaseException as operation_error:
        try:
            db.rollback()
        except BaseException:
            logger.error("Story-item split rollback failed", exc_info=True)
        try:
            durable_responses = _durable_story_item_responses(
                db,
                expected_items=expected_items,
            )
        except BaseException as reconciliation_error:
            raise operation_error from reconciliation_error
        if durable_responses is not None:
            return durable_responses
        raise operation_error


async def duplicate_story_item(
    story_id: str,
    item_id: str,
    db: Session,
) -> StoryItemDetail | None:
    """
    Duplicate a story item, creating a copy with all properties.

    Args:
        story_id: Story ID
        item_id: Story item ID to duplicate
        db: Database session

    Returns:
        New item detail or None if not found
    """
    # Get the original item
    original_item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .first()
    )
    if not original_item:
        return None

    # Get the generation
    generation = db.query(DBGeneration).filter_by(id=original_item.generation_id).first()
    if not generation:
        return None

    if _story_item_count(story_id, db) >= STORY_MAX_ITEMS:
        return None

    # Calculate effective duration
    current_trim_start = getattr(original_item, "trim_start_ms", 0)
    current_trim_end = getattr(original_item, "trim_end_ms", 0)
    try:
        original_duration_ms = int(float(generation.duration) * 1000)
    except (TypeError, ValueError, OverflowError):
        return None
    effective_duration_ms = original_duration_ms - current_trim_start - current_trim_end
    new_start_time_ms = original_item.start_time_ms + effective_duration_ms + 200
    if (
        effective_duration_ms <= 0
        or not _valid_timecode(original_item.start_time_ms)
        or not _valid_timecode(new_start_time_ms)
    ):
        return None

    # Create duplicate item - place it right after the original
    new_item_id = str(uuid.uuid4())
    new_item = DBStoryItem(
        id=new_item_id,
        story_id=story_id,
        generation_id=original_item.generation_id,  # Same generation as original
        version_id=getattr(original_item, "version_id", None),  # Preserve pinned version
        start_time_ms=new_start_time_ms,
        track=original_item.track,
        trim_start_ms=current_trim_start,
        trim_end_ms=current_trim_end,
        volume=getattr(original_item, "volume", 1.0),
        created_at=datetime.utcnow(),
    )

    db.add(new_item)

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    expected_items = [(new_item_id, _story_item_fields(new_item))]
    try:
        db.commit()
        db.refresh(new_item)

        # Get profile name
        profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()

        return _build_item_detail(new_item, generation, profile.name if profile else "Unknown", db)
    except BaseException as operation_error:
        try:
            db.rollback()
        except BaseException:
            logger.error("Story-item duplication rollback failed", exc_info=True)
        try:
            durable_responses = _durable_story_item_responses(
                db,
                expected_items=expected_items,
            )
        except BaseException as reconciliation_error:
            raise operation_error from reconciliation_error
        if durable_responses is not None:
            return durable_responses[0]
        raise operation_error


async def update_story_item_times(
    story_id: str,
    data: StoryItemBatchUpdate,
    db: Session,
) -> bool:
    """
    Update story item timecodes.

    Args:
        story_id: Story ID
        data: Batch update data with timecodes
        db: Database session

    Returns:
        True if updated, False if story not found or invalid
    """
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return False
    if not 1 <= len(data.updates) <= STORY_MAX_ITEMS:
        return False

    # Get all items for this story
    items = db.query(DBStoryItem).filter_by(story_id=story_id).all()
    item_map = {item.id: item for item in items}

    if len({update.item_id for update in data.updates}) != len(data.updates):
        return False

    # Validate the complete request before mutating any row.
    for update in data.updates:
        if update.item_id not in item_map or not _valid_timecode(update.start_time_ms):
            return False

    for update in data.updates:
        item_map[update.item_id].start_time_ms = update.start_time_ms

    # Update story updated_at
    story.updated_at = datetime.utcnow()

    db.commit()
    return True


async def reorder_story_items(
    story_id: str,
    item_ids: list[str],
    db: Session,
    gap_ms: int = 200,
) -> list[StoryItemDetail] | None:
    """
    Reorder story items and recalculate timecodes.

    Args:
        story_id: Story ID
        item_ids: List of Story item IDs in the desired order
        db: Database session
        gap_ms: Gap in milliseconds between items (default 200ms)

    Returns:
        Updated list of story items with new timecodes, or None if invalid
    """
    story = db.query(DBStory).filter_by(id=story_id).first()
    if not story:
        return None
    if (
        not isinstance(gap_ms, int)
        or isinstance(gap_ms, bool)
        or not 0 <= gap_ms <= 60_000
        or not 1 <= len(item_ids) <= STORY_MAX_ITEMS
    ):
        return None

    # Get all items for this story with their generation data
    items_with_gen = (
        db.query(DBStoryItem, DBGeneration, DBVoiceProfile.name.label("profile_name"))
        .join(DBGeneration, DBStoryItem.generation_id == DBGeneration.id)
        .join(DBVoiceProfile, DBGeneration.profile_id == DBVoiceProfile.id)
        .filter(DBStoryItem.story_id == story_id)
        .all()
    )

    # Create maps for quick lookup
    item_map = {item.id: (item, gen, profile_name) for item, gen, profile_name in items_with_gen}

    # Verify every item ID appears exactly once and belongs to this story.
    if len(item_ids) != len(item_map) or len(set(item_ids)) != len(item_ids) or set(item_ids) != set(item_map):
        return None

    # Calculate every position before mutating the session so an invalid or
    # corrupt duration cannot leave a partially updated timeline pending.
    current_time_ms = 0
    positions: list[tuple[str, int]] = []
    for item_id in item_ids:
        item, generation, _profile_name = item_map[item_id]
        try:
            duration_ms = int(float(generation.duration) * 1000)
        except (TypeError, ValueError, OverflowError):
            return None
        duration_ms -= item.trim_start_ms + item.trim_end_ms
        if duration_ms <= 0 or duration_ms > STORY_MAX_CLIP_MS:
            return None
        if current_time_ms + duration_ms > STORY_MAX_TIMELINE_MS:
            return None
        positions.append((item_id, current_time_ms))
        current_time_ms += duration_ms + gap_ms

    updated_items = []
    for item_id, start_time_ms in positions:
        item, generation, profile_name = item_map[item_id]
        item.start_time_ms = start_time_ms
        updated_items.append(_build_item_detail(item, generation, profile_name, db))

    # Update story updated_at
    story.updated_at = datetime.utcnow()

    db.commit()
    return updated_items


async def set_story_item_version(
    story_id: str,
    item_id: str,
    data: StoryItemVersionUpdate,
    db: Session,
) -> StoryItemDetail | None:
    """
    Pin a story item to a specific generation version.

    Args:
        story_id: Story ID
        item_id: Story item ID
        data: Version update data (version_id or null for default)
        db: Database session

    Returns:
        Updated item detail or None if not found
    """
    item = (
        db.query(DBStoryItem)
        .filter_by(
            id=item_id,
            story_id=story_id,
        )
        .first()
    )
    if not item:
        return None

    generation = db.query(DBGeneration).filter_by(id=item.generation_id).first()
    if not generation:
        return None

    # Validate version_id belongs to this generation if provided
    if data.version_id:
        from ..database import GenerationVersion as DBGenerationVersion

        version = (
            db.query(DBGenerationVersion)
            .filter_by(
                id=data.version_id,
                generation_id=item.generation_id,
            )
            .first()
        )
        if not version:
            return None

    item.version_id = data.version_id

    # Update story updated_at
    story = db.query(DBStory).filter_by(id=story_id).first()
    if story:
        story.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(item)

    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()

    return _build_item_detail(item, generation, profile.name if profile else "Unknown", db)


def _check_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise _StoryAudioExportCancelledError


def _validate_source_shape(*, frames: int, sample_rate: int, channels: int, generation_id: str) -> None:
    if frames <= 0 or sample_rate <= 0 or channels <= 0:
        raise StoryAudioExportError(f"Audio for story item {generation_id} has invalid duration metadata")
    if sample_rate > STORY_EXPORT_MAX_SAMPLE_RATE:
        raise StoryAudioExportLimitError(
            f"Audio for story item {generation_id} exceeds the {STORY_EXPORT_MAX_SAMPLE_RATE} Hz sample-rate limit"
        )
    if channels > STORY_EXPORT_MAX_CHANNELS:
        raise StoryAudioExportLimitError(
            f"Audio for story item {generation_id} exceeds the {STORY_EXPORT_MAX_CHANNELS}-channel limit"
        )
    if frames * channels > STORY_EXPORT_MAX_SOURCE_SAMPLE_VALUES:
        raise StoryAudioExportLimitError(f"Audio for story item {generation_id} exceeds the decoded-sample limit")
    duration_seconds = frames / sample_rate
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise StoryAudioExportError(f"Audio for story item {generation_id} has invalid duration metadata")
    if duration_seconds > STORY_MAX_CLIP_MS / 1000:
        raise StoryAudioExportLimitError(
            f"Audio for story item {generation_id} exceeds the {STORY_MAX_CLIP_MS // 60_000}-minute clip limit"
        )


def _probe_clip(clip: _StoryClip, cancel_event: threading.Event) -> _ProbedClip:
    """Read bounded source metadata, decoding once only when libsndfile cannot."""
    _check_cancelled(cancel_event)
    try:
        source_stat = clip.path.stat()
    except OSError as exc:
        raise StoryAudioExportError(f"Audio for story item {clip.generation_id} is unavailable") from exc
    if not stat.S_ISREG(source_stat.st_mode):
        raise StoryAudioExportError(f"Audio for story item {clip.generation_id} is not a regular file")

    decoder = "soundfile"
    try:
        with sf.SoundFile(str(clip.path), mode="r") as source:
            frames = int(source.frames)
            sample_rate = int(source.samplerate)
            channels = int(source.channels)
    except (OSError, RuntimeError, TypeError, ValueError):
        decoder = "audioread"
        try:
            with audioread.audio_open(str(clip.path)) as source:
                sample_rate = int(source.samplerate)
                channels = int(source.channels)
                frames = 0
                frame_bytes = channels * 2
                for payload in source:
                    _check_cancelled(cancel_event)
                    if len(payload) % frame_bytes:
                        raise StoryAudioExportError(
                            f"Audio for story item {clip.generation_id} contains a malformed decoded frame"
                        )
                    frames += len(payload) // frame_bytes
                    _validate_source_shape(
                        frames=frames,
                        sample_rate=sample_rate,
                        channels=channels,
                        generation_id=clip.generation_id,
                    )
        except StoryAudioExportError:
            raise
        except Exception as exc:
            raise StoryAudioExportError(f"Audio for story item {clip.generation_id} could not be decoded") from exc

    _validate_source_shape(
        frames=frames,
        sample_rate=sample_rate,
        channels=channels,
        generation_id=clip.generation_id,
    )
    # libsoxr rounds positive half-frame results up. Python's round() uses
    # ties-to-even and would make valid odd-length 48/96/192 kHz clips fail
    # the post-decode consistency check by one frame.
    output_frames = (frames * STORY_EXPORT_SAMPLE_RATE + sample_rate // 2) // sample_rate
    trim_start_frames = clip.trim_start_ms * STORY_EXPORT_SAMPLE_RATE // 1000
    trim_end_frames = clip.trim_end_ms * STORY_EXPORT_SAMPLE_RATE // 1000
    if trim_start_frames + trim_end_frames >= output_frames:
        raise StoryAudioExportError(f"Trims remove all audio from story item {clip.generation_id}")
    return _ProbedClip(
        clip=clip,
        frames=frames,
        sample_rate=sample_rate,
        channels=channels,
        decoder=decoder,
        output_frames=output_frames,
        trim_start_frames=trim_start_frames,
        trim_end_frames=trim_end_frames,
    )


def _mono_block(block: np.ndarray, channels: int, generation_id: str) -> np.ndarray:
    if block.ndim != 2 or block.shape[1] != channels:
        raise StoryAudioExportError(f"Audio for story item {generation_id} changed while exporting")
    if not np.isfinite(block).all():
        raise StoryAudioExportError(f"Audio for story item {generation_id} contains non-finite samples")
    return block.mean(axis=1, dtype=np.float32)


def _decoded_mono_blocks(
    probed: _ProbedClip,
    cancel_event: threading.Event,
) -> Iterator[np.ndarray]:
    """Decode one source incrementally, never materializing a whole clip."""
    if probed.decoder == "soundfile":
        try:
            with sf.SoundFile(str(probed.clip.path), mode="r") as source:
                if (
                    int(source.samplerate) != probed.sample_rate
                    or int(source.channels) != probed.channels
                    or int(source.frames) != probed.frames
                ):
                    raise StoryAudioExportError(
                        f"Audio for story item {probed.clip.generation_id} changed while exporting"
                    )
                while True:
                    _check_cancelled(cancel_event)
                    block = source.read(
                        STORY_EXPORT_BLOCK_FRAMES,
                        dtype="float32",
                        always_2d=True,
                    )
                    if not len(block):
                        break
                    yield _mono_block(block, probed.channels, probed.clip.generation_id)
        except StoryAudioExportError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise StoryAudioExportError(
                f"Audio for story item {probed.clip.generation_id} could not be decoded"
            ) from exc
        return

    decoded_frames = 0
    try:
        with audioread.audio_open(str(probed.clip.path)) as source:
            if int(source.samplerate) != probed.sample_rate or int(source.channels) != probed.channels:
                raise StoryAudioExportError(f"Audio for story item {probed.clip.generation_id} changed while exporting")
            frame_bytes = probed.channels * 2
            for payload in source:
                _check_cancelled(cancel_event)
                if len(payload) % frame_bytes:
                    raise StoryAudioExportError(
                        f"Audio for story item {probed.clip.generation_id} contains a malformed decoded frame"
                    )
                block = np.frombuffer(payload, dtype="<i2").reshape(-1, probed.channels)
                decoded_frames += len(block)
                float_block = block.astype(np.float32) / 32768.0
                yield _mono_block(float_block, probed.channels, probed.clip.generation_id)
    except StoryAudioExportError:
        raise
    except Exception as exc:
        raise StoryAudioExportError(f"Audio for story item {probed.clip.generation_id} could not be decoded") from exc
    if decoded_frames != probed.frames:
        raise StoryAudioExportError(f"Audio for story item {probed.clip.generation_id} changed while exporting")


def _resampled_mono_blocks(
    probed: _ProbedClip,
    cancel_event: threading.Event,
) -> Iterator[np.ndarray]:
    blocks = _decoded_mono_blocks(probed, cancel_event)
    if probed.sample_rate == STORY_EXPORT_SAMPLE_RATE:
        yield from blocks
        return

    resampler = soxr.ResampleStream(
        probed.sample_rate,
        STORY_EXPORT_SAMPLE_RATE,
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


def _mix_clip(
    timeline: np.memmap,
    probed: _ProbedClip,
    cancel_event: threading.Event,
) -> None:
    output_position = 0
    mix_start = probed.clip.start_time_ms * STORY_EXPORT_SAMPLE_RATE // 1000
    keep_end = probed.output_frames - probed.trim_end_frames
    for block in _resampled_mono_blocks(probed, cancel_event):
        _check_cancelled(cancel_event)
        block_end = output_position + len(block)
        keep_start_in_block = max(probed.trim_start_frames, output_position)
        keep_end_in_block = min(keep_end, block_end)
        if keep_start_in_block < keep_end_in_block:
            source_start = keep_start_in_block - output_position
            source_end = keep_end_in_block - output_position
            destination_start = mix_start + keep_start_in_block - probed.trim_start_frames
            destination_end = destination_start + source_end - source_start
            if destination_end > len(timeline):
                raise StoryAudioExportError(
                    f"Audio for story item {probed.clip.generation_id} exceeded its probed duration"
                )
            source_audio = block[source_start:source_end]
            if probed.clip.volume != 1.0:
                source_audio = source_audio * probed.clip.volume
            timeline[destination_start:destination_end] += source_audio
        output_position = block_end

    if output_position != probed.output_frames:
        raise StoryAudioExportError(
            f"Audio for story item {probed.clip.generation_id} produced an inconsistent decoded duration"
        )


def _render_story_audio_to_path(
    clips: list[_StoryClip],
    *,
    mix_path: Path,
    output_path: Path,
    cancel_event: threading.Event,
) -> None:
    """Render with constant-sized blocks and a disk-backed float timeline."""
    probed_clips: list[_ProbedClip] = []
    total_source_seconds = 0.0
    total_source_sample_values = 0
    total_frames = 0
    for clip in clips:
        probed = _probe_clip(clip, cancel_event)
        probed_clips.append(probed)
        total_source_seconds += probed.frames / probed.sample_rate
        total_source_sample_values += probed.frames * probed.channels
        if total_source_seconds > STORY_EXPORT_MAX_SOURCE_SECONDS:
            raise StoryAudioExportLimitError(
                f"Story source audio exceeds the {STORY_EXPORT_MAX_SOURCE_SECONDS // 3600}-hour decode limit"
            )
        if total_source_sample_values > STORY_EXPORT_MAX_SOURCE_SAMPLE_VALUES:
            raise StoryAudioExportLimitError("Story source audio exceeds the decoded-sample limit")
        effective_frames = probed.output_frames - probed.trim_start_frames - probed.trim_end_frames
        start_frame = clip.start_time_ms * STORY_EXPORT_SAMPLE_RATE // 1000
        total_frames = max(total_frames, start_frame + effective_frames)

    max_timeline_frames = STORY_MAX_TIMELINE_MS * STORY_EXPORT_SAMPLE_RATE // 1000
    if total_frames <= 0:
        raise StoryAudioExportError("Story has no audible audio after trimming")
    if total_frames > max_timeline_frames:
        raise StoryAudioExportLimitError(
            f"Story timeline exceeds the {STORY_MAX_TIMELINE_MS // 3_600_000}-hour export limit"
        )

    output_bytes = 44 + total_frames * 2
    if output_bytes > STORY_EXPORT_MAX_OUTPUT_BYTES:
        raise StoryAudioExportLimitError("Story WAV output exceeds the supported file-size limit")
    required_scratch_bytes = total_frames * np.dtype(np.float32).itemsize + output_bytes
    try:
        reservation = reserve_disk_space(
            mix_path.parent,
            required_scratch_bytes,
            min_free_bytes=STORY_EXPORT_MIN_FREE_BYTES,
        )
    except DiskSpaceReservationError as exc:
        raise StoryAudioExportLimitError(
            "Not enough temporary disk space for the story export "
            f"({required_scratch_bytes / 1024**3:.1f} GiB required plus a 1 GiB reserve)"
        ) from exc
    with _story_export_state_lock:
        old_reservation = _story_export_reservations.get(mix_path.parent)
        if old_reservation is not None:
            reservation.release()
            raise StoryAudioExportError("Story export storage already has an active reservation")
        _story_export_reservations[mix_path.parent] = reservation

    descriptor = os.open(mix_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, total_frames * np.dtype(np.float32).itemsize)
    finally:
        os.close(descriptor)

    timeline: np.memmap | None = None
    try:
        timeline = np.memmap(mix_path, dtype=np.float32, mode="r+", shape=(total_frames,))
        for probed in probed_clips:
            _mix_clip(timeline, probed, cancel_event)

        peak = 0.0
        for offset in range(0, total_frames, STORY_EXPORT_BLOCK_FRAMES):
            _check_cancelled(cancel_event)
            block = timeline[offset : offset + STORY_EXPORT_BLOCK_FRAMES]
            if len(block):
                peak = max(peak, float(np.max(np.abs(block))))
        if not math.isfinite(peak):
            raise StoryAudioExportError("Story mix contains non-finite samples")
        gain = 1.0 / peak if peak > 1.0 else 1.0

        with sf.SoundFile(
            str(output_path),
            mode="x",
            samplerate=STORY_EXPORT_SAMPLE_RATE,
            channels=1,
            format="WAV",
            subtype="PCM_16",
        ) as output:
            for offset in range(0, total_frames, STORY_EXPORT_BLOCK_FRAMES):
                _check_cancelled(cancel_event)
                block = np.asarray(timeline[offset : offset + STORY_EXPORT_BLOCK_FRAMES], dtype=np.float32)
                if gain != 1.0:
                    block = block * gain
                output.write(block)
            output.flush()
    finally:
        if timeline is not None:
            timeline.flush()
            del timeline
        mix_path.unlink(missing_ok=True)


async def _run_export_worker_cancellation_safe(function, /, *args, **kwargs):
    """Signal, drain, and consume a non-cancellable executor operation."""
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


def _validated_export_clips(story_id: str, db: Session) -> list[_StoryClip] | None:
    if not db.query(DBStory.id).filter_by(id=story_id).first():
        return None
    item_count = db.query(func.count(DBStoryItem.id)).filter_by(story_id=story_id).scalar() or 0
    if item_count == 0:
        return None
    if item_count > STORY_MAX_ITEMS:
        raise StoryAudioExportLimitError(f"Story exceeds the {STORY_MAX_ITEMS}-item export limit")

    items = (
        db.query(DBStoryItem, DBGeneration)
        .join(DBGeneration, DBStoryItem.generation_id == DBGeneration.id)
        .filter(DBStoryItem.story_id == story_id)
        .order_by(DBStoryItem.start_time_ms)
        .all()
    )
    if len(items) != item_count:
        raise StoryAudioExportError("One or more story items no longer has generation audio")

    clips: list[_StoryClip] = []
    for item, generation in items:
        start_time_ms = item.start_time_ms
        trim_start_ms = getattr(item, "trim_start_ms", 0)
        trim_end_ms = getattr(item, "trim_end_ms", 0)
        volume_value = getattr(item, "volume", 1.0)
        try:
            volume = 1.0 if volume_value is None else float(volume_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise StoryAudioExportError(f"Story item {generation.id} contains an invalid volume") from exc
        if (
            isinstance(start_time_ms, bool)
            or not isinstance(start_time_ms, int)
            or not 0 <= start_time_ms <= STORY_MAX_TIMELINE_MS
        ):
            raise StoryAudioExportLimitError("Story contains an invalid or out-of-range start time")
        if (
            isinstance(trim_start_ms, bool)
            or not isinstance(trim_start_ms, int)
            or not 0 <= trim_start_ms <= STORY_MAX_CLIP_MS
            or isinstance(trim_end_ms, bool)
            or not isinstance(trim_end_ms, int)
            or not 0 <= trim_end_ms <= STORY_MAX_CLIP_MS
        ):
            raise StoryAudioExportError(f"Story item {generation.id} contains invalid trim values")
        if not math.isfinite(volume) or not 0.0 <= volume <= 2.0:
            raise StoryAudioExportError(f"Story item {generation.id} contains an invalid volume")

        resolved_audio_path = generation.audio_path
        if item.version_id:
            version = (
                db.query(DBGenerationVersion).filter_by(id=item.version_id, generation_id=item.generation_id).first()
            )
            if version is None:
                raise StoryAudioExportError(f"Story item {generation.id} references a missing audio version")
            resolved_audio_path = version.audio_path
        audio_path = config.resolve_storage_path(resolved_audio_path)
        if audio_path is None:
            raise StoryAudioExportError(f"Story item {generation.id} has no audio file")
        clips.append(
            _StoryClip(
                path=audio_path,
                generation_id=generation.id,
                start_time_ms=start_time_ms,
                trim_start_ms=trim_start_ms,
                trim_end_ms=trim_end_ms,
                volume=volume,
            )
        )
    return clips


async def export_story_audio(
    story_id: str,
    db: Session,
) -> StoryAudioExport | None:
    """Export a story to a private, streamed WAV without timeline-sized RAM."""
    clips = _validated_export_clips(story_id, db)
    if not clips:
        return None
    db.rollback()

    if _story_export_lock.locked():
        raise StoryAudioExportBusyError("Another story export is already running; retry when it finishes")
    await _story_export_lock.acquire()
    try:
        temporary_directory = _allocate_story_export_directory()
        mix_path = temporary_directory / "timeline.f32"
        output_path = temporary_directory / "story.wav"
        try:
            await _run_export_worker_cancellation_safe(
                _render_story_audio_to_path,
                clips,
                mix_path=mix_path,
                output_path=output_path,
            )
            return StoryAudioExport(path=output_path, temporary_directory=temporary_directory)
        except BaseException:
            _cleanup_story_export(temporary_directory)
            raise
    finally:
        _story_export_lock.release()
