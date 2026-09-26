"""
Audio channel management module.
"""

import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from ..database import (
    AudioChannel as DBAudioChannel,
    ChannelDeviceMapping as DBChannelDeviceMapping,
    ProfileChannelMapping as DBProfileChannelMapping,
    VoiceProfile as DBVoiceProfile,
)
from ..models import (
    AudioChannelCreate,
    AudioChannelResponse,
    AudioChannelUpdate,
    ChannelVoiceAssignment,
    ProfileChannelAssignment,
)


def _channel_response(channel: DBAudioChannel, device_ids: list[str]) -> AudioChannelResponse:
    """Serialize channel metadata consistently across reads and writes."""
    return AudioChannelResponse(
        id=channel.id,
        name=channel.name,
        is_default=channel.is_default,
        device_ids=device_ids,
        created_at=channel.created_at,
    )


async def list_channels(db: Session) -> list[AudioChannelResponse]:
    """List all audio channels with one shared device-mapping lookup."""
    channel_rows = db.query(DBAudioChannel).all()
    if not channel_rows:
        return []
    device_ids: dict[str, list[str]] = {}
    for mapping in db.query(DBChannelDeviceMapping).all():
        device_ids.setdefault(mapping.channel_id, []).append(mapping.device_id)
    return [_channel_response(channel, device_ids.get(channel.id, [])) for channel in channel_rows]


async def get_channel(channel_id: str, db: Session) -> AudioChannelResponse | None:
    """Get a channel by ID."""
    channel = db.query(DBAudioChannel).filter_by(id=channel_id).first()
    if not channel:
        return None

    # Get device IDs
    device_mappings = db.query(DBChannelDeviceMapping).filter_by(channel_id=channel.id).all()
    device_ids = [m.device_id for m in device_mappings]

    return _channel_response(channel, device_ids)


async def create_channel(
    data: AudioChannelCreate,
    db: Session,
) -> AudioChannelResponse:
    """Create a new audio channel."""
    # Check if name already exists
    existing = db.query(DBAudioChannel).filter_by(name=data.name).first()
    if existing:
        raise ValueError(f"Channel with name '{data.name}' already exists")

    # Create channel
    channel = DBAudioChannel(
        id=str(uuid.uuid4()),
        name=data.name,
        is_default=False,
        created_at=datetime.utcnow(),
    )
    db.add(channel)
    db.flush()

    # Add device mappings
    for device_id in data.device_ids:
        mapping = DBChannelDeviceMapping(
            id=str(uuid.uuid4()),
            channel_id=channel.id,
            device_id=device_id,
        )
        db.add(mapping)

    db.commit()
    db.refresh(channel)

    return _channel_response(channel, data.device_ids)


async def update_channel(
    channel_id: str,
    data: AudioChannelUpdate,
    db: Session,
) -> AudioChannelResponse | None:
    """Update an audio channel."""
    channel = db.query(DBAudioChannel).filter_by(id=channel_id).first()
    if not channel:
        return None

    if channel.is_default:
        raise ValueError("Cannot modify the default channel")

    # Update name if provided
    if data.name is not None:
        # Check if name already exists (excluding current channel)
        existing = (
            db.query(DBAudioChannel).filter(DBAudioChannel.name == data.name, DBAudioChannel.id != channel_id).first()
        )
        if existing:
            raise ValueError(f"Channel with name '{data.name}' already exists")
        channel.name = data.name

    # Update device mappings if provided
    if data.device_ids is not None:
        # Delete existing mappings
        db.query(DBChannelDeviceMapping).filter_by(channel_id=channel_id).delete()

        # Add new mappings
        for device_id in data.device_ids:
            mapping = DBChannelDeviceMapping(
                id=str(uuid.uuid4()),
                channel_id=channel.id,
                device_id=device_id,
            )
            db.add(mapping)

    db.commit()
    db.refresh(channel)

    # Get updated device IDs
    device_mappings = db.query(DBChannelDeviceMapping).filter_by(channel_id=channel.id).all()
    device_ids = [m.device_id for m in device_mappings]

    return _channel_response(channel, device_ids)


async def delete_channel(channel_id: str, db: Session) -> bool:
    """Delete an audio channel."""
    channel = db.query(DBAudioChannel).filter_by(id=channel_id).first()
    if not channel:
        return False

    if channel.is_default:
        raise ValueError("Cannot delete the default channel")

    # Delete device mappings
    db.query(DBChannelDeviceMapping).filter_by(channel_id=channel_id).delete()

    # Delete profile-channel mappings
    db.query(DBProfileChannelMapping).filter_by(channel_id=channel_id).delete()

    # Delete channel
    db.delete(channel)
    db.commit()

    return True


async def get_channel_voices(channel_id: str, db: Session) -> list[str]:
    """Get list of profile IDs assigned to a channel."""
    mappings = db.query(DBProfileChannelMapping).filter_by(channel_id=channel_id).all()
    return [m.profile_id for m in mappings]


def _replace_profile_channel_mappings(
    owner: DBAudioChannel | DBVoiceProfile,
    related_ids: list[str],
    db: Session,
) -> None:
    """Replace either side of the same many-to-many relationship atomically."""
    if isinstance(owner, DBAudioChannel):
        owner_key, related_key = "channel_id", "profile_id"
        related_model, related_label = DBVoiceProfile, "Profile"
    else:
        owner_key, related_key = "profile_id", "channel_id"
        related_model, related_label = DBAudioChannel, "Channel"

    unique_ids = list(dict.fromkeys(related_ids))
    existing_ids: set[str] = set()
    # Bound SQLite placeholders even for a large assignment payload.
    for offset in range(0, len(unique_ids), 500):
        existing_ids.update(
            related_id
            for (related_id,) in db.query(related_model.id).filter(
                related_model.id.in_(unique_ids[offset : offset + 500])
            )
        )
    for related_id in unique_ids:
        if related_id not in existing_ids:
            raise ValueError(f"{related_label} {related_id} not found")

    try:
        db.query(DBProfileChannelMapping).filter_by(**{owner_key: owner.id}).delete()
        db.add_all(
            DBProfileChannelMapping(**{owner_key: owner.id, related_key: related_id}) for related_id in unique_ids
        )
        db.commit()
    except Exception:
        db.rollback()
        raise


async def set_channel_voices(
    channel_id: str,
    data: ChannelVoiceAssignment,
    db: Session,
) -> None:
    """Set which voices are assigned to a channel."""
    # Verify channel exists
    channel = db.query(DBAudioChannel).filter_by(id=channel_id).first()
    if not channel:
        raise ValueError(f"Channel {channel_id} not found")

    _replace_profile_channel_mappings(channel, data.profile_ids, db)


async def get_profile_channels(profile_id: str, db: Session) -> list[str]:
    """Get list of channel IDs assigned to a profile."""
    mappings = db.query(DBProfileChannelMapping).filter_by(profile_id=profile_id).all()
    return [m.channel_id for m in mappings]


async def set_profile_channels(
    profile_id: str,
    data: ProfileChannelAssignment,
    db: Session,
) -> None:
    """Set which channels a profile is assigned to."""
    # Verify profile exists
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    _replace_profile_channel_mappings(profile, data.channel_ids, db)
