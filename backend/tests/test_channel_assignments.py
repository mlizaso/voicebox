"""Channel assignment replacement preserves set semantics and existing mappings."""

import sqlite3

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from backend.database import AudioChannel, Base, ChannelDeviceMapping, ProfileChannelMapping, VoiceProfile
from backend.models import AudioChannelCreate, AudioChannelUpdate, ChannelVoiceAssignment, ProfileChannelAssignment
from backend.services import channels


@pytest.fixture
def database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'channels.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all(
            [
                AudioChannel(id="channel-a", name="Channel A"),
                AudioChannel(id="channel-b", name="Channel B"),
                VoiceProfile(id="profile-a", name="Profile A", language="en"),
                VoiceProfile(id="profile-b", name="Profile B", language="en"),
            ]
        )
        db.commit()
        yield db
    engine.dispose()


@pytest.fixture(params=["channel", "profile"])
def assignment(request):
    if request.param == "channel":
        return channels.set_channel_voices, "channel-a", ChannelVoiceAssignment, "profile_ids", "profile-b"
    return channels.set_profile_channels, "profile-a", ProfileChannelAssignment, "channel_ids", "channel-b"


@pytest.mark.asyncio
async def test_duplicate_assignments_replace_existing_links_once(database, assignment):
    setter, owner_id, payload_type, field, related_id = assignment
    database.add(ProfileChannelMapping(profile_id="profile-a", channel_id="channel-a"))
    database.commit()

    await setter(owner_id, payload_type(**{field: [related_id, related_id]}), database)

    rows = database.query(ProfileChannelMapping).all()
    assert [(row.profile_id, row.channel_id) for row in rows] == [
        (related_id, owner_id) if field == "profile_ids" else (owner_id, related_id)
    ]


@pytest.mark.asyncio
async def test_invalid_assignment_retains_existing_links(database, assignment):
    setter, owner_id, payload_type, field, related_id = assignment
    database.add(ProfileChannelMapping(profile_id="profile-a", channel_id="channel-a"))
    database.commit()

    with pytest.raises(ValueError, match="missing not found"):
        await setter(owner_id, payload_type(**{field: [related_id, "missing"]}), database)

    rows = database.query(ProfileChannelMapping).all()
    assert [(row.profile_id, row.channel_id) for row in rows] == [("profile-a", "channel-a")]


@pytest.mark.asyncio
@pytest.mark.parametrize("clear", [False, True])
async def test_replacement_changes_only_the_selected_owner(database, assignment, clear):
    setter, owner_id, payload_type, field, related_id = assignment
    database.add_all(
        [
            ProfileChannelMapping(profile_id="profile-a", channel_id="channel-a"),
            ProfileChannelMapping(profile_id="profile-b", channel_id="channel-b"),
        ]
    )
    database.commit()

    await setter(owner_id, payload_type(**{field: [] if clear else [related_id]}), database)

    rows = database.query(ProfileChannelMapping).all()
    expected = {("profile-b", "channel-b")}
    if not clear:
        expected.add((related_id, owner_id) if field == "profile_ids" else (owner_id, related_id))
    assert {(row.profile_id, row.channel_id) for row in rows} == expected


@pytest.mark.asyncio
async def test_large_assignment_respects_sqlite_parameter_limit(database, assignment):
    setter, owner_id, payload_type, field, _related_id = assignment
    related_ids = [f"related-{index}" for index in range(501)]
    model = VoiceProfile if field == "profile_ids" else AudioChannel
    database.add_all(model(id=related_id, name=related_id) for related_id in related_ids)
    database.commit()
    connection = database.connection().connection.driver_connection
    previous_limit = connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 500)
    try:
        await setter(owner_id, payload_type(**{field: related_ids}), database)
    finally:
        connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)

    rows = database.query(ProfileChannelMapping).all()
    assert len(rows) == len(related_ids)
    assert {row.profile_id if field == "profile_ids" else row.channel_id for row in rows} == set(related_ids)


@pytest.mark.asyncio
async def test_failed_assignment_commit_rolls_back_and_leaves_session_usable(database, assignment, monkeypatch):
    setter, owner_id, payload_type, field, related_id = assignment
    database.add(ProfileChannelMapping(profile_id="profile-a", channel_id="channel-a"))
    database.commit()

    def fail_commit():
        database.flush()
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(database, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="storage unavailable"):
        await setter(owner_id, payload_type(**{field: [related_id]}), database)

    rows = database.query(ProfileChannelMapping).all()
    assert [(row.profile_id, row.channel_id) for row in rows] == [("profile-a", "channel-a")]


@pytest.mark.asyncio
async def test_channel_list_batches_device_lookup(database):
    database.add_all(
        [
            ChannelDeviceMapping(channel_id="channel-a", device_id="speaker-a"),
            ChannelDeviceMapping(channel_id="channel-a", device_id="speaker-b"),
            ChannelDeviceMapping(channel_id="channel-b", device_id="headphones"),
        ]
    )
    database.commit()
    statements = []

    def record_statement(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)

    engine = database.get_bind()
    event.listen(engine, "before_cursor_execute", record_statement)
    try:
        result = await channels.list_channels(database)
    finally:
        event.remove(engine, "before_cursor_execute", record_statement)

    assert {channel.id: channel.device_ids for channel in result} == {
        "channel-a": ["speaker-a", "speaker-b"],
        "channel-b": ["headphones"],
    }
    assert len(statements) == 2


@pytest.mark.asyncio
async def test_channel_reads_and_writes_share_the_same_response_contract(database):
    created = await channels.create_channel(AudioChannelCreate(name="New", device_ids=["speaker"]), database)
    assert created.device_ids == ["speaker"]
    assert await channels.get_channel(created.id, database) == created
    updated = await channels.update_channel(
        created.id,
        AudioChannelUpdate(name="Renamed", device_ids=["headphones", "speaker"]),
        database,
    )
    assert updated.id == created.id
    assert updated.created_at == created.created_at
    assert updated.name == "Renamed"
    assert updated.device_ids == ["headphones", "speaker"]
    assert updated.is_default is False
    assert await channels.get_channel(created.id, database) == updated
    listed = await channels.list_channels(database)
    assert next(channel for channel in listed if channel.id == created.id) == updated
