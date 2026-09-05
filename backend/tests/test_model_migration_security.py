"""Model migration must not follow redirected roots or delete destination data."""

import pytest
from fastapi import HTTPException

from backend.routes import models


def test_move_refuses_existing_destination_and_keeps_both_caches(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "model").write_bytes(b"source model")
    (destination / "model").write_bytes(b"existing model")
    with pytest.raises(FileExistsError):
        models._move_model_cache_directory(source, destination)
    assert (source / "model").read_bytes() == b"source model"
    assert (destination / "model").read_bytes() == b"existing model"


@pytest.mark.asyncio
@pytest.mark.parametrize("destination_kind", ["symlink", "file"])
async def test_migration_rejects_invalid_destination_before_starting(tmp_path, monkeypatch, destination_kind):
    from huggingface_hub import constants

    source = tmp_path / "source"
    outside = tmp_path / "outside"
    source.mkdir()
    outside.mkdir()
    destination = tmp_path / "redirect"
    if destination_kind == "symlink":
        destination.symlink_to(outside, target_is_directory=True)
    else:
        destination.write_bytes(b"existing data")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(source))
    monkeypatch.setattr(models, "create_background_task", lambda *_args: pytest.fail("Unsafe migration started"))
    with pytest.raises(HTTPException) as raised:
        await models.migrate_models(models.models.ModelMigrateRequest(destination=str(destination)))
    assert raised.value.status_code == 400
    assert list(outside.iterdir()) == []
