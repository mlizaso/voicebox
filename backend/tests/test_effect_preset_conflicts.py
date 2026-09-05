"""Preset conflicts must not poison the session or partially update a preset."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database import Base
from backend.models import EffectPresetCreate, EffectPresetUpdate
from backend.services import effects


def test_rename_conflict_rolls_back_and_allows_retry():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        first = effects.create_preset(EffectPresetCreate(name="First", effects_chain=[]), db)
        second = effects.create_preset(EffectPresetCreate(name="Second", effects_chain=[]), db)

        with pytest.raises(ValueError, match="already exists"):
            effects.update_preset(second.id, EffectPresetUpdate(name=first.name, description="Should roll back"), db)

        unchanged = effects.get_preset(second.id, db)
        assert unchanged.name == "Second"
        assert unchanged.description == second.description
        updated = effects.update_preset(second.id, EffectPresetUpdate(name="Third"), db)
        assert updated.name == "Third"
        assert effects.get_preset(first.id, db).name == "First"
