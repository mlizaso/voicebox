"""Storage-path regressions for custom Voicebox data roots."""

from pathlib import Path

from backend import config


def test_custom_root_below_data_component_does_not_duplicate_path(tmp_path: Path):
    original = config.get_data_dir()
    custom_root = tmp_path / "data" / "custom"
    sample = custom_root / "profiles" / "profile-id" / "sample.wav"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(b"wav")

    try:
        config.set_data_dir(custom_root)

        stored = config.to_storage_path(sample)

        assert stored == "profiles/profile-id/sample.wav"
        assert config.resolve_storage_path(stored) == sample.resolve()
        assert config.resolve_storage_path(sample) == sample.resolve()
    finally:
        config.set_data_dir(original)


def test_absolute_legacy_data_path_still_rebases_to_current_root(tmp_path: Path):
    original = config.get_data_dir()
    custom_root = tmp_path / "current"

    try:
        config.set_data_dir(custom_root)

        resolved = config.resolve_storage_path("/retired/voicebox/data/profiles/profile-id/sample.wav")

        assert resolved == custom_root / "profiles" / "profile-id" / "sample.wav"
    finally:
        config.set_data_dir(original)
