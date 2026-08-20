"""Resource and integrity tests for the bounded voice-prompt cache."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import torch

from backend import app as backend_app, config
from backend.utils import cache


@pytest.fixture(autouse=True)
def _private_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "_data_dir", tmp_path / "data")
    monkeypatch.setattr(cache, "VOICE_PROMPT_MEMORY_MAX_ENTRIES", 2)
    monkeypatch.setattr(cache, "VOICE_PROMPT_MEMORY_MAX_BYTES", 16 * 1024 * 1024)
    monkeypatch.setattr(cache, "VOICE_PROMPT_DISK_MAX_ENTRIES", 2)
    monkeypatch.setattr(cache, "VOICE_PROMPT_DISK_MAX_BYTES", 16 * 1024 * 1024)
    monkeypatch.setattr(cache, "VOICE_PROMPT_MAX_FILE_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(cache, "VOICE_PROMPT_MIN_FREE_BYTES", 0)
    with cache._cache_lock:
        cache._forget_memory_locked()
    yield
    with cache._cache_lock:
        cache._forget_memory_locked()


def _prompt(value: int) -> dict[str, torch.Tensor]:
    return {"prompt": torch.tensor([value], dtype=torch.float32)}


def test_cache_key_streams_audio_and_binds_reference_text(tmp_path: Path):
    audio = tmp_path / "reference.wav"
    audio.write_bytes((b"voice-reference" * 100_000) + b"tail")

    first = cache.get_cache_key(str(audio), "Original transcript")
    repeated = cache.get_cache_key(str(audio), "Original transcript")
    changed = cache.get_cache_key(str(audio), "Changed transcript")

    assert first == repeated
    assert first != changed
    assert len(first) == 64


def test_memory_and_disk_entries_are_lru_bounded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cache, "VOICE_PROMPT_DISK_MAX_ENTRIES", 1)
    keys = [f"key-{index}" for index in range(3)]

    for index, key in enumerate(keys):
        cache.cache_voice_prompt(key, _prompt(index))

    assert list(cache._memory_cache) == keys[-2:]
    prompt_files = sorted(cache._voice_prompt_cache_dir().glob("*.prompt"))
    assert [path.stem for path in prompt_files] == [keys[-1]]
    loaded = cache.get_cached_voice_prompt(keys[-1])
    assert isinstance(loaded, dict)
    assert torch.equal(loaded["prompt"], _prompt(2)["prompt"])


def test_failed_atomic_write_leaves_no_temporary_or_final_file(monkeypatch: pytest.MonkeyPatch):
    def fail_after_write(_value, stream):
        stream.write(b"partial")
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(cache.torch, "save", fail_after_write)

    cache.cache_voice_prompt("broken", _prompt(1))

    assert not list(cache._voice_prompt_cache_dir().iterdir())


def test_profile_invalidation_preserves_shared_prompt_lru():
    prompt_key = "shared-prompt"
    cache.cache_voice_prompt(prompt_key, _prompt(1))
    cache_dir = config.get_cache_dir()
    selected = cache_dir / "combined_profile-a_current.wav"
    unrelated = cache_dir / "combined_profile-b_current.wav"
    selected.write_bytes(b"selected")
    unrelated.write_bytes(b"unrelated")

    removed = cache.clear_profile_cache("profile-a")

    assert removed == 1
    assert not selected.exists()
    assert unrelated.read_bytes() == b"unrelated"
    assert cache.get_cached_voice_prompt(prompt_key) is not None


def test_global_clear_removes_prompt_and_combined_cache():
    cache.cache_voice_prompt("prompt", _prompt(1))
    combined = config.get_cache_dir() / "combined_profile_current.wav"
    combined.write_bytes(b"combined")

    removed = cache.clear_voice_prompt_cache()

    assert removed == 2
    assert not combined.exists()
    assert not list(cache._voice_prompt_cache_dir().glob("*.prompt"))
    assert not cache._memory_cache


def test_startup_prune_removes_legacy_and_stale_entries():
    cache_dir = config.get_cache_dir()
    legacy = cache_dir / "legacy.prompt"
    legacy.write_bytes(b"legacy")
    prompt_root = cache._voice_prompt_cache_dir()
    stale = prompt_root / ".tmp-interrupted"
    stale.write_bytes(b"partial")

    removed = cache.prune_voice_prompt_cache()

    assert removed == 2
    assert not legacy.exists()
    assert not stale.exists()


def test_application_startup_hook_prunes_prompt_cache():
    stale = cache._voice_prompt_cache_dir() / ".tmp-killed-writer"
    stale.write_bytes(b"partial")

    backend_app._prune_voice_prompt_cache()

    assert not stale.exists()


@pytest.mark.skipif(os.name != "posix", reason="O_NOFOLLOW cache assertion is POSIX-specific")
def test_symlink_cache_entry_never_reads_or_changes_target(tmp_path: Path):
    target = tmp_path / "outside.prompt"
    target.write_bytes(b"outside-private-data")
    cache_file = cache._voice_prompt_cache_dir() / "unsafe.prompt"
    cache_file.symlink_to(target)

    assert cache.get_cached_voice_prompt("unsafe") is None
    assert target.read_bytes() == b"outside-private-data"
    cache.prune_voice_prompt_cache()
    assert target.read_bytes() == b"outside-private-data"
    assert not cache_file.exists()


@pytest.mark.skipif(os.name != "posix", reason="private mode assertion is POSIX-specific")
def test_cache_root_and_entries_are_private():
    cache.cache_voice_prompt("private", _prompt(1))
    root = cache._voice_prompt_cache_dir()
    entry = root / "private.prompt"

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(entry.stat().st_mode) == 0o600
