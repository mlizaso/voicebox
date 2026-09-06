"""Rehearsal sampling must retain identity without ever consuming held-out audio."""

import pytest

from .combine import copy_checked, rehearsal_rows


def test_rehearsal_is_deterministic_balanced_and_retains_parent_reference():
    rows = [
        {"id": f"ch{chapter:02d}_{i:05d}", "chapter": chapter, "split": "train"}
        for chapter in (4, 5, 6)
        for i in range(100)
    ]
    first = rehearsal_rows(rows, 30, 42, "ch04_00077")
    assert first == rehearsal_rows(rows, 30, 42, "ch04_00077")
    assert len({row["id"] for row in first}) == 30
    assert "ch04_00077" in {row["id"] for row in first}
    assert all(9 <= sum(row["chapter"] == chapter for row in first) <= 11 for chapter in (4, 5, 6))
    with pytest.raises(ValueError, match="training rows"):
        rehearsal_rows([{**row, "split": "test"} for row in rows], 30, 42, "ch04_00077")


def test_copy_is_repeatable_but_never_overwrites_changed_audio(tmp_path):
    original, destination = tmp_path / "original", tmp_path / "copy"
    original.write_bytes(b"immutable audio")
    copy_checked(original, destination)
    copy_checked(original, destination)
    destination.write_bytes(b"different")
    with pytest.raises(ValueError, match="differs"):
        copy_checked(original, destination)
    assert destination.read_bytes() == b"different"
    assert original.read_bytes() == b"immutable audio"
