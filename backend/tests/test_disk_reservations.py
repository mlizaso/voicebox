"""Concurrency-safe admission tests for managed-storage reservations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.utils import disk_reservations


@pytest.fixture(autouse=True)
def _reset_reservations():
    disk_reservations._clear_reservations_for_tests()
    yield
    disk_reservations._clear_reservations_for_tests()


def test_resize_atomically_replaces_future_allocation_and_preserves_old_claim_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        disk_reservations.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1_000),
    )
    first = disk_reservations.reserve_disk_space(tmp_path, 200, min_free_bytes=100)
    second = disk_reservations.reserve_disk_space(tmp_path, 300, min_free_bytes=100)
    try:
        assert disk_reservations.reserved_bytes(tmp_path) == 500

        first.resize(400, directory=tmp_path, min_free_bytes=100)
        assert disk_reservations.reserved_bytes(tmp_path) == 700

        with pytest.raises(disk_reservations.DiskSpaceReservationError, match="free space"):
            first.resize(700, directory=tmp_path, min_free_bytes=100)
        assert disk_reservations.reserved_bytes(tmp_path) == 700

        first.resize(0, directory=tmp_path, min_free_bytes=100)
        assert disk_reservations.reserved_bytes(tmp_path) == 300
        first.resize(500, directory=tmp_path, min_free_bytes=100)
        assert disk_reservations.reserved_bytes(tmp_path) == 800
    finally:
        first.release()
        second.release()
    assert disk_reservations.reserved_bytes(tmp_path) == 0


def test_released_reservation_cannot_be_resized(tmp_path: Path) -> None:
    reservation = disk_reservations.reserve_disk_space(tmp_path, 1, min_free_bytes=0)
    reservation.release()

    with pytest.raises(RuntimeError, match="already released"):
        reservation.resize(2, directory=tmp_path, min_free_bytes=0)
    assert disk_reservations.reserved_bytes(tmp_path) == 0


def test_active_lease_preserves_its_reserve_floor_from_lower_floor_callers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        disk_reservations.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1_000),
    )
    protected = disk_reservations.reserve_disk_space(tmp_path, 100, min_free_bytes=300)
    try:
        with pytest.raises(disk_reservations.DiskSpaceReservationError, match="free space"):
            disk_reservations.reserve_disk_space(tmp_path, 601, min_free_bytes=0)
        assert disk_reservations.reserved_bytes(tmp_path) == 100
    finally:
        protected.release()

    unprotected = disk_reservations.reserve_disk_space(tmp_path, 601, min_free_bytes=0)
    unprotected.release()


def test_shrink_and_release_succeed_after_external_free_space_drop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    free_bytes = 1_000
    monkeypatch.setattr(
        disk_reservations.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=free_bytes),
    )
    reservation = disk_reservations.reserve_disk_space(tmp_path, 500, min_free_bytes=100)

    free_bytes = 0
    reservation.resize(0, directory=tmp_path, min_free_bytes=100)
    assert disk_reservations.reserved_bytes(tmp_path) == 0
    reservation.release()
    assert disk_reservations.reserved_bytes(tmp_path) == 0
