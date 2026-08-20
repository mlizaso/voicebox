"""Process-wide free-space reservations for managed storage writes.

The Voicebox data-root lifetime lock guarantees a single server process owns a
managed root. This ledger closes the remaining in-process check/write race:
each long operation reserves its complete future allocation before it
writes, and concurrent admission subtracts all outstanding reservations on the
same filesystem.
"""

from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass
from pathlib import Path


class DiskSpaceReservationError(OSError):
    """A managed write cannot preserve its requested free-space reserve."""


_reservation_lock = threading.RLock()
_reserved_bytes_by_device: dict[int, int] = {}
_reserve_floor_counts_by_device: dict[int, dict[int, int]] = {}


def _filesystem_device(directory: Path) -> int:
    try:
        return directory.stat().st_dev
    except OSError as exc:
        raise DiskSpaceReservationError("Could not identify the managed storage filesystem") from exc


def _validate_byte_count(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")


def _free_bytes(directory: Path) -> int:
    try:
        return shutil.disk_usage(directory).free
    except OSError as exc:
        raise DiskSpaceReservationError("Could not determine managed storage capacity") from exc


def _add_reserve_floor(device: int, min_free_bytes: int) -> None:
    counts = _reserve_floor_counts_by_device.setdefault(device, {})
    counts[min_free_bytes] = counts.get(min_free_bytes, 0) + 1


def _remove_reserve_floor(device: int, min_free_bytes: int) -> None:
    counts = _reserve_floor_counts_by_device.get(device)
    if counts is None or counts.get(min_free_bytes, 0) <= 0:
        raise RuntimeError("Disk-space reservation ledger is inconsistent")
    remaining = counts[min_free_bytes] - 1
    if remaining:
        counts[min_free_bytes] = remaining
    else:
        counts.pop(min_free_bytes)
    if not counts:
        _reserve_floor_counts_by_device.pop(device, None)


def _other_reserve_floor(device: int, own_min_free_bytes: int | None = None) -> int:
    counts = _reserve_floor_counts_by_device.get(device, {})
    floor = 0
    for min_free_bytes, count in counts.items():
        if own_min_free_bytes == min_free_bytes:
            count -= 1
        if count > 0:
            floor = max(floor, min_free_bytes)
    return floor


@dataclass(slots=True)
class DiskSpaceReservation:
    """One idempotently releasable future-allocation claim."""

    device: int
    bytes: int
    min_free_bytes: int
    _released: bool = False

    def resize(
        self,
        required_bytes: int,
        *,
        directory: Path,
        min_free_bytes: int,
    ) -> None:
        """Atomically replace this lease's remaining future-allocation claim.

        A caller may shrink only after the consumed allocation is visible to
        ``disk_usage`` (normally after flush/fsync/close) or has been removed.
        A failed growth leaves the existing claim unchanged.
        """
        _validate_byte_count(required_bytes, "required_bytes")
        _validate_byte_count(min_free_bytes, "min_free_bytes")
        directory = Path(directory)
        device = _filesystem_device(directory)
        if device != self.device:
            raise DiskSpaceReservationError("A disk-space reservation cannot move between filesystems")
        with _reservation_lock:
            if self._released:
                raise RuntimeError("Disk-space reservation is already released")
            reserved = _reserved_bytes_by_device.get(self.device, 0)
            if reserved < self.bytes:
                raise RuntimeError("Disk-space reservation ledger is inconsistent")
            other_reserved = reserved - self.bytes
            reserve_floor = max(
                min_free_bytes,
                _other_reserve_floor(self.device, self.min_free_bytes),
            )
            if required_bytes > self.bytes or min_free_bytes > self.min_free_bytes:
                free_bytes = _free_bytes(directory)
                if free_bytes - other_reserved - required_bytes < reserve_floor:
                    raise DiskSpaceReservationError(
                        "Insufficient free space for managed storage while preserving the disk reserve"
                    )
            replacement_total = other_reserved + required_bytes
            if replacement_total:
                _reserved_bytes_by_device[self.device] = replacement_total
            else:
                _reserved_bytes_by_device.pop(self.device, None)
            if min_free_bytes != self.min_free_bytes:
                _remove_reserve_floor(self.device, self.min_free_bytes)
                _add_reserve_floor(self.device, min_free_bytes)
            self.bytes = required_bytes
            self.min_free_bytes = min_free_bytes

    def release(self) -> None:
        if self._released:
            return
        with _reservation_lock:
            if self._released:
                return
            reserved = _reserved_bytes_by_device.get(self.device, 0)
            remaining = reserved - self.bytes
            if remaining > 0:
                _reserved_bytes_by_device[self.device] = remaining
            else:
                _reserved_bytes_by_device.pop(self.device, None)
            _remove_reserve_floor(self.device, self.min_free_bytes)
            self._released = True

    def __enter__(self) -> DiskSpaceReservation:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()


def reserve_disk_space(
    directory: Path,
    required_bytes: int,
    *,
    min_free_bytes: int,
) -> DiskSpaceReservation:
    """Atomically reserve future bytes on *directory*'s filesystem.

    The caller must retain the returned lease until every covered write has
    finished. Before release, the allocation must either be visible to
    ``disk_usage`` or have been removed, so another admission cannot spend the
    same bytes.
    """
    _validate_byte_count(required_bytes, "required_bytes")
    _validate_byte_count(min_free_bytes, "min_free_bytes")
    directory = Path(directory)
    device = _filesystem_device(directory)
    with _reservation_lock:
        free_bytes = _free_bytes(directory)
        already_reserved = _reserved_bytes_by_device.get(device, 0)
        reserve_floor = max(min_free_bytes, _other_reserve_floor(device))
        if free_bytes - already_reserved - required_bytes < reserve_floor:
            raise DiskSpaceReservationError(
                "Insufficient free space for managed storage while preserving the disk reserve"
            )
        if required_bytes:
            _reserved_bytes_by_device[device] = already_reserved + required_bytes
        _add_reserve_floor(device, min_free_bytes)
        return DiskSpaceReservation(
            device=device,
            bytes=required_bytes,
            min_free_bytes=min_free_bytes,
        )


def reserved_bytes(directory: Path) -> int:
    """Return the current reservation total for diagnostics and tests."""
    device = _filesystem_device(Path(directory))
    with _reservation_lock:
        return _reserved_bytes_by_device.get(device, 0)


def _clear_reservations_for_tests() -> None:
    """Reset process state between tests that deliberately interrupt leases."""
    with _reservation_lock:
        _reserved_bytes_by_device.clear()
        _reserve_floor_counts_by_device.clear()
