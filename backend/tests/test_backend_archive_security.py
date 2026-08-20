"""Adversarial release-archive tests shared by the CUDA and ROCm services."""

import asyncio
import io
import tarfile
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from backend.services import cuda, rocm
from backend.utils import backend_archive, disk_reservations
from backend.utils.backend_archive import BackendArchiveError, extract_backend_tar_archive


class _Response:
    def __init__(self, content: bytes):
        self.content = content
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self, chunk_size: int = 1024) -> AsyncIterator[bytes]:
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]


class _Stream:
    def __init__(self, response: _Response):
        self.response = response

    async def __aenter__(self) -> _Response:
        return self.response

    async def __aexit__(self, *_args) -> bool:
        return False


class _Client:
    def __init__(self, content: bytes):
        self.response = _Response(content)

    def stream(self, _method: str, _url: str) -> _Stream:
        return _Stream(self.response)


class _BlockingResponse(_Response):
    def __init__(self, content: bytes, entered: asyncio.Event):
        super().__init__(content)
        self.entered = entered

    async def aiter_bytes(self, chunk_size: int = 1024) -> AsyncIterator[bytes]:
        self.entered.set()
        await asyncio.Event().wait()
        yield self.content[:chunk_size]


class _BlockingClient(_Client):
    def __init__(self, content: bytes, entered: asyncio.Event):
        self.response = _BlockingResponse(content, entered)


class _CancelledChecksumClient(_Client):
    async def get(self, _url: str):
        raise asyncio.CancelledError


def _tar_bytes(name: str, *, kind: bytes = tarfile.REGTYPE, payload: bytes = b"payload") -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.type = kind
        if kind == tarfile.REGTYPE:
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        else:
            info.size = 0
            info.linkname = "../outside-target"
            archive.addfile(info)
    return output.getvalue()


async def _download_archive(service: ModuleType, content: bytes, destination: Path) -> None:
    await service._download_and_extract_archive(
        _Client(content),
        url="https://example.invalid/backend.tar.gz",
        sha256_url=None,
        dest_dir=destination,
        label="test backend",
        progress_offset=0,
        total_size=len(content),
    )


@pytest.mark.parametrize("service", [cuda, rocm], ids=["cuda", "rocm"])
@pytest.mark.parametrize(
    ("member_name", "kind"),
    [
        ("../outside", tarfile.REGTYPE),
        ("/absolute-outside", tarfile.REGTYPE),
        ("unsafe-link", tarfile.SYMTYPE),
        ("unsafe-hardlink", tarfile.LNKTYPE),
        ("unsafe-device", tarfile.CHRTYPE),
    ],
    ids=["traversal", "absolute", "symlink", "hardlink", "device"],
)
@pytest.mark.asyncio
async def test_download_rejects_unsafe_archive_members_without_writes(
    service: ModuleType,
    member_name: str,
    kind: bytes,
    tmp_path: Path,
):
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(BackendArchiveError):
        await _download_archive(service, _tar_bytes(member_name, kind=kind), destination)

    assert list(destination.iterdir()) == []
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("service", [cuda, rocm], ids=["cuda", "rocm"])
@pytest.mark.asyncio
async def test_download_refuses_preexisting_symlink_parent(
    service: ModuleType,
    tmp_path: Path,
):
    destination = tmp_path / "destination"
    outside = tmp_path / "outside"
    destination.mkdir()
    outside.mkdir()
    (destination / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(BackendArchiveError, match="unsafe"):
        await _download_archive(service, _tar_bytes("nested/payload.bin"), destination)

    assert not (outside / "payload.bin").exists()


@pytest.mark.parametrize("service", [cuda, rocm], ids=["cuda", "rocm"])
@pytest.mark.asyncio
async def test_download_enforces_compressed_archive_limit(
    service: ModuleType,
    monkeypatch,
    tmp_path: Path,
):
    destination = tmp_path / "destination"
    destination.mkdir()
    monkeypatch.setattr(service, "BACKEND_ARCHIVE_MAX_COMPRESSED_BYTES", 4)

    with pytest.raises(BackendArchiveError, match="compressed archive size limit"):
        await _download_archive(service, _tar_bytes("payload.bin"), destination)

    assert list(destination.iterdir()) == []


def test_extraction_enforces_member_and_total_size_limits(tmp_path: Path):
    archive_path = tmp_path / "backend.tar.gz"
    archive_path.write_bytes(_tar_bytes("payload.bin", payload=b"12345"))

    with pytest.raises(BackendArchiveError, match="member exceeds"):
        extract_backend_tar_archive(
            archive_path,
            tmp_path / "member-limit",
            max_member_bytes=4,
        )
    with pytest.raises(BackendArchiveError, match="total extracted-size"):
        extract_backend_tar_archive(
            archive_path,
            tmp_path / "total-limit",
            max_total_bytes=4,
        )


def test_extraction_rejects_empty_archive_before_creating_destination(tmp_path: Path):
    archive_path = tmp_path / "empty.tar.gz"
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz"):
        pass
    archive_path.write_bytes(output.getvalue())
    destination = tmp_path / "destination"

    with pytest.raises(BackendArchiveError, match="no members"):
        extract_backend_tar_archive(archive_path, destination)

    assert not destination.exists()


def test_extraction_preserves_disk_reserve_before_writing_members(monkeypatch, tmp_path: Path):
    archive_path = tmp_path / "backend.tar.gz"
    archive_path.write_bytes(_tar_bytes("payload.bin", payload=b"12345"))
    destination = tmp_path / "destination"
    destination.mkdir()
    monkeypatch.setattr(backend_archive.shutil, "disk_usage", lambda _path: SimpleNamespace(free=5))

    with pytest.raises(BackendArchiveError, match="Insufficient free space"):
        extract_backend_tar_archive(archive_path, destination)

    assert list(destination.iterdir()) == []


def test_extraction_streams_multiple_valid_members_after_full_preflight(tmp_path: Path):
    archive_path = tmp_path / "backend.tar.gz"
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        directory = tarfile.TarInfo("runtime")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for name, payload in (("runtime/one.bin", b"one"), ("runtime/two.bin", b"two")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    archive_path.write_bytes(output.getvalue())
    destination = tmp_path / "destination"

    extract_backend_tar_archive(archive_path, destination)

    assert (destination / "runtime" / "one.bin").read_bytes() == b"one"
    assert (destination / "runtime" / "two.bin").read_bytes() == b"two"


@pytest.mark.parametrize("service", [cuda, rocm], ids=["cuda", "rocm"])
@pytest.mark.asyncio
async def test_cancelled_extraction_is_drained_before_operation_release(
    service: ModuleType,
    monkeypatch,
    tmp_path: Path,
):
    if service is cuda:
        monkeypatch.setattr(cuda.sys, "platform", "win32")
    monkeypatch.setattr(service, "get_backends_dir", lambda: tmp_path / "backends")

    content = _tar_bytes("payload.bin")
    extraction_started = threading.Event()
    allow_extraction_to_finish = threading.Event()
    real_extract = extract_backend_tar_archive

    def blocking_extract(archive_path: Path, destination: Path, *, cancel_event=None) -> None:
        extraction_started.set()
        if not allow_extraction_to_finish.wait(timeout=5):
            raise TimeoutError("test did not release extraction")
        real_extract(archive_path, destination, cancel_event=cancel_event)

    monkeypatch.setattr(service, "extract_backend_tar_archive", blocking_extract)
    destination = service.get_backends_dir() / "staging"

    async def download_locked(_version=None):
        destination.mkdir(parents=True)
        await _download_archive(service, content, destination)

    locked_name = "_download_cuda_binary_locked" if service is cuda else "_download_rocm_binary_locked"
    monkeypatch.setattr(service, locked_name, download_locked)

    download = asyncio.create_task(
        service.download_cuda_binary() if service is cuda else service.download_rocm_binary()
    )
    assert await asyncio.to_thread(extraction_started.wait, 2)
    download.cancel()
    await asyncio.sleep(0)
    assert not download.done()
    assert service._active_operation_name() == "download"

    allow_extraction_to_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await download
    assert service._active_operation_name() is None
    assert not list(destination.glob(".download-*.tmp"))


@pytest.mark.parametrize("service", [cuda, rocm], ids=["cuda", "rocm"])
@pytest.mark.asyncio
async def test_cancelled_checksum_setup_releases_local_disk_reservation(
    service: ModuleType,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    disk_reservations._clear_reservations_for_tests()
    try:
        with pytest.raises(asyncio.CancelledError):
            await service._download_and_extract_archive(
                _CancelledChecksumClient(_tar_bytes("payload.bin")),
                url="https://example.invalid/backend.tar.gz",
                sha256_url="https://example.invalid/backend.tar.gz.sha256",
                dest_dir=destination,
                label="cancelled setup",
                progress_offset=0,
                total_size=1,
            )
        assert disk_reservations.reserved_bytes(destination) == 0
    finally:
        disk_reservations._clear_reservations_for_tests()


@pytest.mark.asyncio
async def test_cuda_and_rocm_downloads_share_one_filesystem_reservation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = _tar_bytes("payload.bin")
    available_bytes = len(content) * 2 - 1
    monkeypatch.setattr(
        disk_reservations.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=available_bytes),
    )
    monkeypatch.setattr(cuda, "BACKEND_ARCHIVE_MIN_FREE_BYTES", 0)
    monkeypatch.setattr(rocm, "BACKEND_ARCHIVE_MIN_FREE_BYTES", 0)
    disk_reservations._clear_reservations_for_tests()

    cuda_destination = tmp_path / "cuda-staging"
    rocm_destination = tmp_path / "rocm-staging"
    cuda_destination.mkdir()
    rocm_destination.mkdir()
    cuda_reservation = disk_reservations.reserve_disk_space(
        cuda_destination,
        0,
        min_free_bytes=0,
    )
    rocm_reservation = disk_reservations.reserve_disk_space(
        rocm_destination,
        0,
        min_free_bytes=0,
    )
    entered = asyncio.Event()
    cuda_download = asyncio.create_task(
        cuda._download_and_extract_archive(
            _BlockingClient(content, entered),
            url="https://example.invalid/cuda.tar.gz",
            sha256_url=None,
            dest_dir=cuda_destination,
            label="CUDA test",
            progress_offset=0,
            total_size=len(content),
            storage_reservation=cuda_reservation,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert disk_reservations.reserved_bytes(tmp_path) == len(content)

        with pytest.raises(BackendArchiveError, match="shared capacity"):
            await rocm._download_and_extract_archive(
                _Client(content),
                url="https://example.invalid/rocm.tar.gz",
                sha256_url=None,
                dest_dir=rocm_destination,
                label="ROCm test",
                progress_offset=0,
                total_size=len(content),
                storage_reservation=rocm_reservation,
            )

        assert disk_reservations.reserved_bytes(tmp_path) == len(content)
    finally:
        cuda_download.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cuda_download
        cuda_reservation.release()
        rocm_reservation.release()
        disk_reservations._clear_reservations_for_tests()

    assert disk_reservations.reserved_bytes(tmp_path) == 0
