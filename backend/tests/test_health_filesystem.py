"""Filesystem health contract used by long-running audiobook jobs."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from backend.routes import health


def test_filesystem_health_names_every_reported_directory(tmp_path: Path) -> None:
    roots = {
        "generations": tmp_path / "custom-generations",
        "captures": tmp_path / "custom-captures",
        "profiles": tmp_path / "custom-profiles",
        "data": tmp_path / "custom-data",
    }
    for path in roots.values():
        path.mkdir()

    with (
        patch.object(health.config, "get_generations_dir", return_value=roots["generations"]),
        patch.object(health.config, "get_captures_dir", return_value=roots["captures"]),
        patch.object(health.config, "get_profiles_dir", return_value=roots["profiles"]),
        patch.object(health.config, "get_data_dir", return_value=roots["data"]),
    ):
        result = asyncio.run(health.filesystem_health())

    assert {item.name: item.path for item in result.directories} == {
        name: str(path.resolve()) for name, path in roots.items()
    }


def test_concurrent_writability_probes_use_independent_files(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: health._probe_directory_writable(tmp_path), range(32)))

    assert results == [None] * 32
    assert list(tmp_path.iterdir()) == []
