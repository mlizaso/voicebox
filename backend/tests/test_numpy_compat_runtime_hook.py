"""PyInstaller NumPy/Torch compatibility-hook regressions."""

import runpy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

RUNTIME_HOOK = Path(__file__).parents[1] / "pyi_rth_numpy_compat.py"


def test_runtime_hook_installs_explicit_dtype_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook must not silently abandon the NumPy 2.x compatibility patch."""

    def incompatible_from_numpy(_array):
        raise RuntimeError("Numpy is not available")

    dtype_names = (
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "bool",
        "complex64",
        "complex128",
    )
    fake_torch = SimpleNamespace(
        from_numpy=incompatible_from_numpy,
        **{name: object() for name in dtype_names},
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    runpy.run_path(str(RUNTIME_HOOK))

    deadline = time.monotonic() + 1
    while not getattr(fake_torch, "_vb_from_numpy_patched", False) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert fake_torch._vb_from_numpy_patched is True
    with pytest.raises(TypeError, match="unsupported numpy dtype 'object'"):
        fake_torch.from_numpy(np.array([object()], dtype=object))
