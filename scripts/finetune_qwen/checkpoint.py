"""Crash-safe local optimizer checkpoints and adapter snapshots."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from .corpus import sha256


def durable_json(path: Path, value: dict) -> None:
    """Publish JSON only after its contents are flushed to disk."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path: Path) -> None:
    """Make an atomic rename durable on the local POSIX filesystem."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def cpu_tree(value):
    """Copy optimizer and random states off the GPU for safe serialization."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def adapter_state(model) -> dict[str, torch.Tensor]:
    """Save only the trainable updates; base weights stay immutable."""
    return {name: cpu_tree(value) for name, value in model.named_parameters() if value.requires_grad}


def restore_adapters(model, saved: dict[str, torch.Tensor]) -> None:
    """Require an exact parameter match before restoring any adapter."""
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if saved.keys() != parameters.keys():
        raise ValueError("Checkpoint adapter names do not match this model")
    for name, value in saved.items():
        if value.shape != parameters[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint adapter: {name}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(saved[name])


def publish_checkpoint(directory: Path, state: dict, *, kind: str = "resume") -> Path:
    """Rotate two slots and commit a hash-checked pointer as the last write.

    If interrupted while writing the inactive slot, the previous pointer still
    resolves to the complete checkpoint. Adapter-only best checkpoints use their
    own slots and cannot invalidate the optimizer's recovery point.
    """
    if kind not in {"resume", "best"}:
        raise ValueError("Unknown checkpoint kind")
    if shutil.disk_usage(directory).free < 8 * 2**30:
        raise OSError("Less than 8 GiB free; cannot safely save a training checkpoint")
    pointer = directory / f"{kind}.json"
    previous = json.loads(pointer.read_text()) if pointer.exists() else {}
    slot = 1 - previous.get("slot", 1)
    path = directory / f"{kind}-{slot}.pt"
    with tempfile.NamedTemporaryFile(mode="wb", dir=directory, prefix=f".{kind}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(cpu_tree(state), stream)
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    sync_directory(directory)
    durable_json(pointer, {"slot": slot, "file": path.name, "sha256": sha256(path), "step": state["step"]})
    return path


def read_checkpoint(directory: Path, *, kind: str = "resume") -> dict:
    """Read a committed local checkpoint with PyTorch's restricted loader."""
    if kind not in {"resume", "best"}:
        raise ValueError("Unknown checkpoint kind")
    pointer = json.loads((directory / f"{kind}.json").read_text())
    if pointer["file"] != f"{kind}-{pointer['slot']}.pt" or pointer["slot"] not in (0, 1):
        raise ValueError("Invalid checkpoint pointer")
    path = directory / pointer["file"]
    if sha256(path) != pointer["sha256"]:
        raise ValueError("Checkpoint hash mismatch; preserve both slots for recovery")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["step"] != pointer["step"]:
        raise ValueError("Checkpoint step mismatch")
    return state
