"""Merge the selected adapters into a standalone BF16 Qwen CustomVoice model."""

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .checkpoint import durable_json, read_checkpoint, sync_directory
from .corpus import sha256
from .data import base_identity


def merge_weights(weights: dict, adapters: dict, speaker: torch.Tensor, *, speaker_id: int = 3000) -> dict:
    """Merge FP32 adapter updates once, retaining the original BF16 precision."""
    targets = {name.removesuffix(".adapter_a") for name in adapters if name.endswith(".adapter_a")}
    expected = {f"{name}.{part}" for name in targets for part in ("adapter_a", "adapter_b")}
    if not targets or set(adapters) != expected:
        raise ValueError("Incomplete adapter pairs")
    for name in sorted(targets):
        key = f"{name}.weight"
        a, b = adapters[f"{name}.adapter_a"], adapters[f"{name}.adapter_b"]
        base = weights[key]
        if base.dtype != torch.bfloat16 or a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Expected a BF16 base and matrix adapters: {name}")
        if a.shape[0] != b.shape[1] or (b.shape[0], a.shape[1]) != base.shape:
            raise ValueError(f"Adapter dimensions do not match: {name}")
        merged = base.float() + 2.0 * (b.float() @ a.float())
        if not torch.isfinite(merged).all():
            raise ValueError(f"Nonfinite merged weight: {name}")
        weights[key] = merged.to(torch.bfloat16).contiguous()
    embedding = weights["talker.model.codec_embedding.weight"].clone()
    if (
        not 0 <= speaker_id < len(embedding)
        or speaker.numel() != embedding.shape[1]
        or not torch.isfinite(speaker).all()
    ):
        raise ValueError("Invalid fixed speaker embedding")
    embedding[speaker_id] = speaker.reshape(-1).to(dtype=embedding.dtype)
    weights["talker.model.codec_embedding.weight"] = embedding
    return {name: value for name, value in weights.items() if not name.startswith("speaker_encoder.")}


def retained_checkpoint(run: Path, step: int) -> tuple[dict, dict]:
    """Read a retained best snapshot for validation-audio selection, without repointing a run."""
    if step < 1 or json.loads((run / "status.json").read_text()).get("event") != "complete":
        raise ValueError("An explicit retained checkpoint requires a completed run and positive step")
    contract = json.loads((run / "contract.json").read_text())
    selected = []
    for slot in (0, 1):
        path = run / f"best-{slot}.pt"
        if not path.is_file():
            continue
        identity = {"file": path.name, "sha256": sha256(path)}
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if sha256(path) != identity["sha256"]:
            raise ValueError("Retained checkpoint changed while reading")
        if saved["step"] == step:
            if saved["contract"] != contract or saved["best_step"] != step:
                raise ValueError("Retained checkpoint contract or selection mismatch")
            selected.append((saved, identity))
    if len(selected) != 1:
        raise ValueError("The requested best checkpoint is not uniquely retained; preserve existing snapshots")
    return selected[0]


def export(base: Path, run: Path, output: Path, *, speaker_name: str = "fabian", step: int | None = None) -> None:
    """Create a complete local checkpoint without overwriting an existing export."""
    if output.exists():
        raise FileExistsError(f"Export already exists: {output}")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", speaker_name):
        raise ValueError("Use a lowercase speaker identifier")
    selection = None
    if step is None:
        saved = read_checkpoint(run, kind="best")
    else:
        saved, selection = retained_checkpoint(run, step)
    contract = saved["contract"]
    if saved["step"] < 1 or contract["base"] != base_identity(base):
        raise ValueError("The selected checkpoint is untrained or belongs to another base")
    if (
        not math.isfinite(saved["best_loss"])
        or not math.isfinite(saved["baseline"]["loss"])
        or saved["best_loss"] >= saved["baseline"]["loss"]
    ):
        raise ValueError("The selected checkpoint did not improve held-out validation loss")
    if shutil.disk_usage(output.parent).free < 12 * 2**30:
        raise OSError("Less than 12 GiB free; cannot safely export the model")
    torch.set_num_threads(4)
    config = json.loads((base / "config.json").read_text())
    if config.get("quantization") or config.get("tts_model_type") != "base":
        raise ValueError("Export requires an unquantized Qwen Base checkpoint")
    weights = merge_weights(load_file(base / "model.safetensors"), saved["adapters"], saved["speaker"])
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    # A failed export remains in this private staging directory for inspection;
    # it is never discoverable as an installed voice until the final rename.
    print(f"Writing merged BF16 checkpoint to {temporary}", flush=True)
    save_file(weights, temporary / "model.safetensors", metadata={"format": "pt"})
    del weights
    config["tts_model_type"] = "custom_voice"
    config["talker_config"]["spk_id"] = {speaker_name: 3000}
    config["talker_config"]["spk_is_dialect"] = {speaker_name: False}
    durable_json(temporary / "config.json", config)
    for name in (
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "generation_config.json",
    ):
        shutil.copy2(base / name, temporary / name)
    shutil.copytree(base / "speech_tokenizer", temporary / "speech_tokenizer")
    files = {str(path.relative_to(temporary)): sha256(path) for path in sorted(temporary.rglob("*")) if path.is_file()}
    metadata = {
        "format": 1,
        "speaker": speaker_name,
        "language": "es",
        "model_size": "1.7B",
        "dtype": "bfloat16",
        "method": "merged_lora",
        "step": saved["step"],
        "validation_loss": saved["best_loss"],
        "baseline_loss": saved["baseline"]["loss"],
        "contract_sha256": sha256(run / "contract.json"),
        "files": files,
    }
    if selection is not None:
        metadata["retained_checkpoint"] = selection
    durable_json(temporary / "voicebox_finetune.json", metadata)
    # Flush the weights and copied tokenizer before publishing the directory.
    for path in temporary.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    sync_directory(temporary / "speech_tokenizer")
    sync_directory(temporary)
    os.rename(temporary, output)
    sync_directory(output.parent)
    print(json.dumps({"output": str(output), "step": saved["step"], "validation_loss": saved["best_loss"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--speaker-name", default="fabian")
    parser.add_argument("--step", type=int, help="Retained earlier best snapshot from a completed run for validation")
    args = parser.parse_args()
    export(args.base, args.run, args.output, speaker_name=args.speaker_name, step=args.step)
