"""Extract Qwen 12 Hz codec targets locally, with resumable per-clip files."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .corpus import atomic_json, sha256


def encode(corpus: Path, model_path: Path, *, limit: int | None = None) -> None:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    if not model_path.is_dir():
        raise ValueError("A local unquantized Qwen model directory is required")
    summary = json.loads((corpus / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Corpus preparation has not completed")
    destination = corpus / "codes"
    destination.mkdir(exist_ok=True)
    identity = {
        "tokenizer_sha256": sha256(model_path / "speech_tokenizer/model.safetensors"),
        "mlx_audio": importlib.metadata.version("mlx-audio"),
        "mlx": importlib.metadata.version("mlx"),
    }
    identity_path = destination / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Tokenizer identity changed; use a new codes directory")
    atomic_json(identity_path, identity)
    model = load_model(str(model_path))
    if model.speech_tokenizer is None or not model.speech_tokenizer.has_encoder:
        raise RuntimeError("Local model did not load the speech encoder")
    tokenizer = model.speech_tokenizer
    del model
    mx.clear_cache()
    completed = 0
    started = time.monotonic()
    for split in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (corpus / f"{split}_raw.jsonl").read_text().splitlines()]
        encoded = []
        for row in rows:
            path = destination / f"{row['id']}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved["audio_sha256"]) != row["sha256"]:
                        raise ValueError(f"Audio identity changed: {row['id']}")
                    codes = saved["codes"]
            else:
                if limit is not None and completed >= limit:
                    print(f"Stopped after {completed} new clips; resume by running the same command", flush=True)
                    return
                if sha256(Path(row["audio"])) != row["sha256"]:
                    raise ValueError(f"Audio contents changed: {row['id']}")
                audio, rate = sf.read(row["audio"], dtype="float32")
                if rate != 24000 or audio.ndim != 1:
                    raise ValueError(f"Expected mono 24 kHz audio: {row['id']}")
                array = tokenizer.encode(mx.array(audio)[None, None, :])
                mx.eval(array)
                codes = np.asarray(array)[0].T.astype(np.int16)
                if codes.ndim != 2 or codes.shape[1] != 16 or not np.all((codes >= 0) & (codes < 2048)):
                    raise ValueError(f"Invalid codec targets: {row['id']}")
                if abs(codes.shape[0] - len(audio) / 1920) > 2:
                    raise ValueError(f"Unexpected codec duration: {row['id']}")
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, codes=codes, audio_sha256=row["sha256"])
                temporary.replace(path)
                completed += 1
                if completed % 25 == 0 or completed <= 3:
                    print(f"encoded={completed} elapsed={time.monotonic() - started:.1f}s last={row['id']}", flush=True)
                mx.clear_cache()
            encoded.append({**row, "audio_codes": codes.tolist()})
        output = corpus / f"{split}_with_codes.jsonl"
        temporary = output.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in encoded), encoding="utf-8")
        temporary.replace(output)
    atomic_json(
        corpus / "encoding-complete.json", {**identity, "new_clips": completed, "seconds": time.monotonic() - started}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    encode(args.corpus, args.model, limit=args.limit)
