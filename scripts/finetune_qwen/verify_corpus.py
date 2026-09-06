"""Independently transcribe exported clips and keep exact text agreements.

Run with the private audiobook environment, which already includes mlx-whisper.
This checks the actual cut WAV, not the original chapter timestamp hypothesis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .corpus import WORD, atomic_json, normalized, sha256


def keys(text: str) -> list[str]:
    return [normalized(match.group()) for match in WORD.finditer(text)]


def verify(corpus: Path, model_path: Path, *, limit: int | None = None) -> None:
    import mlx_whisper

    if not model_path.is_dir():
        raise ValueError("A local independent Whisper checkpoint is required")
    destination = corpus / "verification"
    destination.mkdir(exist_ok=True)
    model_identity = {"path": str(model_path), "weight_sha256": sha256(model_path / "weights.npz")}
    processed = 0
    started = time.monotonic()
    counts = {}
    for split in ("validation", "test", "train"):
        rows = [json.loads(line) for line in (corpus / f"{split}_raw.jsonl").read_text().splitlines()]
        accepted = []
        for row in rows:
            if sha256(Path(row["audio"])) != row["sha256"]:
                raise ValueError(f"Audio changed before verification: {row['id']}")
            path = destination / f"{row['id']}.json"
            if path.exists():
                result = json.loads(path.read_text())
                if (
                    result["audio_sha256"] != row["sha256"]
                    or result["model"] != model_identity
                    or result["expected"] != row["text"]
                ):
                    raise ValueError(f"Verification identity changed: {row['id']}")
            else:
                if limit is not None and processed >= limit:
                    print(f"Stopped after {processed} new clips; rerun to resume", flush=True)
                    return
                asr = mlx_whisper.transcribe(
                    row["audio"],
                    path_or_hf_repo=str(model_path),
                    language="es",
                    temperature=0.0,
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    verbose=None,
                )
                actual = asr["text"].strip()
                result = {
                    "id": row["id"],
                    "audio_sha256": row["sha256"],
                    "model": model_identity,
                    "expected": row["text"],
                    "transcribed": actual,
                    "exact_words": keys(actual) == keys(row["text"]),
                    "words": [word for segment in asr["segments"] for word in segment.get("words", [])],
                }
                atomic_json(path, result)
                processed += 1
                if processed % 25 == 0 or processed <= 3:
                    print(
                        f"verified={processed} elapsed={time.monotonic() - started:.1f}s last={row['id']} exact={result['exact_words']}",
                        flush=True,
                    )
            if result["exact_words"]:
                accepted.append(row)
        output = corpus / f"{split}_verified.jsonl"
        temporary = output.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
        temporary.replace(output)
        counts[split] = {
            "accepted": len(accepted),
            "candidate": len(rows),
            "hours": sum(row["duration"] for row in accepted) / 3600,
        }
    atomic_json(corpus / "verification-summary.json", {"complete": True, "model": model_identity, "splits": counts})
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    verify(args.corpus, args.model, limit=args.limit)
