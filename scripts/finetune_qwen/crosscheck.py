"""Independently rescore fixed paired audio without replacing its original report."""

import argparse
import json
from pathlib import Path

from .corpus import atomic_json, sha256
from .evaluate import inference_source_identity, reliable_transcription, summarize, transcribe_audio, word_error


def recognizer_weights(model: Path) -> Path:
    """Match mlx-whisper's safetensors-first local loader exactly."""
    safe = model / "weights.safetensors"
    return safe if safe.is_file() else model / "weights.npz"


def validate_crosscheck(report: dict) -> None:
    """Bind a secondary report to the complete original and actual ASR weights."""
    identity = report["crosscheck"]
    source = Path(identity["generation_summary"])
    primary = Path(identity["primary_evaluation"])
    model = Path(identity["model"])
    if (
        identity["code_sha256"] != sha256(Path(__file__))
        or identity["generation_sha256"] != sha256(source)
        or identity["primary_sha256"] != sha256(primary)
        or identity["model_sha256"] != sha256(recognizer_weights(model))
        or identity["model_config_sha256"] != sha256(model / "config.json")
        or report["identity"] != json.loads(source.read_text())["identity"]
        or report["identity"] != json.loads(primary.read_text())["identity"]
    ):
        raise ValueError("Independent cross-check provenance changed")


def crosscheck(generation_dir: Path, model: Path, output: Path) -> None:
    """Score every paired sample with one other recognizer, preserving all evidence."""
    if output.resolve() == generation_dir.resolve():
        raise ValueError("A cross-check must preserve the original evaluation directory")
    generation_path = generation_dir / "generation-summary.json"
    primary_path = generation_dir / "evaluation.json"
    generation = json.loads(generation_path.read_text())
    primary = json.loads(primary_path.read_text())
    if (
        generation["identity"] != primary["identity"]
        or generation["identity"]["evaluation_code_sha256"] != sha256(Path(__file__).with_name("evaluate.py"))
        or generation["identity"]["inference_sources"] != inference_source_identity()
    ):
        raise ValueError("The original generation/evaluation implementation or identity changed")
    identity = {
        "generation_summary": str(generation_path.resolve()),
        "generation_sha256": sha256(generation_path),
        "primary_evaluation": str(primary_path.resolve()),
        "primary_sha256": sha256(primary_path),
        "model": str(model.resolve()),
        "model_sha256": sha256(recognizer_weights(model)),
        "model_config_sha256": sha256(model / "config.json"),
        "code_sha256": sha256(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Cross-check identity changed; use a new output directory")
    atomic_json(identity_path, identity)
    scored = []
    for row in generation["results"]:
        if sha256(Path(row["audio"])) != row["sha256"]:
            raise ValueError("The original generated audio changed")
        cache = output / f"asr-{row['variant']}-{row['id']}.json"
        if cache.exists():
            result = json.loads(cache.read_text())
            if result["audio_sha256"] != row["sha256"] or result["identity"] != identity:
                raise ValueError("Cross-check ASR identity changed")
        else:
            result = {
                **transcribe_audio(row["audio"], str(model)),
                "audio_sha256": row["sha256"],
                "identity": identity,
            }
            atomic_json(cache, result)
        if result["reliable"] is not True or not reliable_transcription(result["attempts"][-1]["result"]):
            raise ValueError(f"Unresolved cross-check transcription: {row['variant']}/{row['id']}")
        edits, words = word_error(row["text"], result["text"])
        scored.append(
            {**row, "transcribed": result["text"], "word_errors": edits, "words": words, "wer": edits / words}
        )
    report = {**summarize(scored), "identity": generation["identity"], "crosscheck": identity, "results": scored}
    atomic_json(output / "evaluation.json", report)
    print(json.dumps({key: report[key] for key in ("accepted", "gates", "aggregates")}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_dir", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    crosscheck(args.generation_dir, args.model, args.output)
