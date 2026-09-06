"""Install an evaluated local narrator without replacing existing voice profiles."""

import argparse
import asyncio
import fcntl
import json
import re
from pathlib import Path

from .checkpoint import durable_json
from .corpus import sha256
from .data import group_key
from .evaluate import inference_source_identity, summarize, word_error


def checked_registration(model: Path, evaluation: Path, slug: str, name: str) -> dict:
    """Require a complete passing test evaluation of these exact model bytes."""
    from backend.services.finetuned_voices import validate_checkpoint

    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", slug):
        raise ValueError("Use a lowercase voice identifier, at most 48 characters")
    if not name.strip() or len(name) > 100:
        raise ValueError("A voice name between 1 and 100 characters is required")
    model = model.resolve(strict=True)
    manifest_path = model / "voicebox_finetune.json"
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(evaluation.read_text())
    if "crosscheck" in report:
        from .crosscheck import validate_crosscheck

        validate_crosscheck(report)
    identity = report["identity"]
    if identity.get("inference_sources") != inference_source_identity() or identity.get(
        "evaluation_code_sha256"
    ) != sha256(Path(__file__).with_name("evaluate.py")):
        raise ValueError("The evaluated inference implementation changed; repeat the audio evaluation")
    if (
        identity["config"]["split"] != "test"
        or Path(identity["config"]["candidate"]).resolve() != model
        or identity["manifest_sha256"] != sha256(manifest_path)
        or identity["candidate_sha256"] != sha256(model / "model.safetensors")
    ):
        raise ValueError("The final test evaluation does not belong to this checkpoint")
    cases = {row["id"]: row for row in identity["cases"]}
    if len(cases) != len(identity["cases"]) or len(cases) < 8:
        raise ValueError("A complete held-out and novel-prose comparison is required")
    chapters = {group_key(row) for row in cases.values() if row["chapter"] is not None}
    if len(chapters) < 2 or sum(row["chapter"] is None for row in cases.values()) < 4:
        raise ValueError("Evaluation requires two held-out chapters and four new prose passages")
    expected = {(variant, identifier) for variant in ("candidate", "baseline") for identifier in cases}
    seen = set()
    for row in report["results"]:
        key = row["variant"], row["id"]
        if key not in expected or key in seen:
            raise ValueError("Incomplete or duplicate evaluation samples")
        seen.add(key)
        case = cases[row["id"]]
        if any(row[field] != case[field] for field in ("text", "seed", "chapter")):
            raise ValueError("Evaluation passages changed")
        if row.get("source") != case.get("source"):
            raise ValueError("Evaluation source groups changed")
        if sha256(Path(row["audio"])) != row["sha256"]:
            raise ValueError("Evaluated audio changed")
        edits, words = word_error(row["text"], row["transcribed"])
        if not words or (edits, words, edits / words) != (row["word_errors"], row["words"], row["wer"]):
            raise ValueError("Evaluation transcript scores changed")
    if seen != expected:
        raise ValueError("Evaluation is missing paired samples")
    summary = summarize(report["results"])
    if not summary["accepted"] or any(report[key] != value for key, value in summary.items()):
        raise ValueError("This checkpoint did not pass every quality and speed gate")
    spec = {
        "voice_id": f"finetuned:{slug}",
        "name": name,
        "speaker": manifest["speaker"],
        "gender": "male",
        "model_path": str(model),
        "manifest_sha256": sha256(manifest_path),
        "evaluation_path": str(evaluation.resolve()),
        "evaluation_sha256": sha256(evaluation),
    }
    validate_checkpoint(spec, verify_weights=True)
    return spec


async def register_profile(spec: dict, db) -> dict:
    """Idempotently publish one operator registration and normal preset profile."""
    from backend.database.models import VoiceProfile
    from backend.models import VoiceProfileCreate
    from backend.services.finetuned_voices import registry_directory
    from backend.services.profiles import _profile_to_response, create_profile

    directory = registry_directory()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / ".install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / f"{spec['voice_id'].removeprefix('finetuned:')}.json"
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError("That voice identifier already names a different installation; choose a new identifier")
        matches = (
            db.query(VoiceProfile)
            .filter_by(voice_type="preset", preset_engine="qwen_custom_voice", preset_voice_id=spec["voice_id"])
            .all()
        )
        if len(matches) > 1:
            raise ValueError("Multiple existing profiles use this voice; no profiles were changed")
        if not matches and db.query(VoiceProfile).filter_by(name=spec["name"]).first():
            raise ValueError("Another existing voice already uses this name; choose a new name")
        durable_json(path, spec)
        if matches:
            return _profile_to_response(matches[0]).model_dump(mode="json")
        result = await create_profile(
            VoiceProfileCreate(
                name=spec["name"],
                description="Locally fine-tuned Spanish narrator; independently checked on held-out and new prose.",
                language="es",
                voice_type="preset",
                preset_engine="qwen_custom_voice",
                preset_voice_id=spec["voice_id"],
                default_engine="qwen_custom_voice",
            ),
            db,
        )
        return result.model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--slug", default="fabian")
    parser.add_argument("--name", default="Fabián — fine-tuned")
    args = parser.parse_args()
    if not (args.data_dir / "voicebox.db").is_file():
        raise ValueError("Select an existing Voicebox data directory")
    spec = checked_registration(args.model, args.evaluation, args.slug, args.name)
    from backend import config
    from backend.database import session

    config.set_data_dir(args.data_dir)
    session.init_db()
    with session.SessionLocal() as db:
        result = asyncio.run(register_profile(spec, db))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
