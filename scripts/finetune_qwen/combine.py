"""Build a verified continuation corpus with deterministic original-voice rehearsal."""

import argparse
import fcntl
import json
import shutil
from pathlib import Path

from .corpus import atomic_json, sha256
from .data import SPLITS, load_verified_corpus
from .train import validation_rows


def rehearsal_rows(rows: list[dict], count: int, seed: int, reference_id: str) -> list[dict]:
    """Balance original chapters while preserving the parent's exact training reference."""
    if count < 1 or count > len(rows) or any(row["split"] != "train" for row in rows):
        raise ValueError("Rehearsal needs a valid count of verified training rows")
    reference = next((row for row in rows if row["id"] == reference_id), None)
    if reference is None:
        raise ValueError("Parent reference is missing from rehearsal source")
    selected = validation_rows(rows, count, seed)
    if reference not in selected:
        selected[-1] = reference
    return sorted(selected, key=lambda row: row["id"])


def copy_checked(source: Path, destination: Path) -> None:
    """Copy an immutable artifact without ever overwriting different contents."""
    expected = sha256(source)
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError(f"Existing composed artifact differs: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    if sha256(temporary) != expected:
        raise ValueError("Source changed while composing the corpus")
    temporary.replace(destination)


def combine(episodes: Path, original: Path, output: Path, *, count: int, seed: int, reference_id: str) -> None:
    """Copy verified rows/codes and preserve disjoint source groups in a new corpus."""
    if output.resolve() in {episodes.resolve(), original.resolve()}:
        raise ValueError("Composition must use a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".composition.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        new_rows, new_identity = load_verified_corpus(episodes)
        old_rows, old_identity = load_verified_corpus(original)
        if (
            new_identity["encoding"] != old_identity["encoding"]
            or new_identity["verification"]["model"] != old_identity["verification"]["model"]
        ):
            raise ValueError("Corpus codec and independent verifier identities must match")
        selected = rehearsal_rows(old_rows["train"], count, seed, reference_id)
        chosen = {split: new_rows[split] + (selected if split == "train" else old_rows[split]) for split in SPLITS}
        owner = {}
        for split, rows in chosen.items():
            for row in rows:
                owner.setdefault(row["text_sha256"], set()).add(split)
        chosen = {split: [row for row in rows if len(owner[row["text_sha256"]]) == 1] for split, rows in chosen.items()}
        if reference_id not in {row["id"] for row in chosen["train"]}:
            raise ValueError("Cross-corpus text leakage excludes the parent reference")
        identity = {
            "episode_corpus": str(episodes.resolve()),
            "episode_identity": new_identity,
            "original_corpus": str(original.resolve()),
            "original_identity": old_identity,
            "rehearsal_count": count,
            "seed": seed,
            "reference_id": reference_id,
            "selected_ids": {split: [row["id"] for row in rows] for split, rows in chosen.items()},
            "code": {name: sha256(Path(__file__).with_name(name)) for name in ("combine.py", "data.py", "train.py")},
        }
        identity_path = output / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Composition identity changed; use another output directory")
        atomic_json(identity_path, identity)
        for directory in ("audio", "codes", "verification", "speaker_checks"):
            (output / directory).mkdir(exist_ok=True)
        copy_checked(episodes / "speaker-controls.json", output / "speaker-controls.json")
        copy_checked(episodes / "codes/identity.json", output / "codes/identity.json")
        sources = json.loads((episodes / "summary.json").read_text())["sources"]
        summary = {
            "complete": True,
            "sources": sources,
            "screening": {
                "identity_sha256": sha256(identity_path),
                "controls_sha256": sha256(output / "speaker-controls.json"),
            },
            "splits": {},
        }
        counts = {}
        for split, rows in chosen.items():
            result = []
            for row in rows:
                if shutil.disk_usage(output).free < 12 * 1024**3:
                    raise RuntimeError("Less than 12 GiB free; preserve room for training and export")
                source = episodes if row.get("source") else original
                rid = row["id"]
                audio = output / "audio" / f"{rid}.wav"
                copy_checked(Path(row["audio"]), audio)
                copy_checked(source / "codes" / f"{rid}.npz", output / "codes" / f"{rid}.npz")
                copy_checked(source / "verification" / f"{rid}.json", output / "verification" / f"{rid}.json")
                copied = {**row, "audio": str(audio.resolve())}
                if row.get("source"):
                    checked = json.loads((source / "speaker_checks" / f"{rid}.json").read_text())
                    checked["candidate"]["audio"] = copied["audio"]
                    path = output / "speaker_checks" / f"{rid}.json"
                    if path.exists() and json.loads(path.read_text()) != checked:
                        raise ValueError(f"Composed speaker check changed: {rid}")
                    atomic_json(path, checked)
                    copied["speaker_check_sha256"] = sha256(path)
                result.append(copied)
            for suffix in ("raw", "verified"):
                path = output / f"{split}_{suffix}.jsonl"
                temporary = path.with_suffix(".tmp")
                temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result))
                temporary.replace(path)
            hours = sum(row["duration"] for row in result) / 3600
            counts[split] = {"candidate": len(result), "accepted": len(result), "hours": hours}
            summary["splits"][split] = {"clips": len(result), "hours": hours}
        atomic_json(output / "summary.json", summary)
        atomic_json(
            output / "verification-summary.json",
            {"complete": True, "model": new_identity["verification"]["model"], "splits": counts},
        )
        atomic_json(output / "encoding-complete.json", {**new_identity["encoding"], "composed": True})
        verified, final_identity = load_verified_corpus(output)
        print(
            json.dumps(
                {"splits": {split: len(rows) for split, rows in verified.items()}, "sha256": final_identity["sha256"]},
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--reference-id", required=True)
    args = parser.parse_args()
    combine(args.episodes, args.original, args.output, count=args.count, seed=args.seed, reference_id=args.reference_id)
