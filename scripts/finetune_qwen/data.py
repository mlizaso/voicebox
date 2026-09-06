"""Validate private corpus identities and load verified codec examples."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .corpus import sha256

SPLITS = ("train", "validation", "test")


def group_key(row: dict) -> tuple[str, int]:
    """Keep identically numbered chapters of different recordings independent."""
    return row.get("source", ""), row["chapter"]


def validate_source(row: dict, sources: dict) -> str:
    """Validate declared episode ownership, preserving the original book contract."""
    source, chapter = group_key(row)
    if not isinstance(chapter, int) or isinstance(chapter, bool) or chapter < 1:
        raise ValueError(f"Invalid narration chapter: {row['id']}")
    if not source:
        if chapter > 38:
            raise ValueError(f"Invalid narration chapter: {row['id']}")
        return f"ch{chapter:02d}_"
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", source):
        raise ValueError(f"Invalid narration source: {row['id']}")
    declared = sources.get(source, {}).get(str(chapter), {})
    if (
        declared.get("split") != row["split"]
        or not re.fullmatch(r"[0-9a-f]{64}", row.get("source_sha256", ""))
        or declared.get("sha256") != row["source_sha256"]
    ):
        raise ValueError(f"Undeclared or changed narration source: {row['id']}")
    return f"{source}_ch{chapter:02d}_"


def read_jsonl(path: Path) -> list[dict]:
    """Read a complete manifest, rejecting malformed or duplicated examples."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate example IDs in {path}")
    return rows


def code_row(corpus: Path, row: dict) -> dict:
    """Read a cached 12 Hz code sequence bound to the source audio hash."""
    with np.load(corpus / "codes" / f"{row['id']}.npz", allow_pickle=False) as saved:
        if str(saved["audio_sha256"]) != row["sha256"]:
            raise ValueError(f"Codec/audio identity mismatch: {row['id']}")
        codes = saved["codes"]
        if (
            codes.ndim != 2
            or codes.shape[1] != 16
            or not np.issubdtype(codes.dtype, np.integer)
            or not np.all((codes >= 0) & (codes < 2048))
            or abs(len(codes) - row["duration"] * 12.5) > 2
        ):
            raise ValueError(f"Invalid cached codec sequence: {row['id']}")
        return {**row, "audio_codes": codes.astype(np.int64)}


def load_verified_corpus(corpus: Path) -> tuple[dict[str, list[dict]], dict]:
    """Audit every accepted clip and prevent chapter/text leakage across splits."""
    summary = json.loads((corpus / "summary.json").read_text())
    verification = json.loads((corpus / "verification-summary.json").read_text())
    encoding = json.loads((corpus / "codes" / "identity.json").read_text())
    if summary.get("complete") is not True or verification.get("complete") is not True:
        raise ValueError("Corpus preparation and independent verification must finish first")
    if not (corpus / "encoding-complete.json").is_file():
        raise ValueError("Codec encoding has not completed")
    digester = hashlib.sha256()
    digester.update(json.dumps(encoding, sort_keys=True).encode())
    digester.update(json.dumps(verification, sort_keys=True).encode())
    if "sources" in summary:
        digester.update(json.dumps(summary["sources"], sort_keys=True).encode())
        screening = summary.get("screening", {})
        if screening.get("identity_sha256") != sha256(corpus / "identity.json") or screening.get(
            "controls_sha256"
        ) != sha256(corpus / "speaker-controls.json"):
            raise ValueError("Speaker screening provenance changed")
        digester.update(json.dumps(screening, sort_keys=True).encode())
    result = {}
    chapter_owner = {}
    text_owner = {}
    all_ids = set()
    for split in SPLITS:
        candidates = read_jsonl(corpus / f"{split}_raw.jsonl")
        raw = {row["id"]: row for row in candidates}
        rows = read_jsonl(corpus / f"{split}_verified.jsonl")
        counts = verification["splits"][split]
        if not rows or counts["accepted"] != len(rows) or counts["candidate"] != len(candidates):
            raise ValueError(f"Incomplete verified split: {split}")
        for row in rows:
            rid = row["id"]
            if row != raw.get(rid) or row["split"] != split or rid in all_ids:
                raise ValueError(f"Manifest identity mismatch: {rid}")
            # Only generated IDs are used as filenames; never accept a path here.
            prefix = validate_source(row, summary.get("sources", {}))
            if not re.fullmatch(re.escape(prefix) + r"\d{5}", rid):
                raise ValueError(f"Invalid example ID: {rid}")
            all_ids.add(rid)
            if not 3 <= row["duration"] <= 18:
                raise ValueError(f"Invalid narration chapter or duration: {rid}")
            for key, owners in ((group_key(row), chapter_owner), (row["text_sha256"], text_owner)):
                if key in owners and owners[key] != split:
                    raise ValueError(f"Training/held-out leakage: {rid}")
                owners[key] = split
            audio = corpus / "audio" / f"{rid}.wav"
            if Path(row["audio"]).resolve() != audio.resolve() or sha256(audio) != row["sha256"]:
                raise ValueError(f"Audio contents changed: {rid}")
            if row.get("source"):
                validate_speaker_check(corpus, row)
            checked = json.loads((corpus / "verification" / f"{rid}.json").read_text())
            if (
                checked["id"] != rid
                or checked["exact_words"] is not True
                or checked["audio_sha256"] != row["sha256"]
                or checked["expected"] != row["text"]
                or checked["model"] != verification["model"]
            ):
                raise ValueError(f"Independent verification mismatch: {rid}")
            code_row(corpus, row)
            digester.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
            digester.update(sha256(corpus / "codes" / f"{rid}.npz").encode())
        result[split] = rows
    return result, {"sha256": digester.hexdigest(), "encoding": encoding, "verification": verification}


def validate_speaker_check(corpus: Path, row: dict) -> None:
    """Recheck window decisions instead of trusting a manifest's accepted flag."""
    from .speaker import accepted

    path = corpus / "speaker_checks" / f"{row['id']}.json"
    if row.get("speaker_check_sha256") != sha256(path):
        raise ValueError(f"Speaker check contents changed: {row['id']}")
    checked = json.loads(path.read_text())
    controls = json.loads((corpus / "speaker-controls.json").read_text())
    expected = {key: value for key, value in row.items() if key not in {"sha256", "speaker_check_sha256"}}
    window_count = (max(0, round(row["duration"] * 24000) - 72000) + 35999) // 36000 + 2
    if (
        checked["candidate"] != expected
        or checked["accepted"] is not True
        or len(checked["scores"]) != window_count
        or not accepted(checked["scores"], **controls["thresholds"])
        or not controls["controls"]
        or any(accepted(control["scores"], **controls["thresholds"]) for control in controls["controls"])
    ):
        raise ValueError(f"Speaker verification mismatch: {row['id']}")


def base_identity(base: Path) -> dict[str, str]:
    """Hash the frozen base, tokenizer and speech codec used by the experiment."""
    files = (
        "model.safetensors",
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/config.json",
    )
    return {name: sha256(base / name) for name in files}


def select_reference(rows: list[dict]) -> dict:
    """Choose one stable, medium-length reference exclusively from training."""
    eligible = [row for row in rows if row["split"] == "train" and 7 <= row["duration"] <= 12]
    if not eligible:
        raise ValueError("No verified training reference between 7 and 12 seconds")
    # The first ordinary prose chapter avoids introductory announcements.
    return min(
        eligible,
        key=lambda row: (bool(row.get("source")), row["chapter"] != 4, abs(row["duration"] - 9), row["id"]),
    )
