"""Prepare mixed-speaker episodes from timestamp hypotheses, never assumed labels."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from .corpus import atomic_json, candidate_segments, headroom, matched_runs, sha256
from .speaker import RATE, SpeakerEncoder, accepted, speaker_scores, windows
from .verify_corpus import keys


def decode(path: Path, *, start: float = 0, duration: float | None = None) -> np.ndarray:
    """Decode source audio without integer clipping, resampling only to 24 kHz."""
    command = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path)]
    if duration is not None:
        command.extend(["-t", str(duration)])
    command.extend(["-ar", str(RATE), "-ac", "1", "-f", "f32le", "-"])
    return np.frombuffer(subprocess.check_output(command), dtype="<f4").copy()


def inventory(config: dict) -> list[dict]:
    """Bind actual Unicode-normalized filenames, durations and episode splits."""
    source = Path(config["audio_directory"])
    timestamps = Path(config["timestamps"])
    indexed = {}
    for path in timestamps.glob("*.json"):
        name = unicodedata.normalize("NFC", path.stem)
        if name in indexed:
            raise ValueError(f"Ambiguous timestamp filename: {name}")
        indexed[name] = path
    validation, test = set(config["validation_episodes"]), set(config["test_episodes"])
    if not validation or not test or validation & test:
        raise ValueError("Validation and test episodes must be nonempty and disjoint")
    episodes = []
    for audio in sorted(source.glob("*.m4b")):
        timestamp = indexed[unicodedata.normalize("NFC", audio.stem)]
        asr = json.loads(timestamp.read_text())
        if (
            unicodedata.normalize("NFC", asr["audio_file"]) != unicodedata.normalize("NFC", audio.name)
            or asr.get("language") != "es"
            or asr.get("word_timestamps") is not True
            or not asr.get("sanity", {}).get("ok")
            or not isinstance(asr["ep"], int)
        ):
            raise ValueError(f"Invalid timestamp provenance: {timestamp}")
        duration = float(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(audio)]
            )
        )
        if not np.isfinite(duration) or duration < 60 or abs(duration - asr["dur"]) > 0.15:
            raise ValueError(f"Source/timestamp duration mismatch: {audio}")
        chapter = asr["ep"]
        episodes.append(
            {
                "chapter": chapter,
                "audio": str(audio.resolve()),
                "sha256": sha256(audio),
                "timestamps": str(timestamp.resolve()),
                "timestamp_sha256": sha256(timestamp),
                "timestamp_model": asr["model"],
                "duration": duration,
                "split": "validation" if chapter in validation else "test" if chapter in test else "train",
            }
        )
    ids = [episode["chapter"] for episode in episodes]
    if set(ids) != set(range(1, config["expected_episodes"] + 1)) or len(ids) != len(set(ids)):
        raise ValueError("Incomplete or duplicated source episodes")
    if not validation | test <= set(ids) or validation | test == set(ids):
        raise ValueError("Invalid held-out episode assignment")
    return sorted(episodes, key=lambda episode: episode["chapter"])


def timestamp_hypotheses(asr: dict, duration: float) -> tuple[str, list[dict]]:
    """Break alignment at uncertain words, timestamps and wrapper announcements."""
    words = asr["words"]
    text = " ".join(word["w"].strip() for word in words)
    segments = {segment["id"]: segment for segment in asr["segments"]}
    hypotheses = []
    for word in words:
        segment = segments[word["seg"]]
        start, end = float(word["start"]), float(word["end"])
        confident = (
            20 <= start < end <= duration - 25
            and word["prob"] >= 0.90
            and segment["no_speech_prob"] < 0.3
            and segment["avg_logprob"] > -0.5
            and segment["compression_ratio"] < 2.4
        )
        hypotheses.append({"w": word["w"] if confident else "", "s": start, "e": end})
    return text, hypotheses


def speaker_bank(config: dict, encoder: SpeakerEncoder, episodes: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use only declared training references and require rejected announcer controls."""
    references = config["speaker_references"]
    train_episodes = {episode["chapter"] for episode in episodes if episode["split"] == "train"}
    if any(ref.get("episode") is not None and ref["episode"] not in train_episodes for ref in references):
        raise ValueError("Speaker references cannot use held-out episodes")
    targets = np.stack([encoder.embed(headroom(decode(Path(ref["path"]), duration=12))[0]) for ref in references])
    first = next(episode for episode in episodes if episode["chapter"] == config["negative_episode"])
    if first["split"] != "train":
        raise ValueError("Speaker negative reference must belong to training")
    negatives = np.stack(
        [
            encoder.embed(headroom(decode(Path(first["audio"]), start=start, duration=duration))[0])
            for start, duration in ((1.1, 2.1), (4.8, 7.9))
        ]
    )
    controls = []
    for episode in episodes:
        if episode["chapter"] not in config["control_episodes"]:
            continue
        if episode["split"] != "train":
            raise ValueError("Speaker calibration controls must belong to training")
        for start, duration in (
            (1.1, 2.1),
            (4.8, 7.9),
            (episode["duration"] - 19, 6),
            (episode["duration"] - 11.5, 8.5),
        ):
            clip = headroom(decode(Path(episode["audio"]), start=start, duration=duration))[0]
            scores = [speaker_scores(encoder.embed(part), targets, negatives) for part in windows(clip)]
            passed = accepted(scores, **config["speaker_thresholds"])
            controls.append(
                {
                    "episode": episode["chapter"],
                    "start": start,
                    "duration": duration,
                    "scores": scores,
                    "accepted": passed,
                }
            )
            if passed:
                raise ValueError("Speaker filtering incorrectly accepted an announcer control")
    if len(controls) != len(set(config["control_episodes"])) * 4 or len(controls) < 8:
        raise ValueError("Speaker filtering needs complete independent negative controls")
    return targets, negatives, {"controls": controls, "thresholds": config["speaker_thresholds"]}


def prepare(config_path: Path) -> None:
    """Prepare restartable, speaker-screened WAVs for independent ASR verification."""
    config = json.loads(config_path.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _prepare_locked(config_path, config, output)


def _prepare_locked(config_path: Path, config: dict, output: Path) -> None:
    episodes = inventory(config)
    identity = {
        "config_sha256": sha256(config_path),
        "episodes": episodes,
        "speaker_model_sha256": sha256(Path(config["base"]) / "model.safetensors"),
        "references": [{**ref, "sha256": sha256(Path(ref["path"]))} for ref in config["speaker_references"]],
        "code": {name: sha256(Path(__file__).with_name(name)) for name in ("episodes.py", "speaker.py", "corpus.py")},
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "qwen-tts", "numpy")},
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Episode preparation identity changed; use a new output directory")
    atomic_json(identity_path, identity)
    for directory in ("audio", "chapters", "speaker_checks"):
        (output / directory).mkdir(exist_ok=True)
    encoder = SpeakerEncoder(Path(config["base"]))
    targets, negatives, controls = speaker_bank(config, encoder, episodes)
    atomic_json(output / "speaker-controls.json", controls)
    all_rows = []
    counts = Counter()
    started = time.monotonic()
    for episode in episodes:
        chapter = episode["chapter"]
        checkpoint = output / "chapters" / f"ch{chapter:02d}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["episode"] != episode:
                raise ValueError("Saved episode identity changed")
            for row in saved["rows"]:
                if sha256(Path(row["audio"])) != row["sha256"]:
                    raise ValueError(f"Prepared audio changed: {row['id']}")
        else:
            if shutil.disk_usage(output).free < 6 * 1024**3:
                raise RuntimeError("Less than 6 GiB free; stop before preparing more audio")
            audio, gain = headroom(decode(Path(episode["audio"])))
            if abs(len(audio) / RATE - episode["duration"]) > 0.15:
                raise ValueError("Decoded duration differs from timestamp provenance")
            asr = json.loads(Path(episode["timestamps"]).read_text())
            text, hypotheses = timestamp_hypotheses(asr, len(audio) / RATE)
            candidates = candidate_segments(text, matched_runs(text, hypotheses, len(audio) / RATE), audio)
            rows, rejected = [], Counter()
            for index, item in enumerate(candidates):
                rid = f"sisifo_ch{chapter:02d}_{index:05d}"
                a, b = round(item["start"] * RATE), round(item["end"] * RATE)
                clip = audio[a:b]
                if a / RATE < 20 or b / RATE > episode["duration"] - 25:
                    rejected["wrapper_boundary"] += 1
                    continue
                if np.sqrt(np.mean(clip.astype(np.float64) ** 2)) < 0.01:
                    rejected["too_quiet"] += 1
                    continue
                path = output / "audio" / f"{rid}.wav"
                row = {
                    "id": rid,
                    "audio": str(path.resolve()),
                    "text": item["text"],
                    "language": "Spanish",
                    "source": "sisifo",
                    "chapter": chapter,
                    "source_sha256": episode["sha256"],
                    "split": episode["split"],
                    "source_start": a / RATE,
                    "source_end": b / RATE,
                    "duration": len(clip) / RATE,
                    "source_gain": gain,
                    "text_sha256": hashlib.sha256(" ".join(keys(item["text"])).encode()).hexdigest(),
                    "label_origin": "large-v3-turbo hypothesis; independent verification required",
                }
                checked_path = output / "speaker_checks" / f"{rid}.json"
                if checked_path.exists():
                    checked = json.loads(checked_path.read_text())
                    if checked["candidate"] != row:
                        raise ValueError(f"Speaker check identity changed: {rid}")
                else:
                    scores = [
                        speaker_scores(encoder.embed(part), targets, negatives) for part in [clip, *windows(clip)]
                    ]
                    checked = {
                        "candidate": row,
                        "scores": scores,
                        "accepted": accepted(scores, **config["speaker_thresholds"]),
                    }
                    atomic_json(checked_path, checked)
                if not checked["accepted"]:
                    rejected["uncertain_speaker"] += 1
                    continue
                if not path.exists():
                    temporary = path.with_suffix(".tmp.wav")
                    sf.write(temporary, clip, RATE, subtype="FLOAT")
                    temporary.replace(path)
                else:
                    actual, rate = sf.read(path, dtype="float32")
                    if rate != RATE or not np.array_equal(actual, clip):
                        raise ValueError(f"Prepared clip differs from source: {rid}")
                rows.append({**row, "sha256": sha256(path), "speaker_check_sha256": sha256(checked_path)})
            saved = {"episode": episode, "rows": rows, "rejected": dict(rejected), "candidates": len(candidates)}
            atomic_json(checkpoint, saved)
        all_rows.extend(saved["rows"])
        counts.update(saved["rejected"])
        print(
            f"episode={chapter} accepted={len(saved['rows'])}/{saved['candidates']} rejected={saved['rejected']} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    ownership = {}
    for row in all_rows:
        ownership.setdefault(row["text_sha256"], set()).add(row["split"])
    all_rows = [row for row in all_rows if len(ownership[row["text_sha256"]]) == 1]
    summary = {
        "complete": True,
        "screening": {
            "identity_sha256": sha256(identity_path),
            "controls_sha256": sha256(output / "speaker-controls.json"),
        },
        "sources": {"sisifo": {str(e["chapter"]): {"split": e["split"], "sha256": e["sha256"]} for e in episodes}},
        "splits": {},
        "rejected": dict(counts),
    }
    for split in ("train", "validation", "test"):
        rows = [row for row in all_rows if row["split"] == split]
        if not rows:
            raise ValueError(f"No speaker-screened audio in {split}")
        path = output / f"{split}_raw.jsonl"
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        temporary.replace(path)
        summary["splits"][split] = {"clips": len(rows), "hours": sum(row["duration"] for row in rows) / 3600}
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    prepare(parser.parse_args().config)
