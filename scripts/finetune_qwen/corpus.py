"""Align existing word timestamps to an EPUB without inventing training labels.

Audio timestamps are only candidates. Exact consecutive word matches and quiet
waveform boundaries are required before exporting an example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

WORD = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)
SAMPLE_RATE = 24000


def normalized(word: str) -> str:
    folded = unicodedata.normalize("NFKD", word.casefold())
    return "".join(c for c in folded if c.isalnum() and not unicodedata.combining(c))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def epub_text(archive: zipfile.ZipFile, members: list[str]) -> str:
    paragraphs = []
    for member in members:
        if archive.getinfo(member).file_size > 5_000_000:
            raise ValueError(f"Unexpectedly large EPUB document: {member}")
        root = ET.fromstring(archive.read(member))
        body = next(node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "body")
        for node in body.iter():
            if node.tag.rsplit("}", 1)[-1] in {"p", "h1", "h2", "h3", "h4"}:
                text = re.sub(r"\s+", " ", "".join(node.itertext())).strip()
                if text:
                    paragraphs.append(text)
    return "\n\n".join(paragraphs)


def matched_runs(text: str, words: list[dict], chapter_duration: float) -> list[list[dict]]:
    """Return only consecutive exact word matches, retaining original text spans."""
    tokens = list(WORD.finditer(text))
    ref_keys = [normalized(token.group()) for token in tokens]
    observed = []
    for word in words:
        start, end = float(word["s"]), float(word["e"])
        pieces = list(WORD.finditer(word["w"]))
        # A timestamp for several words cannot safely locate their boundaries.
        if len(pieces) != 1 or not (0 <= start < end <= chapter_duration):
            observed.append({"key": "", "s": start, "e": end})
        else:
            observed.append({"key": normalized(pieces[0].group()), "s": start, "e": end})
    matcher = SequenceMatcher(a=ref_keys, b=[w["key"] for w in observed], autojunk=False)
    runs = []
    for block in matcher.get_matching_blocks():
        if block.size < 8:
            continue
        run = []
        for offset in range(block.size):
            index, heard_index = block.a + offset, block.b + offset
            heard = observed[heard_index]
            if run and (heard["s"] < run[-1]["e"] - 0.04 or heard["s"] - run[-1]["e"] > 3):
                if len(run) >= 8:
                    runs.append(run)
                run = []
            token = tokens[index]
            next_start = tokens[index + 1].start() if index + 1 < len(tokens) else len(text)
            run.append(
                {
                    "s": heard["s"],
                    "e": heard["e"],
                    "char_start": token.start(),
                    "char_end": token.end(),
                    "tail": text[token.end() : next_start],
                    "previous_end": observed[heard_index - 1]["e"] if heard_index else 0,
                    "next_start": observed[heard_index + 1]["s"]
                    if heard_index + 1 < len(observed)
                    else chapter_duration,
                }
            )
        if len(run) >= 8:
            runs.append(run)
    return runs


def quiet_cut(audio: np.ndarray, before: float, after: float) -> float | None:
    """Find the middle of >=60 ms of quiet near an ASR inter-word boundary."""
    if after < before - 0.04:
        return None
    lo = max(0, round((before - 0.10) * SAMPLE_RATE))
    hi = min(len(audio), round((after + 0.10) * SAMPLE_RATE))
    if hi - lo < 1440:
        return None
    frame = 240
    segment = audio[lo : lo + ((hi - lo) // frame) * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(segment.astype(np.float64) ** 2, axis=1))
    quiet = rms < 10 ** (-42 / 20)
    changes = np.diff(np.r_[False, quiet, False].astype(np.int8))
    intervals = [
        (a, b) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True) if b - a >= 6
    ]
    if not intervals:
        return None
    middle = (before + after) / 2
    a, b = min(intervals, key=lambda ab: abs((lo + (ab[0] + ab[1]) * frame / 2) / SAMPLE_RATE - middle))
    return (lo + (a + b) * frame / 2) / SAMPLE_RATE


def headroom(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Keep AAC floating-point overshoots without introducing PCM clipping.

    This is one constant gain for the entire chapter, not compression, denoising
    or time/pitch processing. The gain is recorded for reproducibility.
    """
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Empty or nonfinite source audio")
    peak = float(np.max(np.abs(audio)))
    gain = min(1.0, 0.90 / peak) if peak else 1.0
    return audio * gain, gain


def candidate_segments(text: str, runs: list[list[dict]], audio: np.ndarray) -> list[dict]:
    candidates = []
    for run in runs:
        first = 0
        while first < len(run) - 7:
            start = quiet_cut(audio, run[first]["previous_end"], run[first]["s"])
            if start is None:
                first += 1
                continue
            options = []
            for last in range(first + 7, len(run)):
                span = run[last]["e"] - start
                if span > 18:
                    break
                if span < 3:
                    continue
                end = quiet_cut(audio, run[last]["e"], run[last]["next_start"])
                if end is None or not 3 <= end - start <= 18:
                    continue
                tail = run[last]["tail"]
                sentence_end = bool(re.search(r"[.!?…]|\n", tail))
                score = (sentence_end, -abs((end - start) - 9))
                options.append((score, last, end))
            if not options:
                first += 1
                continue
            _, last, end = max(options)
            a, b = run[first]["char_start"], run[last]["char_end"]
            # Preserve punctuation adjacent to the last word, without consuming
            # the opening punctuation of the following sentence.
            trailing = re.match(r"[.,;:!?…»”\u2019\"\)\]]*", run[last]["tail"]).group()
            transcript = re.sub(r"\s+", " ", text[a:b] + trailing).strip()
            prefix = text[max(0, a - 1) : a]
            if prefix in {"¿", "¡", "«", "“"}:
                transcript = prefix + transcript
            candidates.append({"start": start, "end": end, "text": transcript, "words": last - first + 1})
            first = last + 1
    return candidates


def prepare(config_path: Path, *, chapters: list[int] | None = None) -> None:
    config = json.loads(config_path.read_text())
    source, epub = Path(config["audio"]), Path(config["epub"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "chapters").mkdir(exist_ok=True)
    (output / "audio").mkdir(exist_ok=True)
    (output / "text").mkdir(exist_ok=True)
    probe = json.loads(
        subprocess.check_output(["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(source)])
    )
    chapter_info = {int(c["id"]): c for c in probe["chapters"]}
    identity = {
        "config_sha256": sha256(config_path),
        "audio_sha256": sha256(source),
        "epub_sha256": sha256(epub),
        "script_sha256": sha256(Path(__file__)),
    }
    identity_file = output / "identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError("Source, configuration or preparation code changed; use a new output directory")
    atomic_json(identity_file, identity)
    with zipfile.ZipFile(epub) as archive:
        for number, members in sorted(config["chapter_documents"].items(), key=lambda item: int(item[0])):
            cid = int(number)
            if chapters is not None and cid not in chapters:
                continue
            checkpoint = output / "chapters" / f"ch{cid:02d}.json"
            if checkpoint.exists():
                print(f"ch{cid:02d}: cached", flush=True)
                continue
            if shutil.disk_usage(output).free < 6 * 2**30:
                raise OSError("Less than 6 GiB free; refusing further dataset extraction")
            info = chapter_info[cid]
            offset = float(info["start_time"])
            duration = float(info["end_time"]) - offset
            words = json.loads((Path(config["timestamps"]) / f"ch{cid:02d}.json").read_text())
            text = epub_text(archive, members)
            (output / "text" / f"ch{cid:02d}.txt").write_text(text, encoding="utf-8")
            raw = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-ss",
                    str(offset),
                    "-i",
                    str(source),
                    "-t",
                    str(duration),
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    str(SAMPLE_RATE),
                    "-f",
                    "f32le",
                    "-",
                ]
            )
            audio, gain = headroom(np.frombuffer(raw, dtype="<f4"))
            runs = matched_runs(text, words, duration)
            candidates = candidate_segments(text, runs, audio)
            split = (
                "validation"
                if cid in config["validation_chapters"]
                else "test"
                if cid in config["test_chapters"]
                else "train"
            )
            rows = []
            rejected = Counter()
            for index, item in enumerate(candidates):
                a, b = round(item["start"] * SAMPLE_RATE), round(item["end"] * SAMPLE_RATE)
                clip = audio[a:b]
                if not np.isfinite(clip).all() or np.max(np.abs(clip)) >= 1:
                    rejected["nonfinite_or_clipped"] += 1
                    continue
                if np.sqrt(np.mean(clip.astype(np.float64) ** 2)) < 0.01:
                    rejected["too_quiet"] += 1
                    continue
                name = f"ch{cid:02d}_{index:05d}"
                path = output / "audio" / f"{name}.wav"
                sf.write(path, clip, SAMPLE_RATE, subtype="FLOAT")
                rows.append(
                    {
                        "id": name,
                        "audio": str(path),
                        "text": item["text"],
                        "ref_audio": config["reference_audio"],
                        "language": "Spanish",
                        "chapter": cid,
                        "split": split,
                        "source_start": offset + a / SAMPLE_RATE,
                        "source_end": offset + b / SAMPLE_RATE,
                        "duration": len(clip) / SAMPLE_RATE,
                        "source_gain": gain,
                        "sha256": sha256(path),
                        "text_sha256": hashlib.sha256(
                            " ".join(normalized(w.group()) for w in WORD.finditer(item["text"])).encode()
                        ).hexdigest(),
                    }
                )
            atomic_json(
                checkpoint,
                {
                    "chapter": cid,
                    "rows": rows,
                    "rejected": dict(rejected),
                    "timestamp_sha256": sha256(Path(config["timestamps"]) / f"ch{cid:02d}.json"),
                },
            )
            print(
                f"ch{cid:02d}: {len(rows)} {split} clips, {sum(r['duration'] for r in rows) / 60:.1f} minutes",
                flush=True,
            )
    manifests = [json.loads(path.read_text()) for path in sorted((output / "chapters").glob("ch*.json"))]
    all_rows = [row for manifest in manifests for row in manifest["rows"]]
    text_splits: dict[str, set[str]] = {}
    for row in all_rows:
        text_splits.setdefault(row["text_sha256"], set()).add(row["split"])
    all_rows = [row for row in all_rows if len(text_splits[row["text_sha256"]]) == 1]
    summary = {
        "complete": len(manifests) == len(config["chapter_documents"]),
        "prepared_chapters": len(manifests),
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        rows = [row for row in all_rows if row["split"] == split]
        path = output / f"{split}_raw.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        temporary.replace(path)
        summary["splits"][split] = {
            "clips": len(rows),
            "hours": sum(row["duration"] for row in rows) / 3600,
            "chapters": sorted({row["chapter"] for row in rows}),
        }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--chapters", type=int, nargs="+")
    args = parser.parse_args()
    prepare(args.config, chapters=args.chapters)
