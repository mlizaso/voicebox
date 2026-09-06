import json

import numpy as np
import pytest

from scripts.finetune_qwen.corpus import sha256
from scripts.finetune_qwen.data import (
    code_row,
    group_key,
    load_verified_corpus,
    select_reference,
    validate_source,
    validate_speaker_check,
)


@pytest.fixture
def corpus(tmp_path):
    for directory in ("codes", "audio", "verification"):
        (tmp_path / directory).mkdir()
    model = {"path": "local-whisper", "weight_sha256": "whisper-hash"}
    counts = {}
    for split, chapter in (("train", 4), ("validation", 8), ("test", 12)):
        rid = f"ch{chapter:02d}_00000"
        audio = tmp_path / "audio" / f"{rid}.wav"
        audio.write_bytes(b"test audio contents")
        row = {
            "id": rid,
            "chapter": chapter,
            "split": split,
            "duration": 8,
            "audio": str(audio),
            "sha256": sha256(audio),
            "text": f"Frase distinta en el capítulo {chapter}.",
            "text_sha256": f"text-{chapter}",
        }
        for suffix in ("raw", "verified"):
            (tmp_path / f"{split}_{suffix}.jsonl").write_text(json.dumps(row) + "\n")
        np.savez(
            tmp_path / "codes" / f"{rid}.npz", codes=np.zeros((100, 16), dtype=np.int16), audio_sha256=row["sha256"]
        )
        checked = {
            "id": rid,
            "expected": row["text"],
            "exact_words": True,
            "model": model,
            "audio_sha256": row["sha256"],
        }
        (tmp_path / "verification" / f"{rid}.json").write_text(json.dumps(checked))
        counts[split] = {"accepted": 1, "candidate": 1}
    (tmp_path / "summary.json").write_text(json.dumps({"complete": True}))
    (tmp_path / "encoding-complete.json").write_text("{}")
    (tmp_path / "codes" / "identity.json").write_text("{}")
    (tmp_path / "verification-summary.json").write_text(
        json.dumps({"complete": True, "model": model, "splits": counts})
    )
    return tmp_path


def test_verified_corpus_identity_binds_to_codec_bytes_and_train_reference(corpus):
    rows, original = load_verified_corpus(corpus)
    assert select_reference(rows["train"])["chapter"] == 4
    row = rows["train"][0]
    np.savez(
        corpus / "codes" / f"{row['id']}.npz", codes=np.ones((100, 16), dtype=np.int16), audio_sha256=row["sha256"]
    )
    _, changed = load_verified_corpus(corpus)
    assert original["sha256"] != changed["sha256"]


def test_training_rejects_changed_audio_despite_cached_verification(corpus):
    (corpus / "audio/ch04_00000.wav").write_bytes(b"different recording")
    with pytest.raises(ValueError, match="Audio contents changed"):
        load_verified_corpus(corpus)


def test_training_rejects_changed_text_despite_cached_verification(corpus):
    for suffix in ("raw", "verified"):
        path = corpus / f"train_{suffix}.jsonl"
        row = json.loads(path.read_text())
        row["text"] = "This text was never independently verified."
        path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Independent verification mismatch"):
        load_verified_corpus(corpus)


def test_training_rejects_duplicate_text_across_splits(corpus):
    for suffix in ("raw", "verified"):
        path = corpus / f"validation_{suffix}.jsonl"
        row = json.loads(path.read_text())
        row["text_sha256"] = "text-4"
        path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        load_verified_corpus(corpus)


@pytest.mark.parametrize("codes", [np.zeros((99, 15), dtype=np.int16), np.full((100, 16), 2048), np.zeros((100, 16))])
def test_invalid_cached_codec_shape_range_and_dtype_are_rejected(corpus, codes):
    row = json.loads((corpus / "train_raw.jsonl").read_text())
    np.savez(corpus / "codes" / f"{row['id']}.npz", codes=codes, audio_sha256=row["sha256"])
    with pytest.raises(ValueError, match="Invalid cached codec"):
        code_row(corpus, row)


def test_reference_cannot_be_taken_from_held_out_chapters(corpus):
    rows, _ = load_verified_corpus(corpus)
    with pytest.raises(ValueError, match="No verified training reference"):
        select_reference(rows["validation"] + rows["test"])


def test_new_episode_number_cannot_displace_original_speaker_reference():
    original = {"id": "ch04_00077", "chapter": 4, "split": "train", "duration": 8.99}
    episode = {**original, "id": "sisifo_ch04_00000", "source": "sisifo", "duration": 9.0}
    assert select_reference([episode, original]) == original


def test_named_episode_requires_hash_and_split_declaration():
    row = {"id": "sisifo_ch04_00000", "source": "sisifo", "chapter": 4, "split": "train", "source_sha256": "a" * 64}
    sources = {"sisifo": {"4": {"split": "train", "sha256": "a" * 64}}}
    assert validate_source(row, sources) == "sisifo_ch04_"
    assert group_key(row) != group_key({"chapter": 4})
    for changed in ({}, {"sisifo": {"4": {"split": "test", "sha256": "a" * 64}}}):
        with pytest.raises(ValueError, match="source"):
            validate_source(row, changed)


@pytest.mark.parametrize("chapter", [0, 39, True, 1.5])
def test_legacy_chapter_validation_remains_strict(chapter):
    with pytest.raises(ValueError, match="chapter"):
        validate_source({"id": "bad", "chapter": chapter}, {})


def test_speaker_verification_recomputes_all_windows_and_binds_audio_labels(tmp_path):
    directory = tmp_path / "speaker_checks"
    directory.mkdir()
    row = {"id": "sisifo_ch01_00000", "duration": 8, "text": "texto", "sha256": "audio-hash"}
    score = {"target": 0.99, "negative": 0.95, "margin": 0.04}
    other = {"target": 0.94, "negative": 0.99, "margin": -0.05}
    controls = {"thresholds": {"minimum_target": 0.97, "minimum_margin": 0.015}, "controls": [{"scores": [other]}]}
    (tmp_path / "speaker-controls.json").write_text(json.dumps(controls))
    checked = {"candidate": {k: v for k, v in row.items() if k != "sha256"}, "scores": [score] * 6, "accepted": True}
    path = directory / f"{row['id']}.json"

    def save_and_verify():
        path.write_text(json.dumps(checked))
        row["speaker_check_sha256"] = sha256(path)
        validate_speaker_check(tmp_path, row)

    save_and_verify()
    checked["scores"][-1] = other
    with pytest.raises(ValueError, match="Speaker verification mismatch"):
        save_and_verify()
    checked["scores"] = [score] * 5
    with pytest.raises(ValueError, match="Speaker verification mismatch"):
        save_and_verify()
    checked["scores"] = [score] * 6
    checked["candidate"]["text"] = "different text"
    with pytest.raises(ValueError, match="Speaker verification mismatch"):
        save_and_verify()
