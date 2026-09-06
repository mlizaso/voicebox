"""Independent recognizers rescore whole comparisons without editing the original."""

import json
from pathlib import Path

import pytest

from . import crosscheck as checker
from .corpus import sha256


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    source, model = tmp_path / "original", tmp_path / "recognizer"
    source.mkdir()
    model.mkdir()
    (model / "weights.npz").write_bytes(b"weights")
    (model / "config.json").write_text("{}")
    audio = source / "audio.wav"
    audio.write_bytes(b"audio fixture")
    identity = {
        "evaluation_code_sha256": sha256(Path(checker.__file__).with_name("evaluate.py")),
        "inference_sources": {"runtime": "unchanged"},
    }
    rows = [
        {
            "variant": variant,
            "id": "clip",
            "text": "Una historia hasta el final.",
            "audio": str(audio),
            "sha256": sha256(audio),
            "seconds": 1,
            "duration": 5,
            "speaker_cosine": 0.98,
            "hit_token_limit": False,
        }
        for variant in ("candidate", "baseline")
    ]
    (source / "generation-summary.json").write_text(json.dumps({"identity": identity, "results": rows}))
    (source / "evaluation.json").write_text(json.dumps({"identity": identity, "accepted": False}))
    monkeypatch.setattr(checker, "inference_source_identity", lambda: identity["inference_sources"])
    return source, model, tmp_path / "other-recognizer"


def test_crosscheck_scores_every_sample_and_preserves_original(comparison, monkeypatch):
    source, model, output = comparison
    original = (source / "evaluation.json").read_bytes()
    calls = []

    def recognize(audio, model_path):
        calls.append((audio, model_path))
        return {
            "text": "Una historia hasta el final.",
            "reliable": True,
            "attempts": [
                {"result": {"text": "Una historia hasta el final.", "segments": [{"compression_ratio": 1.1}]}}
            ],
        }

    monkeypatch.setattr(checker, "transcribe_audio", recognize)
    checker.crosscheck(source, model, output)
    assert len(calls) == 2
    assert (source / "evaluation.json").read_bytes() == original
    report = json.loads((output / "evaluation.json").read_text())
    assert report["accepted"]
    assert len(report["results"]) == 2
    checker.validate_crosscheck(report)
    checker.crosscheck(source, model, output)
    assert len(calls) == 2  # Resume uses only identity-bound cached ASR.
    (model / "weights.npz").write_bytes(b"other recognizer")
    with pytest.raises(ValueError, match="provenance changed"):
        checker.validate_crosscheck(report)
    with pytest.raises(ValueError, match="identity changed"):
        checker.crosscheck(source, model, output)


def test_crosscheck_never_replaces_original_or_accepts_changed_runtime(comparison, monkeypatch):
    source, model, output = comparison
    with pytest.raises(ValueError, match="preserve the original"):
        checker.crosscheck(source, model, source)
    monkeypatch.setattr(checker, "inference_source_identity", lambda: {"runtime": "changed"})
    with pytest.raises(ValueError, match="implementation or identity changed"):
        checker.crosscheck(source, model, output)


def test_recognizer_weight_identity_matches_upstream_loader(tmp_path):
    npz, safe = tmp_path / "weights.npz", tmp_path / "weights.safetensors"
    npz.write_bytes(b"npz")
    assert checker.recognizer_weights(tmp_path) == npz
    safe.write_bytes(b"safe")
    assert checker.recognizer_weights(tmp_path) == safe


def test_crosscheck_still_rejects_incorrect_endings(comparison, monkeypatch):
    source, model, output = comparison
    monkeypatch.setattr(
        checker,
        "transcribe_audio",
        lambda *_args: {
            "text": "Una historia",
            "reliable": True,
            "attempts": [{"result": {"text": "Una historia", "segments": [{"compression_ratio": 1.1}]}}],
        },
    )
    checker.crosscheck(source, model, output)
    report = json.loads((output / "evaluation.json").read_text())
    assert not report["accepted"]
    assert not report["gates"]["complete_endings"]
