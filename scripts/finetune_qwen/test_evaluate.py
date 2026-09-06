"""Fair baseline conditioning and predeclared quality-gate regression tests."""

import asyncio
import json
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from .evaluate import (
    NOVEL,
    baseline_reference,
    evaluation_cases,
    has_complete_ending,
    local_baseline_voice,
    summarize,
    transcribe_audio,
    word_error,
)


def test_new_prose_includes_long_form_generalization_without_split_reuse():
    assert set(NOVEL["validation"]).isdisjoint(NOVEL["test"])
    for passages in NOVEL.values():
        assert len(passages) == 4
        assert max(len(passage.split()) for passage in passages) >= 150


@pytest.mark.parametrize(
    ("expected", "actual", "result"),
    [
        ("¡Hola, MUNDO!", "hola mundo", (0, 2)),
        ("El tren llegó ayer.", "El tren llegó", (1, 4)),
        ("el tren", "el último tren pasó", (2, 2)),
        ("a b c", "a d c", (1, 3)),
        ("", "una palabra", (2, 0)),
    ],
)
def test_normalized_word_error(expected, actual, result):
    assert word_error(expected, actual) == result


def test_baseline_reference_matches_real_voicebox_processing(tmp_path):
    from backend.backends.base import combine_voice_prompts
    from backend.utils.audio import save_audio

    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    sf.write(first, np.sin(np.arange(2400) * 0.07) * 0.04, 24000)
    sf.write(second, np.sin(np.arange(4800) * 0.13) * 0.3, 24000)
    with sqlite3.connect(tmp_path / "voicebox.db") as db:
        db.execute(
            "CREATE TABLE profile_samples (id TEXT,profile_id TEXT,ordinal INTEGER,audio_path TEXT,reference_text TEXT)"
        )
        db.executemany(
            "INSERT INTO profile_samples VALUES (?,?,?,?,?)",
            [("a", "p", 1, second.name, "Segunda frase."), ("z", "p", 0, first.name, "Primera frase.")],
        )
    actual, text = baseline_reference(tmp_path, "p", tmp_path)
    audio, expected_text = asyncio.run(
        combine_voice_prompts([str(first), str(second)], ["Primera frase.", "Segunda frase."])
    )
    expected = tmp_path / "expected.wav"
    save_audio(audio, str(expected), 24000)
    assert text == expected_text
    assert actual.read_bytes() == expected.read_bytes()


def test_local_baseline_resolves_registered_profile_with_weight_verification(tmp_path, monkeypatch):
    from sqlalchemy import create_engine

    from backend.database.models import VoiceProfile

    engine = create_engine(f"sqlite:///{tmp_path / 'voicebox.db'}")
    VoiceProfile.__table__.create(bind=engine)
    with engine.begin() as connection:
        connection.execute(
            VoiceProfile.__table__.insert(),
            [
                {
                    "id": "local",
                    "name": "local",
                    "voice_type": "preset",
                    "preset_engine": "qwen_custom_voice",
                    "preset_voice_id": "finetuned:fabian",
                },
                {"id": "old", "name": "old", "voice_type": "cloned", "preset_engine": None, "preset_voice_id": None},
                {
                    "id": "unsupported",
                    "name": "unsupported",
                    "voice_type": "preset",
                    "preset_engine": "kokoro",
                    "preset_voice_id": "other",
                },
            ],
        )
    engine.dispose()
    calls = []

    def read_voice(voice_id, *, verify_weights):
        calls.append((voice_id, verify_weights))
        return {"voice_id": voice_id, "model_path": "registered-v1"}

    monkeypatch.setattr("backend.services.finetuned_voices.read_voice", read_voice)
    assert local_baseline_voice(tmp_path, "local")["model_path"] == "registered-v1"
    assert calls == [("finetuned:fabian", True)]
    assert local_baseline_voice(tmp_path, "old") is None
    with pytest.raises(ValueError, match="does not exist"):
        local_baseline_voice(tmp_path, "missing")
    with pytest.raises(ValueError, match="cloned or locally fine-tuned"):
        local_baseline_voice(tmp_path, "unsupported")


def test_evaluation_separates_sources_and_accepts_predeclared_new_prose(tmp_path):
    rows = [
        {"id": f"{source}-{i}", "source": source, "chapter": 8, "duration": 3 + i, "text": f"Texto {source} {i}."}
        for source in ("", "sisifo")
        for i in range(10)
    ]
    (tmp_path / "test_verified.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    prose = [
        "Esta frase abre una evaluación nueva.",
        "Mañana tendremos noticias de la expedición.",
        "El café se enfrió junto a la ventana.",
        "Una historia original recorre el paisaje. " * 30,
    ]
    cases = evaluation_cases(tmp_path, "test", prose)
    assert len(cases) == 16
    assert sum(row.get("source") == "sisifo" for row in cases) == 6
    assert [row["text"] for row in cases[-4:]] == prose
    assert cases == evaluation_cases(tmp_path, "test", prose)
    with pytest.raises(ValueError, match="four distinct"):
        evaluation_cases(tmp_path, "test", prose[:3])
    with pytest.raises(ValueError, match="four distinct"):
        evaluation_cases(tmp_path, "test", ["too short"] * 4)


def score_rows():
    return [
        {
            "variant": variant,
            "word_errors": 0,
            "words": 100,
            "wer": 0.0,
            "speaker_cosine": 0.8,
            "seconds": 1.0,
            "duration": 5.0,
            "hit_token_limit": False,
            "text": "La historia llega hasta el final.",
            "transcribed": "La historia llega hasta el final.",
        }
        for variant in ("candidate", "baseline")
    ]


def test_matching_quality_and_speed_passes():
    assert summarize(score_rows())["accepted"] is True


@pytest.mark.parametrize(
    ("change", "gate"),
    [
        ({"word_errors": 21, "wer": 0.21}, "intelligible"),
        ({"word_errors": 3, "wer": 0.03}, "transcript_not_regressed"),
        ({"speaker_cosine": 0.76}, "speaker_not_regressed"),
        ({"seconds": 1.11}, "warm_speed_not_regressed"),
        ({"hit_token_limit": True}, "finished_audio"),
        ({"transcribed": "La historia llega hasta"}, "complete_endings"),
    ],
)
def test_each_predeclared_regression_rejects_promotion(change, gate):
    rows = score_rows()
    rows[0].update(change)
    report = summarize(rows)
    assert report["accepted"] is False
    assert report["gates"][gate] is False


def test_ending_check_ignores_case_but_rejects_premature_eos():
    text = "Esta es una historia muy larga, y sin embargo somos incapaces de hacerle frente."
    assert not has_complete_ending(text, "Esta es una historia muy larga, y sin embargo.")
    assert has_complete_ending(text, text.upper())
    assert not has_complete_ending("", "")


@pytest.mark.parametrize("ratio", [5.912, float("nan"), float("inf")])
def test_asr_retries_invalid_compression_once_without_expected_text(monkeypatch, ratio):
    attempts = []
    raw = [
        {"text": " a la cultura," * 40, "segments": [{"compression_ratio": ratio}]},
        {"text": " a la literatura, a la emoción,", "segments": [{"compression_ratio": 1.1}]},
    ]

    def recognize(audio, **options):
        attempts.append((audio, options))
        return raw[len(attempts) - 1]

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=recognize))
    result = transcribe_audio("same.wav", "local-medium")
    assert result["reliable"] is True
    assert result["text"] == "a la literatura, a la emoción,"
    assert [attempt["result"] for attempt in result["attempts"]] == raw
    assert [options["without_timestamps"] for _, options in attempts] == [False, True]
    assert all(audio == "same.wav" for audio, _ in attempts)
    assert all(options["temperature"] == 0.0 for _, options in attempts)
    assert all("initial_prompt" not in options for _, options in attempts)


def test_asr_does_not_retry_a_genuine_wrong_ending(monkeypatch):
    calls = []

    def recognize(*args, **kwargs):
        calls.append((args, kwargs))
        return {"text": "No está claro el nazismo y el nazismo.", "segments": [{"compression_ratio": 1.2}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=recognize))
    result = transcribe_audio("wrong-ending.wav", "local-medium")
    assert result["reliable"] is True
    assert len(calls) == 1
    rows = score_rows()
    rows[0]["transcribed"] = result["text"]
    assert summarize(rows)["gates"]["complete_endings"] is False


def test_unresolved_asr_repetition_cannot_be_scored(monkeypatch):
    from .evaluate import reliable_transcription

    monkeypatch.setitem(
        sys.modules,
        "mlx_whisper",
        SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: {
                "text": " una repetición," * 40,
                "segments": [{"compression_ratio": 5.9}],
            }
        ),
    )
    result = transcribe_audio("repeated.wav", "local-medium")
    assert result["reliable"] is False
    assert len(result["attempts"]) == 2
    assert not reliable_transcription({"text": "", "segments": []})


def test_unresolved_transcription_is_retained_but_never_promoted(tmp_path, monkeypatch):
    from pathlib import Path

    from . import evaluate
    from .corpus import sha256

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture audio")
    (tmp_path / "weights.npz").write_bytes(b"fixture recognizer")
    generation = {
        "identity": {"evaluation_code_sha256": sha256(Path(evaluate.__file__))},
        "results": [{"variant": "candidate", "id": "clip", "audio": str(audio), "sha256": sha256(audio)}],
    }
    (tmp_path / "generation-summary.json").write_text(json.dumps(generation))
    monkeypatch.setattr(
        evaluate,
        "transcribe_audio",
        lambda *_args: {
            "text": "repeated text",
            "reliable": False,
            "attempts": [{"result": {"text": "repeated text", "segments": [{"compression_ratio": 5.9}]}}],
        },
    )
    with pytest.raises(ValueError, match="Unresolved ASR"):
        evaluate.transcribe(tmp_path, tmp_path)
    assert not (tmp_path / "evaluation.json").exists()
    cache = tmp_path / "asr-candidate-clip.json"
    saved = json.loads(cache.read_text())
    assert saved["reliable"] is False
    assert saved["attempts"][0]["result"]["segments"][0]["compression_ratio"] == 5.9
    saved["evaluation_code_sha256"] = "0" * 64
    cache.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="ASR evaluation identity changed"):
        evaluate.transcribe(tmp_path, tmp_path)


def test_changed_asr_policy_requires_a_new_comparison(tmp_path):
    from .evaluate import transcribe

    (tmp_path / "generation-summary.json").write_text(
        json.dumps({"identity": {"evaluation_code_sha256": "0" * 64}, "results": []})
    )
    with pytest.raises(ValueError, match="Evaluation code changed"):
        transcribe(tmp_path, tmp_path)


@pytest.mark.parametrize("fail", [False, True])
def test_candidate_evaluation_uses_real_bounded_backend_and_propagates_limits(tmp_path, monkeypatch, fail):
    from backend import config
    from backend.backends.qwen_finetuned_backend import LocalQwenCustomVoiceBackend
    from backend.utils.chunked_tts import split_text_into_chunks

    from .evaluate import synthesize_sample

    core = SimpleNamespace(random=SimpleNamespace(seed=lambda _seed: None))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(config, "_data_dir", tmp_path / "data")
    config.initialize_data_permissions()
    calls = []

    def generate(_self, text, seed):
        calls.append((text, seed))
        if fail:
            raise RuntimeError("Fine-tuned model reached its duration limit before finishing")
        return np.ones(2400, dtype=np.float32), 24000

    monkeypatch.setattr(LocalQwenCustomVoiceBackend, "_generate", generate)
    text = "Una historia nueva empieza en la estación. " * 10
    model = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text: text.split()))
    if fail:
        with pytest.raises(RuntimeError, match="duration limit"):
            synthesize_sample(model, "candidate", text, 12, "fabian", None, None)
    else:
        audio, seconds, tokens = synthesize_sample(model, "candidate", text, 12, "fabian", None, None)
        assert calls == [(chunk, 12 + index) for index, chunk in enumerate(split_text_into_chunks(text, 200))]
        assert len(audio) > 0
        assert seconds >= 0
        assert not isinstance(audio, np.memmap)
        assert tokens is None  # The public backend reports audio; no token count is invented.
