"""Source binding and conservative timestamp filtering for mixed recordings."""

import json
import unicodedata

import pytest

from .episodes import inventory, timestamp_hypotheses


@pytest.fixture
def episode_config(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    timestamps = tmp_path / "timestamps"
    audio.mkdir()
    timestamps.mkdir()
    for episode in range(1, 4):
        name = f"Ep {episode} filósofos"
        (audio / unicodedata.normalize("NFD", name + ".m4b")).write_bytes(b"source")
        (timestamps / (name + ".json")).write_text(
            json.dumps(
                {
                    "audio_file": name + ".m4b",
                    "language": "es",
                    "word_timestamps": True,
                    "sanity": {"ok": True},
                    "ep": episode,
                    "dur": 100.0,
                    "model": "timestamp-model",
                }
            )
        )
    monkeypatch.setattr("scripts.finetune_qwen.episodes.subprocess.check_output", lambda _command: b"100.0\n")
    return {
        "audio_directory": str(audio),
        "timestamps": str(timestamps),
        "validation_episodes": [2],
        "test_episodes": [3],
        "expected_episodes": 3,
    }


def test_inventory_matches_unicode_and_binds_actual_source(episode_config):
    episodes = inventory(episode_config)
    assert [episode["split"] for episode in episodes] == ["train", "validation", "test"]
    assert all(len(episode["sha256"]) == 64 for episode in episodes)
    assert all(len(episode["timestamp_sha256"]) == 64 for episode in episodes)


def test_inventory_rejects_duration_drift(episode_config, monkeypatch):
    monkeypatch.setattr("scripts.finetune_qwen.episodes.subprocess.check_output", lambda _command: b"101.0\n")
    with pytest.raises(ValueError, match="duration mismatch"):
        inventory(episode_config)


def test_inventory_rejects_overlapping_holdouts(episode_config):
    episode_config["test_episodes"] = [2]
    with pytest.raises(ValueError, match="disjoint"):
        inventory(episode_config)


def test_timestamp_filter_excludes_wrappers_low_confidence_and_suspicious_segments():
    asr = {
        "words": [
            {"w": "intro", "start": 19.8, "end": 20.2, "prob": 1, "seg": 0},
            {"w": "narrador", "start": 20.5, "end": 21, "prob": 0.99, "seg": 0},
            {"w": "incierto", "start": 22, "end": 23, "prob": 0.89, "seg": 0},
            {"w": "silencio", "start": 24, "end": 25, "prob": 1, "seg": 1},
            {"w": "outro", "start": 74.9, "end": 75.2, "prob": 1, "seg": 0},
        ],
        "segments": [
            {"id": 0, "no_speech_prob": 0.01, "avg_logprob": -0.1, "compression_ratio": 1.2},
            {"id": 1, "no_speech_prob": 0.5, "avg_logprob": -0.1, "compression_ratio": 1.2},
        ],
    }
    text, words = timestamp_hypotheses(asr, 100)
    assert text == "intro narrador incierto silencio outro"
    assert [word["w"] for word in words] == ["", "narrador", "", "", ""]
