import io
import zipfile
from itertools import pairwise

import numpy as np

from scripts.finetune_qwen.corpus import (
    SAMPLE_RATE,
    candidate_segments,
    epub_text,
    headroom,
    matched_runs,
    quiet_cut,
)


def test_epub_keeps_words_on_both_sides_of_inline_markup():
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("chapter.xhtml", "<html><body><h1>Título</h1><p>La <em>voz</em> continúa.</p></body></html>")
    with zipfile.ZipFile(memory) as archive:
        assert epub_text(archive, ["chapter.xhtml"]) == "Título\n\nLa voz continúa."


def test_alignment_never_uses_printed_words_that_were_not_spoken():
    text = "Uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince dieciséis diecisiete dieciocho."
    words = [{"w": word, "s": i * 0.5, "e": i * 0.5 + 0.3} for i, word in enumerate(text.split())]
    words[8]["w"] = "DISTINTO"
    runs = matched_runs(text, words, 20)
    assert [len(run) for run in runs] == [8, 9]
    assert all(text[word["char_start"] : word["char_end"]] != "nueve" for run in runs for word in run)


def test_zero_length_and_overlapping_timestamps_break_runs():
    text = " ".join(f"palabra{i}" for i in range(30))
    words = [{"w": word, "s": i, "e": i + 0.5} for i, word in enumerate(text.split())]
    words[10]["e"] = words[10]["s"]
    words[20]["s"] = 18.5
    runs = matched_runs(text, words, 31)
    assert all(word["s"] != 10 for run in runs for word in run)
    assert all(current["s"] >= previous["e"] - 0.04 for run in runs for previous, current in pairwise(run))


def test_quiet_cut_refuses_to_cut_inside_continuous_speech():
    audio = np.full(SAMPLE_RATE, 0.1, dtype=np.float32)
    assert quiet_cut(audio, 0.45, 0.55) is None
    audio[int(0.46 * SAMPLE_RATE) : int(0.54 * SAMPLE_RATE)] = 0
    cut = quiet_cut(audio, 0.45, 0.55)
    assert cut is not None
    assert 0.46 <= cut <= 0.54


def test_decoded_aac_overshoots_are_scaled_without_clipping_or_compression():
    source = np.array([-1.2, -0.1, 0.0, 0.2, 1.1], dtype=np.float32)
    converted, gain = headroom(source)
    np.testing.assert_allclose(converted / gain, source)
    assert np.max(np.abs(converted)) <= 0.901
    assert np.array_equal(source, np.array([-1.2, -0.1, 0.0, 0.2, 1.1], dtype=np.float32))


def test_segments_have_real_silence_boundaries_and_retain_punctuation():
    text = "¿Uno dos tres cuatro cinco seis siete ocho nueve diez?"
    words = [{"w": word, "s": 0.5 + i * 0.5, "e": 0.8 + i * 0.5} for i, word in enumerate(text.split())]
    audio = np.zeros(6 * SAMPLE_RATE, dtype=np.float32)
    for word in words:
        audio[round(word["s"] * SAMPLE_RATE) : round(word["e"] * SAMPLE_RATE)] = 0.1
    segments = candidate_segments(text, matched_runs(text, words, 6), audio)
    assert len(segments) == 1
    assert segments[0]["text"] == text
    assert segments[0]["start"] < words[0]["s"]
    assert segments[0]["end"] > words[-1]["e"]
