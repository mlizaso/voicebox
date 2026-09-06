"""Regression tests for conservative, whole-clip speaker rejection."""

from itertools import pairwise

import numpy as np
import pytest

from .speaker import RATE, accepted, speaker_scores, unit, windows


def test_windows_cover_tail_without_short_final_window():
    audio = np.arange(RATE * 8 + 17, dtype=np.float32)
    parts = windows(audio)
    assert all(len(part) == RATE * 3 for part in parts)
    assert parts[0][0] == 0
    assert parts[-1][-1] == audio[-1]
    assert all(b[0] <= a[-1] for a, b in pairwise(parts))


@pytest.mark.parametrize("audio", [np.ones(RATE - 1), np.ones((RATE, 2)), np.full(RATE, np.nan)])
def test_windows_reject_invalid_audio(audio):
    with pytest.raises(ValueError, match="Speaker audio"):
        windows(audio)


@pytest.mark.parametrize("vector", [np.zeros(2), np.array([1, np.nan]), np.array([np.inf])])
def test_unit_rejects_bad_vectors(vector):
    with pytest.raises(ValueError, match="Invalid speaker embedding"):
        unit(vector)


def test_one_guest_window_rejects_otherwise_matching_clip():
    targets = np.array([[1.0, 0.0]])
    negatives = np.array([[0.0, 1.0]])
    same = speaker_scores(np.array([2, 0]), targets, negatives)
    other = speaker_scores(np.array([0, 2]), targets, negatives)
    assert accepted([same] * 5, minimum_target=0.9, minimum_margin=0.1)
    assert not accepted([same] * 5 + [other], minimum_target=0.9, minimum_margin=0.1)
    assert not accepted([], minimum_target=0.9, minimum_margin=0.1)
