"""Tests for the deterministic heuristic layer.

Covers the tokenizer, the three rules, and the sentinel labels the
detector emits when Rules 1 and 2 fire (bypassing the model entirely).
"""

from unittest import mock

import pytest

import detector
import heuristics
from config import config


# ---------------------------------------------------------------------------
# Pure heuristic functions
# ---------------------------------------------------------------------------


def test_tokenize_splits_on_any_non_alnum_and_lowercases():
    assert heuristics.tokenize("Tech Week 2026!") == ["tech", "week", "2026"]
    assert heuristics.tokenize("hello, world.") == ["hello", "world"]


def test_alpha_letter_count_ignores_digits_and_punctuation():
    assert heuristics.alpha_letter_count("Tech Week 2026!") == 8
    assert heuristics.alpha_letter_count("12345") == 0


def test_has_real_word_matches_only_at_min_length():
    # "tech" is in the word set at length 4 exactly.
    assert heuristics.has_real_word("Tech 2026", min_length=4) is True
    # A shorter token like "the" doesn't count when min_length is 4.
    assert heuristics.has_real_word("the", min_length=4) is False
    # Unknown 4-letter string doesn't count.
    assert heuristics.has_real_word("zzqq", min_length=4) is False


def test_is_short_input_respects_both_limits():
    assert heuristics.is_short_input("Tech Week 2026", max_chars=20, max_tokens=4) is True
    # Too many chars AND too many tokens -> not short.
    assert (
        heuristics.is_short_input(
            "one two three four five six seven", max_chars=10, max_tokens=4
        )
        is False
    )


def test_force_gibberish_hits_repetitive_noise():
    # 21 alphabetic chars, no dictionary word, not short.
    assert heuristics.should_force_gibberish(
        "awawawaawawawaawawawa",
        min_word_len=4,
        min_alpha=5,
        short_max_chars=20,
        short_max_tokens=4,
    ) is True


def test_force_gibberish_skips_pure_numeric():
    assert heuristics.should_force_gibberish(
        "12345 67890 11223",
        min_word_len=4,
        min_alpha=5,
        short_max_chars=20,
        short_max_tokens=4,
    ) is False


def test_force_gibberish_skips_when_disabled_by_min_alpha_zero():
    assert heuristics.should_force_gibberish(
        "awawawaawawawaawawawa",
        min_word_len=4,
        min_alpha=0,
        short_max_chars=20,
        short_max_tokens=4,
    ) is False


def test_force_clean_protects_short_title_with_real_word():
    assert heuristics.should_force_clean(
        "Tech Week 2026",
        min_word_len=4,
        short_max_chars=20,
        short_max_tokens=4,
    ) is True


def test_force_clean_skips_when_no_real_word():
    # Short but nothing dictionary-valid in it.
    assert heuristics.should_force_clean(
        "zzqq ww",
        min_word_len=4,
        short_max_chars=20,
        short_max_tokens=4,
    ) is False


# ---------------------------------------------------------------------------
# Detector integration (rules firing around the mocked model)
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_config(monkeypatch):
    """Restore the heuristic settings to their defaults for each test."""
    monkeypatch.setattr(config, "enable_heuristics", True)
    monkeypatch.setattr(config, "min_real_word_length", 4)
    monkeypatch.setattr(config, "force_gibberish_min_alpha", 5)
    monkeypatch.setattr(config, "short_input_max_chars", 20)
    monkeypatch.setattr(config, "short_input_max_tokens", 4)
    monkeypatch.setattr(config, "confidence_threshold", 0.85)
    monkeypatch.setattr(config, "gibberish_labels", [])
    return monkeypatch


def test_predict_rule1_short_circuits_model(reset_config):
    # Model would say mild_gibberish (which is not in the flag set), but
    # Rule 1 fires first so the pipeline is never invoked.
    pipe = mock.Mock()
    reset_config.setattr(detector, "_pipe", pipe)
    result = detector.predict("awawawaawawawaawawawa")
    assert result == {
        "is_gibberish": True,
        "label": "heuristic:no_real_word",
        "score": 1.0,
    }
    pipe.assert_not_called()


def test_predict_rule2_short_circuits_model(reset_config):
    pipe = mock.Mock()
    reset_config.setattr(detector, "_pipe", pipe)
    result = detector.predict("Tech Week 2026")
    assert result == {
        "is_gibberish": False,
        "label": "heuristic:short_with_real_word",
        "score": 1.0,
    }
    pipe.assert_not_called()


def test_predict_falls_through_to_model(reset_config):
    fake_pipe = mock.Mock(return_value=[{"label": "clean", "score": 0.99}])
    reset_config.setattr(detector, "_pipe", fake_pipe)
    # Long enough and has real words, so neither rule fires; model decides.
    result = detector.predict("I love using this website for our annual event")
    assert result == {"is_gibberish": False, "label": "clean", "score": 0.99}
    fake_pipe.assert_called_once()


def test_predict_applies_confidence_gate(reset_config):
    # Model picks noise at 0.70, below the 0.85 threshold -> downgrade.
    fake_pipe = mock.Mock(return_value=[{"label": "noise", "score": 0.70}])
    reset_config.setattr(detector, "_pipe", fake_pipe)
    result = detector.predict("some input that does not trigger a heuristic rule")
    assert result["is_gibberish"] is False
    assert result["label"] == "noise"
    assert result["score"] == 0.70


def test_confidence_gate_can_be_disabled(reset_config):
    reset_config.setattr(config, "confidence_threshold", 0.0)
    fake_pipe = mock.Mock(return_value=[{"label": "noise", "score": 0.55}])
    reset_config.setattr(detector, "_pipe", fake_pipe)
    result = detector.predict("some input that does not trigger a heuristic rule")
    assert result["is_gibberish"] is True


def test_predict_many_only_batches_needed_strings(reset_config):
    fake_pipe = mock.Mock(return_value=[{"label": "clean", "score": 0.99}])
    reset_config.setattr(detector, "_pipe", fake_pipe)
    texts = [
        "awawawaawawawaawawawa",  # Rule 1 -> gibberish
        "Tech Week 2026",          # Rule 2 -> clean
        "Our company launched last quarter",  # needs model
    ]
    results = detector.predict_many(texts)
    assert results[0]["label"] == "heuristic:no_real_word"
    assert results[1]["label"] == "heuristic:short_with_real_word"
    assert results[2] == {"is_gibberish": False, "label": "clean", "score": 0.99}
    # Only the third string reached the model.
    fake_pipe.assert_called_once_with(["Our company launched last quarter"], truncation=True)
