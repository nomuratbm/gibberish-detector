"""Thin wrapper around the HuggingFace gibberish-classification pipeline.

The pipeline is loaded once (lazily) and reused across requests. CPU is
forced via ``device=-1`` since the service is intended to run on a
CPU-only machine. Two cheap heuristic rules (see ``heuristics.py``) run
around the model to fix consistent failure modes:

* Rule 1 - force-gibberish when a longer alphabetic string contains no
  dictionary word (catches "awawawa..." that the model mislabels as
  `mild_gibberish`).
* Rule 2 - force-clean when a short string does contain a dictionary word
  (protects event titles like "Tech Week 2026").

Rule 3 - a confidence gate on the model's gibberish output - is applied on
the way back. Each rule can be tuned or disabled via env vars; see
``.env.example``.
"""

from __future__ import annotations

import threading
from typing import Any

from config import config
import heuristics

# The label the model uses for well-formed, meaningful text.
_CLEAN_LABEL = "clean"

# Sentinel labels returned when a heuristic rule fires; useful for
# debugging from the API response without changing the `is_gibberish`
# contract the front-end sees.
_LABEL_FORCE_GIBBERISH = "heuristic:no_real_word"
_LABEL_FORCE_CLEAN = "heuristic:short_with_real_word"

_pipe = None
_lock = threading.Lock()


def _get_pipe():
    """Return the shared text-classification pipeline, loading it on demand.

    Loading a transformer is relatively slow, so it happens once and the
    instance is reused. The lock keeps concurrent first-requests from
    loading the model more than once.
    """
    global _pipe
    if _pipe is not None:
        return _pipe
    with _lock:
        if _pipe is None:
            # Imported lazily so lightweight consumers (e.g. the offline test
            # suite, which patches predict/predict_many) do not need the heavy
            # torch/transformers stack installed just to import this module.
            from transformers import pipeline

            _pipe = pipeline(
                task="text-classification",
                model=config.model_id,
                device=-1,  # force CPU inference
            )
    return _pipe


def is_loaded() -> bool:
    """True if the model has already been loaded into memory."""
    return _pipe is not None


def preload() -> None:
    """Eagerly load the model (used by the production entrypoint at boot)."""
    _get_pipe()


def _is_gibberish_label(label: str) -> bool:
    """Map a raw model label to a boolean gibberish decision.

    When ``GIBBERISH_LABELS`` is configured, only those labels count as
    gibberish. Otherwise the default rule applies: anything that is not
    ``clean`` is treated as gibberish. Comparison is case-insensitive so
    the logic stays robust to the exact casing of the model's labels.
    """
    normalized = label.strip().lower()
    if config.gibberish_labels:
        explicit = {item.strip().lower() for item in config.gibberish_labels}
        return normalized in explicit
    return normalized != _CLEAN_LABEL


def _shape(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a pipeline result into the API's response shape."""
    label = str(raw.get("label", ""))
    return {
        "is_gibberish": _is_gibberish_label(label),
        "label": label,
        "score": float(raw.get("score", 0.0)),
    }


# ---------------------------------------------------------------------------
# Heuristic layer
# ---------------------------------------------------------------------------


def _decide_by_rules(text: str) -> dict[str, Any] | None:
    """Return a final decision if Rule 1 or Rule 2 fires, else ``None``."""
    if not config.enable_heuristics:
        return None
    if heuristics.should_force_gibberish(
        text,
        min_word_len=config.min_real_word_length,
        min_alpha=config.force_gibberish_min_alpha,
        short_max_chars=config.short_input_max_chars,
        short_max_tokens=config.short_input_max_tokens,
    ):
        return {
            "is_gibberish": True,
            "label": _LABEL_FORCE_GIBBERISH,
            "score": 1.0,
        }
    if heuristics.should_force_clean(
        text,
        min_word_len=config.min_real_word_length,
        short_max_chars=config.short_input_max_chars,
        short_max_tokens=config.short_input_max_tokens,
    ):
        return {
            "is_gibberish": False,
            "label": _LABEL_FORCE_CLEAN,
            "score": 1.0,
        }
    return None


def _apply_confidence_gate(model_out: dict[str, Any]) -> dict[str, Any]:
    """Rule 3: if the model called it gibberish but is unsure, downgrade."""
    if (
        config.confidence_threshold > 0
        and model_out["is_gibberish"]
        and model_out["score"] < config.confidence_threshold
    ):
        return {**model_out, "is_gibberish": False}
    return model_out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict(text: str) -> dict[str, Any]:
    """Classify a single string and return {is_gibberish, label, score}."""
    decided = _decide_by_rules(text)
    if decided is not None:
        return decided
    raw = _shape(_get_pipe()(text, truncation=True)[0])
    return _apply_confidence_gate(raw)


def predict_many(texts: list[str]) -> list[dict[str, Any]]:
    """Classify a batch of strings, preserving input order in the output.

    Rules 1 and 2 short-circuit without touching the model, so a mixed batch
    only pays inference for the strings that genuinely need it.
    """
    results: list[dict[str, Any] | None] = [None] * len(texts)
    model_needed: list[tuple[int, str]] = []
    for index, text in enumerate(texts):
        decided = _decide_by_rules(text)
        if decided is not None:
            results[index] = decided
        else:
            model_needed.append((index, text))

    if model_needed:
        outs = _get_pipe()([t for _, t in model_needed], truncation=True)
        for (index, _), raw in zip(model_needed, outs):
            results[index] = _apply_confidence_gate(_shape(raw))

    # The list is fully populated at this point; help type checkers.
    return [r for r in results if r is not None]
