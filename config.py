"""Environment-based configuration for the gibberish detector API.

Values are read from the process environment (optionally loaded from a
local `.env` file via python-dotenv). See `.env.example` for documentation
of each key.
"""

import os

from dotenv import load_dotenv

# Load variables from a local `.env` file (if present) into the environment.
load_dotenv()


def _split_csv(raw: str) -> list[str]:
    """Split a comma-separated env value into a clean list."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _as_bool(raw: str, default: bool = False) -> bool:
    """Interpret common truthy strings as True; everything else falls back."""
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Simple container exposing typed settings read from the environment."""

    def __init__(self) -> None:
        self.model_id: str = os.getenv(
            "MODEL_ID", "madhurjindal/autonlp-Gibberish-Detector-492513457"
        )

        # Explicit gibberish labels. Empty -> detector uses "not clean" rule.
        self.gibberish_labels: list[str] = _split_csv(
            os.getenv("GIBBERISH_LABELS", "")
        )

        try:
            self.max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "512"))
        except ValueError:
            self.max_text_length = 512

        self.allowed_origins: list[str] = _split_csv(
            os.getenv("ALLOWED_ORIGINS", "")
        )

        self.require_api_key: bool = _as_bool(
            os.getenv("REQUIRE_API_KEY", "true"), default=True
        )
        self.api_key: str = os.getenv("API_KEY", "")

        try:
            self.port: int = int(os.getenv("PORT", "5000"))
        except ValueError:
            self.port = 5000

        # --- Heuristic rules layered around the model (see heuristics.py) ---
        self.enable_heuristics: bool = _as_bool(
            os.getenv("ENABLE_HEURISTICS", "true"), default=True
        )
        # Rule 1 (force gibberish when no dictionary word is present).
        self.min_real_word_length: int = _as_int(
            os.getenv("MIN_REAL_WORD_LENGTH"), default=4
        )
        self.force_gibberish_min_alpha: int = _as_int(
            os.getenv("FORCE_GIBBERISH_MIN_ALPHA"), default=5
        )
        # Rule 2 (force clean for short inputs that contain a real word).
        self.short_input_max_chars: int = _as_int(
            os.getenv("SHORT_INPUT_MAX_CHARS"), default=20
        )
        self.short_input_max_tokens: int = _as_int(
            os.getenv("SHORT_INPUT_MAX_TOKENS"), default=4
        )
        # Rule 3 (confidence gate on the model's gibberish output).
        try:
            self.confidence_threshold: float = float(
                os.getenv("CONFIDENCE_THRESHOLD", "0.85")
            )
        except ValueError:
            self.confidence_threshold = 0.85


def _as_int(raw: str | None, default: int) -> int:
    """Parse an int env value with a safe fallback."""
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Single shared instance imported by the app modules.
config = Config()
