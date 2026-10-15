"""Flask application exposing the gibberish-detection REST API.

Endpoints:
    POST /api/detect  -> classify one string (``{"text": ...}``) or a batch
                         (``{"texts": [...]}``).
    GET  /health      -> liveness/readiness probe.
"""

from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS
import hmac

import detector
from config import config


def _error(message: str, status: int):
    """Build a consistent JSON error response."""
    return jsonify({"error": message}), status


def _check_api_key() -> bool:
    """Validate the X-API-Key header when key enforcement is enabled."""
    if not config.require_api_key:
        return True
    provided = request.headers.get("X-API-Key", "")
    # Constant-time compare to avoid trivial timing leaks on the secret.
    return bool(config.api_key) and hmac.compare_digest(provided, config.api_key)


def _validate_text(value: object) -> tuple[bool, str]:
    """Return (ok, cleaned_text) for a single candidate input string."""
    if not isinstance(value, str):
        return False, "text must be a string"
    text = value.strip()
    if not text:
        return False, "text must not be empty"
    if len(text) > config.max_text_length:
        return False, f"text exceeds maximum length of {config.max_text_length} characters"
    return True, text


def create_app() -> Flask:
    app = Flask(__name__)

    # Restrict CORS to the configured front-end origin(s).
    CORS(app, resources={"/*": {"origins": config.allowed_origins}})

    @app.post("/api/detect")
    def detect():
        if not _check_api_key():
            return _error("missing or invalid X-API-Key header", 401)

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error("request body must be a JSON object", 400)

        has_single = "text" in payload
        has_batch = "texts" in payload
        if not has_single and not has_batch:
            return _error("provide either 'text' (string) or 'texts' (array)", 400)
        if has_single and has_batch:
            return _error("provide only one of 'text' or 'texts'", 400)

        if has_single:
            ok, result = _validate_text(payload["text"])
            if not ok:
                return _error(result, 400)
            return jsonify(detector.predict(result))

        # Batch path.
        raw_texts = payload["texts"]
        if not isinstance(raw_texts, list) or not raw_texts:
            return _error("'texts' must be a non-empty array", 400)

        cleaned: list[str] = []
        for index, item in enumerate(raw_texts):
            ok, result = _validate_text(item)
            if not ok:
                return _error(f"texts[{index}]: {result}", 400)
            cleaned.append(result)

        return jsonify({"results": detector.predict_many(cleaned)})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": detector.is_loaded()})

    @app.errorhandler(404)
    def not_found(_e):
        return _error("resource not found", 404)

    @app.errorhandler(405)
    def method_not_allowed(_e):
        return _error("method not allowed", 405)

    @app.errorhandler(500)
    def server_error(_e):
        return _error("internal server error", 500)

    return app


# Module-level app instance so `flask --app app run` can discover it.
app = create_app()
