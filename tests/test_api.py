"""Endpoint tests for the gibberish detector API.

The heavy HuggingFace model is never invoked here: ``detector.predict`` and
``detector.predict_many`` are patched with stubs, so these tests run fast and
offline. They cover routing, validation, the API-key guard, and the response
shape your Amplify front-end depends on.
"""

from unittest import mock

import pytest

import app as app_module
import detector
from config import config

API_KEY = "test-secret"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config, "require_api_key", True)
    monkeypatch.setattr(config, "api_key", API_KEY)
    monkeypatch.setattr(config, "allowed_origins", ["http://localhost:5173"])
    monkeypatch.setattr(config, "max_text_length", 50)
    flask_app = app_module.create_app()
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def _detect_headers():
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def test_health_reports_model_state(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_detect_single_gibberish(client):
    with mock.patch.object(
        detector, "predict", return_value={"is_gibberish": True, "label": "noise", "score": 0.97}
    ):
        resp = client.post("/api/detect", json={"text": "asdf qwerty"}, headers=_detect_headers())
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"is_gibberish": True, "label": "noise", "score": 0.97}


def test_detect_single_clean(client):
    with mock.patch.object(
        detector, "predict", return_value={"is_gibberish": False, "label": "clean", "score": 0.99}
    ):
        resp = client.post(
            "/api/detect", json={"text": "I love this website"}, headers=_detect_headers()
        )
    assert resp.status_code == 200
    assert resp.get_json()["is_gibberish"] is False


def test_detect_batch(client):
    stub = [
        {"is_gibberish": False, "label": "clean", "score": 0.99},
        {"is_gibberish": True, "label": "noise", "score": 0.95},
    ]
    with mock.patch.object(detector, "predict_many", return_value=stub) as patched:
        resp = client.post(
            "/api/detect", json={"texts": ["good sentence", "zzx cvv"]}, headers=_detect_headers()
        )
    assert resp.status_code == 200
    assert resp.get_json() == {"results": stub}
    patched.assert_called_once_with(["good sentence", "zzx cvv"])


def test_missing_api_key_is_unauthorized(client):
    resp = client.post("/api/detect", json={"text": "hello"})
    assert resp.status_code == 401
    assert "error" in resp.get_json()


def test_wrong_api_key_is_unauthorized(client):
    resp = client.post(
        "/api/detect", json={"text": "hello"}, headers={"X-API-Key": "wrong"}
    )
    assert resp.status_code == 401


def test_empty_text_is_rejected(client):
    resp = client.post("/api/detect", json={"text": "   "}, headers=_detect_headers())
    assert resp.status_code == 400


def test_oversized_text_is_rejected(client):
    resp = client.post(
        "/api/detect", json={"text": "a" * 51}, headers=_detect_headers()
    )
    assert resp.status_code == 400


def test_non_string_text_is_rejected(client):
    resp = client.post("/api/detect", json={"text": 123}, headers=_detect_headers())
    assert resp.status_code == 400


def test_both_text_and_texts_is_rejected(client):
    resp = client.post(
        "/api/detect",
        json={"text": "a", "texts": ["b"]},
        headers=_detect_headers(),
    )
    assert resp.status_code == 400


def test_neither_field_is_rejected(client):
    resp = client.post("/api/detect", json={"foo": "bar"}, headers=_detect_headers())
    assert resp.status_code == 400


def test_invalid_json_body_is_rejected(client):
    resp = client.post(
        "/api/detect",
        data="not json",
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_unknown_route_returns_json_404(client):
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404
    assert "error" in resp.get_json()
