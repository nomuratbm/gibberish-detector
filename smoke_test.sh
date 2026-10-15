#!/usr/bin/env bash
# Smoke test against a running gibberish detector server.
# Usage:  ./smoke_test.sh [BASE_URL] [API_KEY]
# Defaults match a local dev server with the .env API_KEY.
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:5000}"
API_KEY="${2:-change-me-to-a-long-random-string}"

echo "== Health =="
curl -s -o /dev/null -w "GET /health -> %{http_code}\n" "$BASE_URL/health"

echo "== Single clean sentence =="
curl -s -X POST "$BASE_URL/api/detect" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"text": "I love using this website."}'
echo

echo "== Single gibberish =="
curl -s -X POST "$BASE_URL/api/detect" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"text": "asdf qwerty zxcv jjkl"}'
echo

echo "== Batch =="
curl -s -X POST "$BASE_URL/api/detect" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"texts": ["Perfectly fine sentence", "xkcd qwl jhk"]}'
echo

echo "== Missing API key (expect 401) =="
curl -s -o /dev/null -w "-> %{http_code}\n" -X POST "$BASE_URL/api/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'

echo "== Empty text (expect 400) =="
curl -s -o /dev/null -w "-> %{http_code}\n" -X POST "$BASE_URL/api/detect" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"text": "   "}'

echo "Done."
