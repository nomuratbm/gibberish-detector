# Gibberish Detector API

A small Flask service that wraps a fine-tuned HuggingFace gibberish
classifier ([`madhurjindal/autonlp-Gibberish-Detector-492513457`](https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457),
DistilBERT, English). Your AWS Amplify front-end calls the JSON API when a
user enters text in a field; if the response says the text is gibberish, the
front-end marks the field as an error.

The model runs on CPU and is loaded once at startup and reused across
requests.

## Requirements

- **Python 3.10 - 3.12** (recommended). Note: `torch` does not publish wheels
  for very new Python versions such as 3.14, so create your virtual environment
  with a supported interpreter (for example, `py -3.12 -m venv .venv` on
  Windows).
- Internet access on first run to download the model weights (roughly 250 MB).
  Later runs use the local HuggingFace cache.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your local config from the template
copy .env.example .env        # Windows  (cp .env.example .env elsewhere)
# then edit .env (set API_KEY, ALLOWED_ORIGINS, etc.)
```

## Configuration

All settings come from environment variables (loaded from `.env`). See
`.env.example` for the full list:

| Variable           | Purpose                                                                      | Default                                       |
| ------------------ | ---------------------------------------------------------------------------- | --------------------------------------------- |
| `MODEL_ID`         | HuggingFace model id.                                                        | madhurjindal/autonlp-Gibberish-Detector-...   |
| `GIBBERISH_LABELS` | Comma list of labels treated as gibberish. Empty = "anything not `clean`".   | (empty)                                       |
| `MAX_TEXT_LENGTH`  | Reject inputs longer than this many characters.                             | 512                                           |
| `ALLOWED_ORIGINS`  | Comma list of CORS origins (your Amplify domain).                           | (empty)                                       |
| `REQUIRE_API_KEY`  | When true, `/api/detect` requires a matching `X-API-Key` header.            | true                                          |
| `API_KEY`          | Shared secret sent in the `X-API-Key` header.                               | (empty)                                       |
| `PORT`             | Port the server listens on (ngrok forwards to this).                        | 5000                                          |

## Running

Development (auto-reload, model loads on first request):

```bash
flask --app app run --debug
```

Production (preloads the model at boot, then serves via Waitress):

```bash
python run.py
```

## API

### `POST /api/detect`

Single input:

```json
{ "text": "asdf qwerty zxcv" }
```

Batch input:

```json
{ "texts": ["I love this website", "dfg hjk lpo"] }
```

Response (single):

```json
{ "is_gibberish": true, "label": "noise", "score": 0.98 }
```

Response (batch):

```json
{ "results": [ { "is_gibberish": false, "label": "clean", "score": 0.99 } ] }
```

Send the `X-API-Key` header when `REQUIRE_API_KEY=true`.

### `GET /health`

Returns `{ "status": "ok", "model_loaded": true }`.

## Heuristics

The DistilBERT classifier alone has two well-known failure modes that matter
for form fields:

- Repetitive noise like `awawawaawawawaawawawa` gets classified as
  `mild_gibberish` because the tokenizer sees valid-looking pieces.
- Short, title-cased strings like `Tech Week 2026` also come back as
  `mild_gibberish` because they lack sentence structure.

We layer three cheap, deterministic rules around the model in
[heuristics.py](heuristics.py):

1. **Force gibberish** - if the string has at least
   `FORCE_GIBBERISH_MIN_ALPHA` alphabetic characters, is not short, and
   contains no dictionary word of length `MIN_REAL_WORD_LENGTH` or more,
   return `is_gibberish: true` with the sentinel label
   `heuristic:no_real_word` and skip the model.
2. **Force clean** - if the string is short (both `SHORT_INPUT_MAX_CHARS`
   and `SHORT_INPUT_MAX_TOKENS` are met) AND contains a real dictionary
   word, return `is_gibberish: false` with the sentinel label
   `heuristic:short_with_real_word` and skip the model.
3. **Confidence gate** - if the model returned a gibberish label with a
   softmax score below `CONFIDENCE_THRESHOLD`, downgrade the response to
   `is_gibberish: false` (the raw label and score are still returned for
   debugging).

The built-in word set in `heuristics.py` is a compact, curated list of common
English words. Extend it with your own product names, event titles, or
industry terms whenever you notice the detector flagging legitimate input.

Every rule has an env-var knob (see `.env.example`), and
`ENABLE_HEURISTICS=false` reverts to raw model output for A/B comparison.

## Testing

- Offline unit tests (model is mocked, no downloads): `pip install -r
  requirements-dev.txt` then `pytest`.
- Live smoke test against a running server:
  `bash smoke_test.sh http://127.0.0.1:5000 <API_KEY>`.

For a full step-by-step walkthrough (Python version, ngrok, Amplify wiring,
troubleshooting), see [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Exposing it publicly with ngrok

The hosting computer runs the server locally; ngrok provides the public HTTPS
URL that the Amplify site calls.

```bash
# One-time: authenticate your ngrok agent
ngrok config add-authtoken <YOUR_AUTHTOKEN>

# Forward the public HTTPS endpoint to the local server
ngrok http 5000
```

Useful notes:

- Copy the `https://<id>.ngrok-free.app` URL ngrok prints and put it in your
  Amplify app's configuration as the API base URL.
- **Free-tier URLs change each time you restart ngrok.** Reserve a static
  domain (ngrok dashboard -> Domains) and run `ngrok http --domain=<x> 5000`
  so the URL stays stable.
- Free-tier ngrok shows a **browser interstitial** and applies a **request
  rate limit (about 20 requests/minute)**. The interstitial does not affect
  plain `fetch`/API calls, but if you do hit it, send the header
  `ngrok-skip-browser-warning: true`. For real traffic, upgrade ngrok or move
  to a proper reverse proxy (Caddy/nginx) later.
- ngrok terminates TLS, so no separate reverse proxy is required for now.

## Calling it from your Amplify front-end

Example check when a text field changes or is submitted:

```js
const API_URL = process.env.REACT_APP_GIBBERISH_API_URL; // the ngrok https URL
const API_KEY = process.env.REACT_APP_GIBBERISH_API_KEY;

async function checkGibberish(text) {
  const res = await fetch(`${API_URL}/api/detect`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
    },
    body: JSON.stringify({ text }),
  });
  const { is_gibberish } = await res.json();
  return is_gibberish; // if true -> mark the field as an error
}
```

Add your Amplify domain to `ALLOWED_ORIGINS` so CORS preflight succeeds.
Understand the trade-off: any key shipped in front-end code is public, so
`API_KEY` mainly deters casual abuse. If that is not acceptable, put a small
Amplify backend (API Gateway/Lambda) in front of this API later; the endpoint
contract does not change.

## Project layout

```
app.py         Flask app + routes
detector.py    Model loading + prediction logic
config.py      Environment-based settings
run.py         Waitress production entrypoint
requirements.txt
.env.example
```
