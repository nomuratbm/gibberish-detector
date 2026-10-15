# Setup Guide - Gibberish Detector API

This guide walks you through getting the gibberish detector running on your
hosting computer, exposing it with ngrok, and wiring it into your AWS Amplify
site so a text field can be flagged as an error when the input is gibberish.

It assumes no prior backend experience.

---

## 0. What you are building (one picture)

```
[Amplify website (browser)]
        |  HTTPS POST /api/detect  { "text": "..." }
        v
[ngrok public URL]  ->  tunnel  ->  [Flask server on the other computer :5000]
                                            |
                                            v
                                   [HuggingFace DistilBERT model (CPU)]
```

The browser sends the field's text to the API. The API returns
`{ "is_gibberish": true/false, ... }`. If `true`, your front-end marks the
field as an error.

---

## 1. Prerequisites on the hosting computer

1. **Python 3.10, 3.11, or 3.12** (64-bit).
   - Important: the ML library `torch` does not provide wheels for Python
     3.13/3.14 yet. If 3.14 is installed, install 3.12 alongside it and use it
     explicitly (see step 2).
   - Download from https://www.python.org/downloads/ (check "Add to PATH").
2. **ngrok** installed and signed up at https://ngrok.com (free tier is fine
   for testing).
3. **Internet access** for the first run (downloads ~250 MB of model weights;
   later runs are cached locally).

---

## 2. Create the project's virtual environment

From this project folder:

```bash
# Windows, if you installed 3.12 explicitly:
py -3.12 -m venv .venv

# Then activate it:
.venv\Scripts\activate
```

Once activated, your prompt shows `(.venv)`. Verify:

```bash
python --version   # should show 3.10/3.11/3.12
```

> If you see 3.14 here, you activated with the wrong interpreter. Recreate the
> venv with `py -3.12 -m venv .venv`.

---

## 3. Install dependencies

```bash
# App dependencies (Flask server + the HuggingFace model stack)
pip install -r requirements.txt

# Development/test dependencies
pip install -r requirements-dev.txt
```

If `pip install -r requirements.txt` fails while resolving `torch`, your venv
is on an unsupported Python version. Go back to step 2.

---

## 4. Configure the app

```bash
# Windows
copy .env.example .env
```

Open `.env` and set at least these two values:

```ini
# A long random secret only your site knows. Generate one, e.g. run:
#   python -c "import secrets; print(secrets.token_urlsafe(32))"
API_KEY=<paste-the-random-string>

# Your Amplify site's domain, so CORS allows it. Example:
ALLOWED_ORIGINS=https://main.d1a2b3c4d5.amplifyapp.com
```

Other keys (`MODEL_ID`, `MAX_TEXT_LENGTH`, `REQUIRE_API_KEY`, `PORT`) are
already sensible defaults; change them only if needed. Full list in `README.md`.

---

## 5. Run the tests (sanity check)

These run offline with the model mocked, so they are fast:

```bash
pytest
```

You should see all tests pass. This confirms the API's routing, validation,
auth guard, and response shape are correct before you involve the real model.

---

## 6. Start the server (with the real model)

```bash
python run.py
```

- The first launch downloads the model, so it can take a minute or two. Wait
  for: `Model ready. Serving on http://0.0.0.0:5000`.
- Keep this terminal open; it is the running server.

In a second terminal, confirm it responds:

```bash
curl http://127.0.0.1:5000/health
```

Then run the full end-to-end smoke test against the local server:

```bash
bash smoke_test.sh http://127.0.0.1:5000 <your-API_KEY>
```

You should see a clean sentence return `"is_gibberish": false`, gibberish
return `true`, and the "missing API key" / "empty text" cases return `401` and
`400`.

---

## 7. Expose it publicly with ngrok

In a new terminal (leave the server running):

```bash
# One-time setup
ngrok config add-authtoken <YOUR_AUTHTOKEN>

# Forward the public HTTPS endpoint to the local server
ngrok http 5000
```

ngrok prints a public URL like `https://ab12cd34.ngrok-free.app`. That is the
`API_URL` your website will call.

Tips:

- **The free URL changes every restart.** Reserve a stable domain in the ngrok
  dashboard (Domains) and launch with
  `ngrok http --domain=your-name.ngrok-free.app 5000`.
- ngrok's free plan also has a **rate limit (~20 requests/min)**. Good enough
  for demos; for production traffic, use a paid plan or a real host/reverse
  proxy later.
- If a browser ever gets ngrok's interstitial page, add the request header
  `ngrok-skip-browser-warning: true` from your front-end (plain JSON `fetch`
  calls are usually unaffected).

---

## 8. Wire it into your Amplify site

Give your front-end the two values via Amplify environment variables (Amplify
Console -> App settings -> Environment variables), then read them in code:

- `REACT_APP_GIBBERISH_API_URL` = the ngrok https URL (no trailing slash)
- `REACT_APP_GIBBERISH_API_KEY` = the same `API_KEY` you put in `.env`

Example validation helper:

```js
const API_URL = process.env.REACT_APP_GIBBERISH_API_URL;
const API_KEY = process.env.REACT_APP_GIBBERISH_API_KEY;

export async function isGibberish(text) {
  // Skip the network call for obviously empty input.
  if (!text || !text.trim()) return false;
  const res = await fetch(`${API_URL}/api/detect`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
      "ngrok-skip-browser-warning": "true",
    },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    // Fail open: a detector outage should not block your users.
    console.warn("Gibberish check failed:", res.status);
    return false;
  }
  const data = await res.json();
  return data.is_gibberish === true;
}
```

Use it on submit or field blur:

```js
if (await isGibberish(value)) {
  setError("This text does not look valid.");
}
```

To send many fields at once (for example, a whole form), post `{"texts": [...]}`
and read `results`.

> Security note: an `API_KEY` baked into front-end code is visible to anyone
> who inspects the site, so it only deters casual abuse. If that matters, put a
> small Amplify backend (API Gateway/Lambda) in front of this API and call that
> instead; the endpoint contract here does not change.

---

## 9. Keeping it running

`python run.py` runs in the foreground. On the hosting computer you will want
it to survive reboots and log out. Options:

- Simplest: keep an ngrok tunnel config and start both on login (Task Scheduler
  on Windows).
- More robust: run behind a Windows service (NSSM) or a process manager, and
  use `ngrok` as a service.

---

### Tuning the classifier (false positives / negatives)

The ML model has two known quirks: repetitive noise like
`awawawaawawawaawawawa` gets labeled `mild_gibberish`, and short titles
like `Tech Week 2026` do too. See the **Heuristics** section of
[README.md](README.md) for the three-rule fix already wired in. Adjust it
via these env vars in `.env`:

```
ENABLE_HEURISTICS=true
MIN_REAL_WORD_LENGTH=4
FORCE_GIBBERISH_MIN_ALPHA=5
SHORT_INPUT_MAX_CHARS=20
SHORT_INPUT_MAX_TOKENS=4
CONFIDENCE_THRESHOLD=0.85
```

If a legitimate word in your field is being flagged as gibberish, add it
to `_COMMON_WORDS` in [heuristics.py](heuristics.py). If a specific kind
of gibberish still slips through, lower `MIN_REAL_WORD_LENGTH` to `3` or
raise `FORCE_GIBBERISH_MIN_ALPHA` for stricter noise detection.
Remember to restart `python run.py` after any change.

## 10. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `pip install torch` fails | Venv is on Python 3.13/3.14. Recreate with `py -3.12`. |
| `/health` works locally but browser call fails | ngrok not running, wrong `API_URL`, or CORS blocked. Add the Amplify domain to `ALLOWED_ORIGINS`. |
| Browser gets an HTML page instead of JSON | ngrok interstitial. Add `ngrok-skip-browser-warning: true`. |
| `401 Unauthorized` | `X-API-Key` missing or does not match `.env` `API_KEY`. |
| `400` on valid text | Text exceeds `MAX_TEXT_LENGTH`, or empty after trimming. |
| First request very slow | Model download/load. `python run.py` preloads it at boot. |

---

## Quick command recap

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
copy .env.example .env   # then edit API_KEY and ALLOWED_ORIGINS
pytest                    # offline sanity check
python run.py             # start server (downloads model on first run)
ngrok http 5000           # in another terminal -> public HTTPS URL
```
