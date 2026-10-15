"""Production entrypoint: preload the model, then serve via Waitress.

Waitress is a pure-Python WSGI server that runs the same way on Windows and
Linux, which suits a CPU-only host machine. Run with:

    python run.py
"""

from waitress import serve

import detector
from app import create_app
from config import config


def main() -> None:
    # Warm the model up before accepting traffic so the first real request
    # does not pay the one-time load cost.
    print("Loading gibberish model (first run downloads weights)...")
    detector.preload()
    print(f"Model ready. Serving on http://0.0.0.0:{config.port}")

    app = create_app()
    serve(app, host="0.0.0.0", port=config.port, threads=4)


if __name__ == "__main__":
    main()
