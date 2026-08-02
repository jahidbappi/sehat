"""Module entry point: ``python -m sehat.serving`` starts the API server.

Host/port come from ``SEHAT_HOST`` (default ``0.0.0.0``) and ``SEHAT_PORT``
(default ``8000``); the model artifact comes from ``SEHAT_MODEL_PATH``.
"""

from __future__ import annotations

import os

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def main() -> None:
    """Run uvicorn against the :func:`sehat.serving.app.create_app` factory."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            "uvicorn is required to serve Sehat; install the serving extras "
            "(e.g. `pip install sehat[serving]` or `pip install uvicorn fastapi`)."
        ) from exc

    host = os.environ.get("SEHAT_HOST", DEFAULT_HOST)
    port = int(os.environ.get("SEHAT_PORT", str(DEFAULT_PORT)))
    uvicorn.run(
        "sehat.serving.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=os.environ.get("SEHAT_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
