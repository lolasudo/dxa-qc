"""``python -m dxa_qc.api [--host 0.0.0.0] [--port 8000] [--reload]`` - one process, models loaded once.

``--reload`` (development only) restarts the server when the sources change.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="python -m dxa_qc.api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="development: restart on source changes")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from dxa_qc.config import load_settings

    api = load_settings().api
    uvicorn.run(
        "dxa_qc.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=1,  # models are loaded once per process and use one GPU
        limit_concurrency=api.max_connections,  # HTTP 503 beyond, instead of exhausting the server
        timeout_keep_alive=5,  # idle keep-alive connections are not allowed to pile up
        server_header=False,
        date_header=False,
        reload=args.reload,
        reload_dirs=[str(Path(__file__).resolve().parents[1])] if args.reload else None,
    )


if __name__ == "__main__":
    main()
