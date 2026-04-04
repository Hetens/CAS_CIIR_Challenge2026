from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys

from agentic_search.config import load_settings
from agentic_search.pipeline import AgenticSearchPipeline


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
        force=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic Search Challenge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cli_parser = subparsers.add_parser("cli", help="Run a single query")
    cli_parser.add_argument("query", help="Topic query to search for")
    cli_parser.add_argument("--debug", action="store_true", help="Include debug details")
    cli_parser.add_argument(
        "--log-file",
        default=None,
        help="Optional JSONL path for extracted search logs",
    )

    streamlit_parser = subparsers.add_parser("streamlit", help="Run the Streamlit UI")
    streamlit_parser.add_argument("--host", default="127.0.0.1")
    streamlit_parser.add_argument("--port", type=int, default=8501)

    args = parser.parse_args()
    if args.command == "cli":
        _configure_logging(args.debug)
        pipeline = AgenticSearchPipeline(load_settings())
        response = pipeline.run(
            args.query,
            debug=args.debug,
            log_path=args.log_file,
        )
        print(json.dumps(response.to_dict(), indent=2))
        return 0
    if args.command == "streamlit":
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "agentic_search/streamlit_app.py",
            "--server.address",
            args.host,
            "--server.port",
            str(args.port),
        ]
        return subprocess.call(command)
    parser.error("Unknown command")
    return 2
