from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOG_PATH = Path("logs") / "search_runs.jsonl"


def append_jsonl_log(path: str | None, payload: dict[str, Any]) -> str:
    target = Path(path) if path else DEFAULT_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(target.resolve())
