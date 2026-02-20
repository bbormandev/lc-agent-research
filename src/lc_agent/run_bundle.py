import json
import os
import secrets
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def _safe_json(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj

class RunBundler:
    def __init__(self, base_dir: str = "runs") -> None:
        self.base_dir = Path(base_dir)
        self.run_dir: Optional[Path] = None
        self.started_at: Optional[float] = None

    def start(self) -> str:
        stamp = _utc_now_stamp()
        suffix = secrets.token_hex(3)  # 6 hex chars
        run_id = f"{stamp}_{suffix}"

        day_dir = self.base_dir / _utc_today()
        self.run_dir = day_dir / run_id
        (self.run_dir / "fetch").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "extracts").mkdir(parents=True, exist_ok=True)

        self.started_at = time.time()
        return run_id

    def path(self) -> Path:
        if not self.run_dir:
            raise RuntimeError("RunBundler not started")
        return self.run_dir

    def write_json(self, rel_path: str, data: Any) -> None:
        p = self.path() / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(_safe_json(data), f, indent=2, ensure_ascii=False)

    def write_text(self, rel_path: str, text: str) -> None:
        p = self.path() / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def finish_meta(self, meta: dict) -> dict:
        if self.started_at is not None:
            meta["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            meta["elapsed_ms"] = int((time.time() - self.started_at) * 1000)
        return meta