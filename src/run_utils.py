import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def resolve_run_id(run_id: Optional[str], run_name: Optional[str], default: str) -> str:
    if run_id:
        return run_id
    if run_name:
        return run_name
    return default


def get_git_state(root: Path) -> Dict[str, Optional[str]]:
    git_dir = root / ".git"
    if not git_dir.exists():
        return {"commit": None, "branch": None, "dirty": None}

    def _run(cmd: list[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                cmd,
                cwd=str(root),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    dirty = None if status is None else bool(status)

    return {"commit": commit, "branch": branch, "dirty": dirty}


def write_metadata(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def resolve_ultralytics_output_dir(result: Any, fallback_dir: Path) -> Path:
    save_dir = getattr(result, "save_dir", None)
    if save_dir is None:
        return fallback_dir
    return Path(save_dir)
