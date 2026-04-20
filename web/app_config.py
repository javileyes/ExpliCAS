from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
BUILD_CONFIG_PATH = Path(__file__).resolve().parent / "build-config.js"

DEFAULT_WEB_TIME_BUDGET_MS = 1700
DEFAULT_WEB_MIN_HARD_TIMEOUT_SECONDS = 8
DEFAULT_WEB_TIMEOUT_GRACE_SECONDS = 5


def _parse_dotenv(path: Path = DOTENV_PATH) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and value[0] in {"'", '"'} and value[-1:] == value[0]:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()

        values[key] = value

    return values


_DOTENV_VALUES = _parse_dotenv()


def env_value(name: str, default: str | None = None) -> str | None:
    if name in os.environ:
        return os.environ[name]
    return _DOTENV_VALUES.get(name, default)


def env_int(name: str, default: int) -> int:
    raw_value = env_value(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


def frontend_build_config() -> dict[str, int]:
    return {
        "defaultTimeBudgetMs": env_int(
            "WEB_TIME_BUDGET_MS", DEFAULT_WEB_TIME_BUDGET_MS
        ),
    }


def render_frontend_build_config_js() -> str:
    payload = json.dumps(frontend_build_config(), sort_keys=True)
    return (
        "// Generated from .env by scripts/generate_web_build_config.py\n"
        f"window.EXPLICAS_BUILD_CONFIG = Object.freeze({payload});\n"
    )


def write_frontend_build_config(path: Path = BUILD_CONFIG_PATH) -> Path:
    path.write_text(render_frontend_build_config_js(), encoding="utf-8")
    return path
