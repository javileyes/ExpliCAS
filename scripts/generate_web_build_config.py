#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
sys.path.insert(0, str(WEB_DIR))

from app_config import BUILD_CONFIG_PATH, frontend_build_config, write_frontend_build_config


def main() -> int:
    output_path = write_frontend_build_config(BUILD_CONFIG_PATH)
    config = frontend_build_config()
    print(
        f"Generated {output_path} "
        f"(defaultTimeBudgetMs={config['defaultTimeBudgetMs']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
