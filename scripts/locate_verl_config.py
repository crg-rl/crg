#!/usr/bin/env python3
"""Print the Hydra configuration directory from the installed VERL package."""

from pathlib import Path

import verl


config_root = Path(verl.__file__).resolve().parent / "trainer" / "config"
if not config_root.is_dir():
    raise SystemExit(f"VERL configuration directory not found: {config_root}")
print(config_root)
