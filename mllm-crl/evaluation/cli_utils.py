"""Shared helpers for evaluation command entrypoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from omegaconf import ListConfig

SOURCE_REPO_ROOT = Path(__file__).resolve().parents[1]


def path_is_under(path: Path, parent: Path) -> bool:
    """Return true when path resolves inside parent."""

    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_output_outside_repo(
    path: Path | None,
    label: str,
    *,
    allow_output_in_repo: bool,
) -> None:
    """Prevent evaluation artifacts from being written into the source repo."""

    if path is None or allow_output_in_repo:
        return
    if path_is_under(path, SOURCE_REPO_ROOT):
        raise ValueError(
            f'{label} must be outside source repo {SOURCE_REPO_ROOT}; '
            'use the workspace runs/results directory instead, or pass '
            '--allow-output-in-repo only for a throwaway local debug run'
        )


def optional_path(value: Any) -> Path | None:
    """Convert a config path value to ``Path`` or ``None``."""

    if value is None or value == '':
        return None
    return Path(str(value))


def required_path(value: Any, label: str) -> Path:
    """Convert a required config path value to ``Path``."""

    path = optional_path(value)
    if path is None:
        raise ValueError(f'{label} is required')
    return path


def optional_str(value: Any) -> str | None:
    """Convert a config scalar to ``str`` or ``None``."""

    if value is None or value == '':
        return None
    return str(value)


def required_str(value: Any, label: str) -> str:
    """Convert a required config scalar to ``str``."""

    text = optional_str(value)
    if text is None:
        raise ValueError(f'{label} is required')
    return text


def string_list(value: Any) -> list[str]:
    """Convert a config string/list value to a list of strings."""

    if value is None or value == '':
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, ListConfig | list | tuple):
        return [str(item) for item in value]
    raise TypeError(f'expected string or list, got {type(value).__name__}')


def optional_string_list(value: Any) -> list[str] | None:
    """Convert a config string/list value to a list, preserving ``None``."""

    values = string_list(value)
    return values or None


def path_list(value: Any) -> list[Path]:
    """Convert a config path/list value to ``Path`` objects."""

    return [Path(item) for item in string_list(value)]


def write_json_document(document: dict[str, Any], output: Path | None) -> None:
    """Write a JSON document either to a file or stdout."""

    text = json.dumps(document, indent=2, sort_keys=True) + '\n'
    if output is None:
        sys.stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
