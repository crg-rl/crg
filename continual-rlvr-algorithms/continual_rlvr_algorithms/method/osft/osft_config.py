from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from omegaconf import ListConfig


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _normalize_target_patterns(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(item) for item in value]
    raise TypeError(f"Unsupported target_patterns type: {type(value)!r}")


def _resolve_dtype(value: str | None) -> torch.dtype | None:
    if value is None:
        return None
    normalized = value.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype string: {value}")
    return mapping[normalized]


@dataclass(slots=True)
class OSFTConfig:
    enabled: bool = False
    rank_ratio: float = 0.5
    target_patterns: list[str] = field(default_factory=list)
    initialize_osft: bool = True
    fsdp2_lazy_init: bool = False
    upcast_dtype: str = "float32"
    output_dtype: str = "bfloat16"
    require_patch_hit: bool = True
    verify_actor_only: bool = True
    log_patch_summary: bool = True
    reinit_on_task_switch: bool = True
    reset_optimizer_on_reinit: bool = True
    sync_rollout_on_reinit: bool = True
    skip_svd_on_resume: bool = True
    resume_mode: str = "disable"
    resume_from_path: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.rank_ratio < 1.0:
            raise ValueError(
                f"OSFT rank_ratio must be in (0, 1), got {self.rank_ratio}"
            )
        if (
            self.enabled
            and self.reinit_on_task_switch
            and not self.initialize_osft
        ):
            raise ValueError(
                "OSFT task-switch re-protect requires initialize_osft=true"
            )

    def resolve_upcast_dtype(self) -> torch.dtype | None:
        return _resolve_dtype(self.upcast_dtype)

    def resolve_output_dtype(self) -> torch.dtype | None:
        return _resolve_dtype(self.output_dtype)


def load_osft_config(config: Any) -> OSFTConfig:
    raw_osft = _cfg_get(config, "osft", None)
    if raw_osft is None:
        model_config = _cfg_get(config, "model", None)
        raw_osft = _cfg_get(model_config, "osft", None)
    if raw_osft is None:
        return OSFTConfig()
    return OSFTConfig(
        enabled=bool(_cfg_get(raw_osft, "enabled", False)),
        rank_ratio=float(_cfg_get(raw_osft, "rank_ratio", 0.5)),
        target_patterns=_normalize_target_patterns(
            _cfg_get(raw_osft, "target_patterns", [])
        ),
        initialize_osft=bool(_cfg_get(raw_osft, "initialize_osft", True)),
        fsdp2_lazy_init=bool(_cfg_get(raw_osft, "fsdp2_lazy_init", False)),
        upcast_dtype=str(_cfg_get(raw_osft, "upcast_dtype", "float32")),
        output_dtype=str(_cfg_get(raw_osft, "output_dtype", "bfloat16")),
        require_patch_hit=bool(_cfg_get(raw_osft, "require_patch_hit", True)),
        verify_actor_only=bool(_cfg_get(raw_osft, "verify_actor_only", True)),
        log_patch_summary=bool(_cfg_get(raw_osft, "log_patch_summary", True)),
        reinit_on_task_switch=bool(
            _cfg_get(raw_osft, "reinit_on_task_switch", True)
        ),
        reset_optimizer_on_reinit=bool(
            _cfg_get(raw_osft, "reset_optimizer_on_reinit", True)
        ),
        sync_rollout_on_reinit=bool(
            _cfg_get(raw_osft, "sync_rollout_on_reinit", True)
        ),
        skip_svd_on_resume=bool(
            _cfg_get(raw_osft, "skip_svd_on_resume", True)
        ),
        resume_mode=str(_cfg_get(raw_osft, "resume_mode", "disable")),
        resume_from_path=str(_cfg_get(raw_osft, "resume_from_path", "")),
    )
