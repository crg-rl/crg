from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
)

from .osft_config import OSFTConfig
from .upstream import ensure_upstream_paths

ensure_upstream_paths()

from mini_trainer.osft_utils import (
    _build_osft_kwargs,
    _set_osft_dtypes,
    create_osft_model_class,
)
from mini_trainer.utils import get_model_class_from_config

AUTO_MODEL_CLASSES = (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
)


@dataclass(slots=True)
class OSFTPatchState:
    model_path: str
    actual_model_class_name: str
    osft_model_class_name: str
    hit_count: int = 0
    requested_loader_classes: list[str] = field(default_factory=list)
    resolved_model_paths: list[str] = field(default_factory=list)

    def record_hit(self, loader_class_name: str, model_path: str) -> None:
        self.hit_count += 1
        self.requested_loader_classes.append(loader_class_name)
        self.resolved_model_paths.append(model_path)


def _make_patched_from_pretrained(model_path: str, osft_config: OSFTConfig):
    actual_model_class = get_model_class_from_config(model_path)
    osft_model_class = create_osft_model_class(actual_model_class)
    osft_kwargs = _build_osft_kwargs(
        osft_rank_ratio=osft_config.rank_ratio,
        osft_target_patterns=osft_config.target_patterns,
    )
    patch_state = OSFTPatchState(
        model_path=str(model_path),
        actual_model_class_name=actual_model_class.__name__,
        osft_model_class_name=osft_model_class.__name__,
    )

    def _patched(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if patch_state.hit_count == 0 and osft_config.log_patch_summary:
            print(
                "[OSFT] intercepted AutoModel.from_pretrained "
                f"(loader={cls.__name__}, "
                f"actual={patch_state.actual_model_class_name}, "
                f"osft={patch_state.osft_model_class_name}, "
                f"model_path={pretrained_model_name_or_path})"
            )
        patch_state.record_hit(
            loader_class_name=cls.__name__,
            model_path=str(pretrained_model_name_or_path),
        )
        model = osft_model_class.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            initialize_osft=osft_config.initialize_osft,
            fsdp2_lazy_init=osft_config.fsdp2_lazy_init,
            **osft_kwargs,
            **kwargs,
        )
        _set_osft_dtypes(
            model,
            osft_upcast_dtype=osft_config.resolve_upcast_dtype(),
            train_dtype=osft_config.resolve_output_dtype(),
        )
        return model

    return classmethod(_patched), patch_state


@contextmanager
def patch_auto_model_classes_for_osft(
    model_path: str,
    osft_config: OSFTConfig,
) -> Iterator[OSFTPatchState | None]:
    if not osft_config.enabled:
        yield None
        return

    patched_descriptor, patch_state = _make_patched_from_pretrained(
        model_path, osft_config
    )
    originals = {
        model_class: inspect.getattr_static(model_class, "from_pretrained")
        for model_class in AUTO_MODEL_CLASSES
    }
    try:
        for model_class in AUTO_MODEL_CLASSES:
            model_class.from_pretrained = patched_descriptor
        yield patch_state
    finally:
        for model_class, original in originals.items():
            model_class.from_pretrained = original
