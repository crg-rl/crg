"""Notify algorithms after mllm-crl dataset task switches.

The dataset owns ``current_task`` and ``on_batch_end``. This helper wraps the
callback and runs method-specific task-switch side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def current_task_name(train_dataset: Any) -> str | None:
    current = getattr(train_dataset, "current_task", None)
    if current is None:
        return None
    return str(current)


def resolve_initial_task_name(train_dataset: Any, fallback: str) -> str:
    current = current_task_name(train_dataset)
    if current is not None:
        return current
    return fallback


def install_task_switch_hook(
    train_dataset: Any,
    *,
    on_task_switch: Callable[..., None],
    hook_attr: str = "_algorithm_task_switch_hook_installed",
    pass_batch: bool = False,
) -> bool:
    if getattr(train_dataset, hook_attr, False):
        return False

    original_on_batch_end = getattr(train_dataset, "on_batch_end", None)
    if original_on_batch_end is None:
        return False

    if current_task_name(train_dataset) is None:
        return False

    def patched_on_batch_end(batch: Any) -> None:
        previous_task = current_task_name(train_dataset)
        original_on_batch_end(batch)
        current_task = current_task_name(train_dataset)
        if (
            previous_task is None
            or current_task is None
            or current_task == previous_task
        ):
            return
        if pass_batch:
            on_task_switch(previous_task, current_task, batch)
            return
        on_task_switch(previous_task, current_task)

    train_dataset.on_batch_end = patched_on_batch_end
    setattr(train_dataset, hook_attr, True)
    return True
