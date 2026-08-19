from __future__ import annotations

import importlib.util
import os
import sys
import types
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

ReprotectTargetContext = Callable[[str], AbstractContextManager[None]]


@dataclass(slots=True)
class OSFTReprotectSummary:
    applied: bool
    reason: str | None
    previous_task: str
    current_task: str
    reprotected_modules: int
    optimizer_state_cleared: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "previous_task": self.previous_task,
            "current_task": self.current_task,
            "reprotected_modules": self.reprotected_modules,
            "optimizer_state_cleared": self.optimizer_state_cleared,
        }


def require_osft_reprotect_api(actor_model: nn.Module) -> None:
    required_callables = ("project_gradients",)
    missing = [
        name
        for name in required_callables
        if not callable(getattr(actor_model, name, None))
    ]
    if not isinstance(getattr(actor_model, "osft_config", None), dict):
        missing.append("osft_config")
    if not isinstance(
        getattr(actor_model, "osft_paramspec_registry", None), dict
    ):
        missing.append("osft_paramspec_registry")
    if missing:
        raise TypeError(
            "OSFT actor model is missing required mini_trainer OSFT APIs: "
            + ", ".join(missing)
        )


def mini_trainer_hint() -> str:
    return (
        "Set MINI_TRAINER_SRC_ROOT to the osft-mini_trainer/src checkout "
        "or place that checkout next to continual-rlvr-algorithms."
    )


def load_osft_utils_directly():
    raw_root = os.environ.get("MINI_TRAINER_SRC_ROOT")
    if raw_root is None:
        repo_root = Path(__file__).resolve().parents[3]
        workspace_root = repo_root.parent
        candidates = [
            workspace_root / "osft-mini_trainer" / "src",
            repo_root.parents[2]
            / "continual-reasoning-gym-baseline-sources"
            / "code"
            / "osft-mini_trainer"
            / "src",
        ]
    else:
        candidates = [Path(raw_root)]

    src_root = next(
        (path.resolve() for path in candidates if path.exists()), None
    )
    if src_root is None:
        return None
    module_path = src_root / "mini_trainer" / "osft_utils.py"
    if not module_path.exists():
        return None

    existing = sys.modules.get("mini_trainer.osft_utils")
    if existing is not None and getattr(existing, "__file__", None) == str(
        module_path
    ):
        return existing

    package = sys.modules.get("mini_trainer")
    if package is None:
        package = types.ModuleType("mini_trainer")
        package.__path__ = [str(src_root / "mini_trainer")]
        package.__file__ = str(src_root / "mini_trainer" / "__init__.py")
        sys.modules["mini_trainer"] = package

    spec = importlib.util.spec_from_file_location(
        "mini_trainer.osft_utils",
        module_path,
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["mini_trainer.osft_utils"] = module
    spec.loader.exec_module(module)
    return module


def load_create_svd_dict():
    try:
        from mini_trainer.osft_utils import create_svd_dict

        return create_svd_dict
    except ModuleNotFoundError as exc:
        module = load_osft_utils_directly()
        if module is not None:
            return module.create_svd_dict
        raise ModuleNotFoundError(
            "OSFT task-switch re-protect requires mini_trainer.osft_utils. "
            + mini_trainer_hint()
        ) from exc
    except ImportError as exc:
        module = load_osft_utils_directly()
        if module is not None:
            return module.create_svd_dict
        raise ImportError(
            "Failed to import mini_trainer.osft_utils for OSFT task-switch "
            f"re-protect: {exc}. {mini_trainer_hint()}"
        ) from exc


def osft_target_count(actor_model: nn.Module) -> int:
    registry = getattr(actor_model, "osft_paramspec_registry", None)
    if registry is None:
        return 0
    return len(registry)


def empty_accelerator_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch_npu = getattr(torch, "npu", None)
    if torch_npu is not None and hasattr(torch_npu, "empty_cache"):
        torch_npu.empty_cache()


def get_osft_target_module(
    actor_model: nn.Module, logical_key: str
) -> nn.Module:
    get_by_logical = getattr(actor_model, "_get_module_by_logical_key", None)
    if callable(get_by_logical):
        module, _ = get_by_logical(logical_key)
        if module is not None:
            return module

    get_by_name = getattr(actor_model, "_get_module_by_name", None)
    if callable(get_by_name):
        module, _ = get_by_name(logical_key)
        if module is not None:
            return module

    parts = logical_key.split(".")[:-1]
    module = actor_model
    for part in parts:
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def osft_module_factor_tensors(
    *,
    module: nn.Module,
    logical_key: str,
) -> dict[str, torch.Tensor]:
    required = {
        "U_high": getattr(module, "osft_U_high", None),
        "S_high": getattr(module, "osft_S_high", None),
        "V_high": getattr(module, "osft_V_high", None),
        "U_low": getattr(getattr(module, "osft_params", None), "U_low", None),
        "S_low": getattr(getattr(module, "osft_params", None), "S_low", None),
        "V_low": getattr(getattr(module, "osft_params", None), "V_low", None),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            f"OSFT target module is missing factors for {logical_key}: "
            + ", ".join(missing)
        )
    return required


def reconstruct_dense_osft_weight(
    *,
    actor_model: nn.Module,
    logical_key: str,
) -> torch.Tensor:
    module = get_osft_target_module(actor_model, logical_key)
    factors = osft_module_factor_tensors(
        module=module, logical_key=logical_key
    )
    upcast_dtype = getattr(actor_model, "upcast_dtype", torch.float32)

    U_high = factors["U_high"].to(dtype=upcast_dtype)
    S_high = factors["S_high"].to(dtype=upcast_dtype)
    V_high = factors["V_high"].to(dtype=upcast_dtype)
    U_low = factors["U_low"].to(dtype=upcast_dtype)
    S_low = factors["S_low"].to(dtype=upcast_dtype)
    V_low = factors["V_low"].to(dtype=upcast_dtype)

    high_part = None
    if U_high.numel() > 0 and S_high.numel() > 0:
        high_part = torch.mm(U_high * S_high.unsqueeze(0), V_high)

    low_part = None
    if U_low.numel() > 0 and S_low.numel() > 0:
        low_part = torch.mm(U_low * S_low.unsqueeze(0), V_low)

    if high_part is None and low_part is None:
        raise RuntimeError(
            f"OSFT re-protect cannot reconstruct an empty factorization: {logical_key}"
        )
    if high_part is None:
        dense_weight = low_part
    elif low_part is None:
        dense_weight = high_part
    else:
        dense_weight = high_part + low_part

    output_dtype = getattr(actor_model, "output_dtype", None)
    if output_dtype is None:
        output_dtype = factors["U_high"].dtype
    return dense_weight.to(dtype=output_dtype)


def copy_tensor_in_place(
    target: torch.Tensor, source: torch.Tensor, name: str
) -> None:
    if target.shape != source.shape:
        raise RuntimeError(
            f"OSFT re-protect shape mismatch for {name}: "
            f"target={tuple(target.shape)}, source={tuple(source.shape)}"
        )
    target.data.copy_(source.to(device=target.device, dtype=target.dtype))


def replace_osft_factors(
    *,
    actor_model: nn.Module,
    logical_key: str,
    dense_weight: torch.Tensor,
) -> None:
    module = get_osft_target_module(actor_model, logical_key)
    if not hasattr(module, "osft_params"):
        raise RuntimeError(
            f"OSFT target module has no osft_params: {logical_key}"
        )

    create_svd_dict = load_create_svd_dict()
    top_k = actor_model.osft_config[logical_key]
    svd_dict = create_svd_dict(
        dense_weight,
        top_k=top_k,
        decompose_existing=True,
        upcast_dtype=getattr(actor_model, "upcast_dtype", torch.float32),
        output_dtype=dense_weight.dtype,
        use_meta=False,
    )

    copy_tensor_in_place(
        module.osft_U_high,
        svd_dict["U_high"],
        f"{logical_key}.U_high",
    )
    copy_tensor_in_place(
        module.osft_S_high,
        svd_dict["S_high"],
        f"{logical_key}.S_high",
    )
    copy_tensor_in_place(
        module.osft_V_high,
        svd_dict["V_high"],
        f"{logical_key}.V_high",
    )
    copy_tensor_in_place(
        module.osft_params.U_low,
        svd_dict["U_low"],
        f"{logical_key}.U_low",
    )
    copy_tensor_in_place(
        module.osft_params.S_low,
        svd_dict["S_low"],
        f"{logical_key}.S_low",
    )
    copy_tensor_in_place(
        module.osft_params.V_low,
        svd_dict["V_low"],
        f"{logical_key}.V_low",
    )
    module.osft_params.rank_high = int(svd_dict["rank_high"])


def reinitialize_osft_from_current_dense_weights(
    actor_model: nn.Module,
    target_context: ReprotectTargetContext | None = None,
) -> int:
    target_keys = list(actor_model.osft_paramspec_registry)
    if not target_keys:
        raise RuntimeError(
            "OSFT task-switch re-protect found zero OSFT targets"
        )

    for logical_key in target_keys:
        context = (
            target_context(logical_key) if target_context else nullcontext()
        )
        with context:
            dense_weight = reconstruct_dense_osft_weight(
                actor_model=actor_model,
                logical_key=logical_key,
            )
            replace_osft_factors(
                actor_model=actor_model,
                logical_key=logical_key,
                dense_weight=dense_weight,
            )
            del dense_weight
        empty_accelerator_cache()
    return len(target_keys)


def clear_optimizer_state(optimizer: Any | None) -> bool:
    if optimizer is None:
        return False
    state = getattr(optimizer, "state", None)
    if state is None:
        return False
    state.clear()
    return True


def apply_osft_task_switch_reprotect(
    *,
    actor_model: nn.Module,
    optimizer: Any | None,
    previous_task: str,
    current_task: str,
    reset_optimizer: bool,
    target_context: ReprotectTargetContext | None = None,
) -> dict[str, object]:
    require_osft_reprotect_api(actor_model)
    reprotected_modules = reinitialize_osft_from_current_dense_weights(
        actor_model,
        target_context=target_context,
    )
    if reprotected_modules <= 0:
        raise RuntimeError(
            "OSFT task-switch re-protect found zero OSFT targets"
        )

    summary = OSFTReprotectSummary(
        applied=True,
        reason=None,
        previous_task=previous_task,
        current_task=current_task,
        reprotected_modules=reprotected_modules,
        optimizer_state_cleared=(
            clear_optimizer_state(optimizer) if reset_optimizer else False
        ),
    )
    return summary.as_dict()
