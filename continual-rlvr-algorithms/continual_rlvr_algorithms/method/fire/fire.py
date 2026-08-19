from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

LINEAR_WEIGHT_DIMENSIONS = 2
FIRE_RESET_SCOPE_ALL_LINEAR = "all_linear"
FIRE_RESET_SCOPE_ATTENTION_QK = "attention_qk"
FIRE_RESET_SCOPES = frozenset(
    {FIRE_RESET_SCOPE_ALL_LINEAR, FIRE_RESET_SCOPE_ATTENTION_QK}
)
ATTENTION_QK_MODULE_SUFFIXES = ("q_proj", "k_proj", "to_q", "to_k")
FUSED_QKV_MODULE_SUFFIXES = ("c_attn",)
FUSED_QKV_CHUNK_LABELS = ("q", "k", "v")


def mapping_get(payload: Any, key: str, default: Any) -> Any:
    if hasattr(payload, "get"):
        return payload.get(key, default)
    return getattr(payload, key, default)


def mapping_get_bool(payload: Any, key: str, *, default: bool) -> bool:
    value = mapping_get(payload, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def normalize_reset_scope(value: Any) -> str:
    scope = str(value or FIRE_RESET_SCOPE_ALL_LINEAR).strip().lower()
    if scope not in FIRE_RESET_SCOPES:
        allowed = ", ".join(sorted(FIRE_RESET_SCOPES))
        raise ValueError(
            f"unsupported FIRE reset_scope={scope!r}; allowed={allowed}"
        )
    return scope


@dataclass(frozen=True)
class FIREConfig:
    enabled: bool = True
    ns_steps: int = 5
    include_lm_head: bool = False
    reset_optimizer: bool = True
    sync_rollout_after_reset: bool = True
    reset_scope: str = FIRE_RESET_SCOPE_ATTENTION_QK

    @classmethod
    def from_mapping(cls, payload: Any) -> FIREConfig:
        return cls(
            enabled=mapping_get_bool(payload, "enabled", default=True),
            ns_steps=max(1, int(mapping_get(payload, "ns_steps", 5))),
            include_lm_head=mapping_get_bool(
                payload,
                "include_lm_head",
                default=False,
            ),
            reset_optimizer=mapping_get_bool(
                payload,
                "reset_optimizer",
                default=True,
            ),
            sync_rollout_after_reset=mapping_get_bool(
                payload,
                "sync_rollout_after_reset",
                default=True,
            ),
            reset_scope=normalize_reset_scope(
                mapping_get(
                    payload,
                    "reset_scope",
                    FIRE_RESET_SCOPE_ATTENTION_QK,
                )
            ),
        )

    def hydra_overrides(self) -> list[str]:
        return [
            f"++fire.enabled={str(self.enabled).lower()}",
            f"++fire.ns_steps={self.ns_steps}",
            f"++fire.include_lm_head={str(self.include_lm_head).lower()}",
            f"++fire.reset_optimizer={str(self.reset_optimizer).lower()}",
            "++fire.sync_rollout_after_reset="
            f"{str(self.sync_rollout_after_reset).lower()}",
            f"++fire.reset_scope={self.reset_scope}",
        ]


def newton_schulz_orthogonalize(matrix, *, num_iters: int):
    if matrix.ndim != LINEAR_WEIGHT_DIMENSIONS:
        raise ValueError(
            f"expected a 2D matrix, got shape={tuple(matrix.shape)}"
        )

    needs_transpose = matrix.shape[1] > matrix.shape[0]
    working = matrix.T if needs_transpose else matrix
    working = working / working.norm()

    for _ in range(num_iters):
        gram = working.T @ working
        working = 1.5 * working - 0.5 * working @ gram

    return working.T if needs_transpose else working


def should_reset_module(
    module_name: str,
    module,
    *,
    include_lm_head: bool,
    reset_scope: str,
) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if not include_lm_head and module_name.endswith("lm_head"):
        return False
    if reset_scope == FIRE_RESET_SCOPE_ALL_LINEAR:
        return True
    if reset_scope == FIRE_RESET_SCOPE_ATTENTION_QK:
        module_leaf = module_name.split(".")[-1]
        return (
            module_leaf in ATTENTION_QK_MODULE_SUFFIXES
            or module_leaf in FUSED_QKV_MODULE_SUFFIXES
        )
    raise ValueError(f"unsupported FIRE reset_scope={reset_scope!r}")


def reset_linear_weight_(weight, *, num_iters: int) -> None:
    if weight.ndim != LINEAR_WEIGHT_DIMENSIONS:
        raise ValueError(
            f"expected 2D linear weight, got shape={tuple(weight.shape)}"
        )

    orthogonal = newton_schulz_orthogonalize(
        weight.detach().float().clone(),
        num_iters=num_iters,
    )
    scale = math.sqrt(weight.shape[0] / weight.shape[1])
    weight.copy_(
        orthogonal.to(dtype=weight.dtype, device=weight.device) * scale
    )


def reset_gpt_fused_qkv_weight_(
    weight,
    *,
    num_iters: int,
    reset_scope: str,
) -> str | None:
    """Apply the official FIRE GPT ``c_attn`` split-qkv path.

    The official language implementation orthogonalizes a fused GPT attention
    projection as three independent Q/K/V matrices instead of treating the
    concatenated ``3d x d`` weight as one matrix.  For the benchmark-native
    ``attention_qk`` scope, only the Q/K chunks are reset and the V chunk is
    left intact.
    """

    if weight.ndim != LINEAR_WEIGHT_DIMENSIONS:
        raise ValueError(
            f"expected 2D linear weight, got shape={tuple(weight.shape)}"
        )
    if weight.shape[0] != 3 * weight.shape[1]:
        return None

    chunk_size = weight.shape[1]
    reset_chunks: list[str] = []
    for index, label in enumerate(FUSED_QKV_CHUNK_LABELS):
        if reset_scope == FIRE_RESET_SCOPE_ATTENTION_QK and label == "v":
            continue
        start = index * chunk_size
        reset_linear_weight_(
            weight[start : start + chunk_size],
            num_iters=num_iters,
        )
        reset_chunks.append(label)

    return "gpt_fused_qkv_" + "".join(reset_chunks)


def reset_fire_module_weight_(
    module_name: str,
    module,
    *,
    num_iters: int,
    reset_scope: str,
) -> str:
    if module_name.split(".")[-1] == "c_attn":
        reset_kind = reset_gpt_fused_qkv_weight_(
            module.weight,
            num_iters=num_iters,
            reset_scope=reset_scope,
        )
        if reset_kind is not None:
            return reset_kind
    reset_linear_weight_(module.weight, num_iters=num_iters)
    return "linear"


def clear_optimizer_state(optimizer) -> dict[str, object]:
    if optimizer is None:
        return {"cleared": False, "reason": "missing optimizer"}

    state = getattr(optimizer, "state", None)
    if state is None:
        return {"cleared": False, "reason": "missing state"}

    num_state_entries = len(state)
    state.clear()
    zero_grad = getattr(optimizer, "zero_grad", None)
    if callable(zero_grad):
        zero_grad(set_to_none=True)
    return {"cleared": True, "num_state_entries": num_state_entries}


def apply_fire_reset(module, config: FIREConfig) -> dict[str, object]:
    if not config.enabled:
        return {"applied": False, "reason": "disabled"}

    reset_modules: list[str] = []
    reset_kinds: dict[str, int] = {}
    with torch.no_grad():
        for module_name, child in module.named_modules():
            if not should_reset_module(
                module_name,
                child,
                include_lm_head=config.include_lm_head,
                reset_scope=config.reset_scope,
            ):
                continue
            reset_kind = reset_fire_module_weight_(
                module_name,
                child,
                num_iters=config.ns_steps,
                reset_scope=config.reset_scope,
            )
            reset_kinds[reset_kind] = reset_kinds.get(reset_kind, 0) + 1
            reset_modules.append(module_name or "<root>")

    return {
        "applied": True,
        "ns_steps": config.ns_steps,
        "include_lm_head": config.include_lm_head,
        "reset_optimizer": config.reset_optimizer,
        "sync_rollout_after_reset": config.sync_rollout_after_reset,
        "reset_scope": config.reset_scope,
        "num_reset_modules": len(reset_modules),
        "reset_kinds": reset_kinds,
        "sample_modules": reset_modules[:8],
    }
