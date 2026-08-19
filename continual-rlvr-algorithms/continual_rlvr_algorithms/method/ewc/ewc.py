from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def mapping_get(payload: Any, key: str, default: Any) -> Any:
    if hasattr(payload, "get"):
        return payload.get(key, default)
    return getattr(payload, key, default)


def mapping_get_bool(
    payload: Any,
    key: str,
    *,
    default_value: bool,
) -> bool:
    return bool(mapping_get(payload, key, default_value))


@dataclass(frozen=True)
class EWCConfig:
    enabled: bool = True
    penalty_coef: float = 1.0
    fisher_decay: float = 1.0
    max_fisher_samples: int | None = None
    eps: float = 1e-8

    @classmethod
    def from_mapping(cls, payload: Any) -> EWCConfig:
        max_fisher_samples = mapping_get(payload, "max_fisher_samples", None)
        if max_fisher_samples in (None, "", -1):
            max_fisher_samples = None
        else:
            max_fisher_samples = max(1, int(max_fisher_samples))
        return cls(
            enabled=mapping_get_bool(
                payload,
                "enabled",
                default_value=True,
            ),
            penalty_coef=float(mapping_get(payload, "penalty_coef", 1.0)),
            fisher_decay=float(mapping_get(payload, "fisher_decay", 1.0)),
            max_fisher_samples=max_fisher_samples,
            eps=float(mapping_get(payload, "eps", 1e-8)),
        )

    def hydra_overrides(self) -> list[str]:
        max_fisher_samples = (
            -1 if self.max_fisher_samples is None else self.max_fisher_samples
        )
        return [
            f"++ewc.enabled={str(self.enabled).lower()}",
            f"++ewc.penalty_coef={self.penalty_coef}",
            f"++ewc.fisher_decay={self.fisher_decay}",
            f"++ewc.max_fisher_samples={max_fisher_samples}",
            f"++ewc.eps={self.eps}",
        ]


class OnlineEWCState:
    def __init__(self, config: EWCConfig):
        self.config = config
        self.reference_means: dict[str, torch.Tensor] = {}
        self.reference_fishers: dict[str, torch.Tensor] = {}
        self.pending_fisher_sums: dict[str, torch.Tensor] = {}
        self.pending_task_name: str | None = None
        self.pending_samples = 0

    def has_reference(self) -> bool:
        return bool(self.reference_means)

    def iter_trainable_named_params(
        self,
        module,
    ) -> list[tuple[str, torch.Tensor]]:
        return [
            (name, param)
            for name, param in module.named_parameters()
            if param.requires_grad
        ]

    def estimate_penalty(self, module) -> torch.Tensor:
        named_params = self.iter_trainable_named_params(module)
        if not named_params or not self.reference_means:
            device = named_params[0][1].device if named_params else "cpu"
            return torch.zeros((), device=device, dtype=torch.float32)

        penalties = []
        for name, param in named_params:
            fisher = self.reference_fishers.get(name)
            mean = self.reference_means.get(name)
            if fisher is None or mean is None:
                continue
            penalties.append(
                torch.sum(
                    fisher
                    * (param.float() - mean.to(param.device).float()) ** 2
                )
            )
        if not penalties:
            return torch.zeros(
                (),
                device=named_params[0][1].device,
                dtype=torch.float32,
            )
        return torch.stack(penalties).sum()

    def accumulate_fisher_from_grads(
        self,
        *,
        named_params: list[tuple[str, torch.Tensor]],
        grads: tuple[torch.Tensor | None, ...],
        task_name: str | None,
    ) -> None:
        if not self.config.enabled:
            return
        if (
            self.config.max_fisher_samples is not None
            and self.pending_samples >= self.config.max_fisher_samples
        ):
            return

        if task_name is not None:
            self.pending_task_name = task_name

        for (name, _param), grad in zip(named_params, grads, strict=False):
            if grad is None:
                continue
            fisher_value = grad.detach().float().pow(2)
            if name not in self.pending_fisher_sums:
                self.pending_fisher_sums[name] = fisher_value.clone()
            else:
                self.pending_fisher_sums[name].add_(fisher_value)
        self.pending_samples += 1

    def consolidate_current_task(
        self,
        module,
        *,
        task_name: str | None,
    ) -> dict[str, object]:
        named_params = self.iter_trainable_named_params(module)
        if not named_params:
            return {"consolidated": False, "reason": "no trainable params"}
        if not self.pending_fisher_sums or self.pending_samples <= 0:
            return {
                "consolidated": False,
                "reason": "no fisher samples",
                "task_name": task_name,
            }

        fisher_scale = 1.0 / max(self.pending_samples, 1)
        fisher_decay = self.config.fisher_decay
        eps = self.config.eps

        for name, param in named_params:
            fisher_sum = self.pending_fisher_sums.get(name)
            if fisher_sum is None:
                continue
            new_fisher = fisher_sum * fisher_scale
            new_mean = param.detach().float().clone()
            old_fisher = self.reference_fishers.get(name)
            old_mean = self.reference_means.get(name)
            if old_fisher is None or old_mean is None:
                combined_fisher = new_fisher
                combined_mean = new_mean
            else:
                combined_fisher = fisher_decay * old_fisher + new_fisher
                combined_mean = (
                    fisher_decay * old_fisher * old_mean
                    + new_fisher * new_mean
                ) / (combined_fisher + eps)
            self.reference_fishers[name] = combined_fisher.detach().clone()
            self.reference_means[name] = combined_mean.detach().clone()

        summary = {
            "consolidated": True,
            "task_name": task_name,
            "num_params": len(self.reference_fishers),
            "fisher_samples": self.pending_samples,
        }
        self.pending_fisher_sums = {}
        self.pending_samples = 0
        self.pending_task_name = None
        return summary

    def state_dict(self) -> dict[str, Any]:
        return {
            "reference_means": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.reference_means.items()
            },
            "reference_fishers": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.reference_fishers.items()
            },
            "pending_fisher_sums": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.pending_fisher_sums.items()
            },
            "pending_task_name": self.pending_task_name,
            "pending_samples": int(self.pending_samples),
        }

    def load_state_dict(self, payload: dict[str, Any], module=None) -> None:
        device_map: dict[str, torch.device] = {}
        if module is not None:
            device_map = {
                name: param.device
                for name, param in self.iter_trainable_named_params(module)
            }

        def restore_named_tensors(
            key: str,
        ) -> dict[str, torch.Tensor]:
            restored: dict[str, torch.Tensor] = {}
            for name, tensor in payload.get(key, {}).items():
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"{key}[{name!r}] must be a torch.Tensor, got "
                        f"{type(tensor)!r}"
                    )
                target_device = device_map.get(name, tensor.device)
                restored[name] = tensor.detach().to(target_device).clone()
            return restored

        self.reference_means = restore_named_tensors("reference_means")
        self.reference_fishers = restore_named_tensors("reference_fishers")
        self.pending_fisher_sums = restore_named_tensors("pending_fisher_sums")
        self.pending_task_name = payload.get("pending_task_name")
        self.pending_samples = int(payload.get("pending_samples", 0))
