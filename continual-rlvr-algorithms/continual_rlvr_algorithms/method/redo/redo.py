from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

DEFAULT_REDO_ENABLED = True
DEFAULT_REDO_TAU = 0.1
DEFAULT_USE_LECUN_INIT = False
DEFAULT_CALIBRATION_EXAMPLES = 8
MIN_REDO_ACTIVATION_DIMS = 2


def mapping_get(payload: Any, key: str, default: Any) -> Any:
    if hasattr(payload, "get"):
        return payload.get(key, default)
    return getattr(payload, key, default)


def mapping_get_bool(payload: Any, key: str, *, default: bool) -> bool:
    return bool(mapping_get(payload, key, default))


@dataclass(frozen=True)
class RedoConfig:
    enabled: bool = DEFAULT_REDO_ENABLED
    tau: float = DEFAULT_REDO_TAU
    use_lecun_init: bool = DEFAULT_USE_LECUN_INIT
    calibration_examples: int = DEFAULT_CALIBRATION_EXAMPLES

    @classmethod
    def from_mapping(cls, payload: Any) -> RedoConfig:
        return cls(
            enabled=mapping_get_bool(
                payload,
                "enabled",
                default=DEFAULT_REDO_ENABLED,
            ),
            tau=max(
                0.0,
                float(mapping_get(payload, "tau", DEFAULT_REDO_TAU)),
            ),
            use_lecun_init=mapping_get_bool(
                payload,
                "use_lecun_init",
                default=DEFAULT_USE_LECUN_INIT,
            ),
            calibration_examples=max(
                1,
                int(
                    mapping_get(
                        payload,
                        "calibration_examples",
                        DEFAULT_CALIBRATION_EXAMPLES,
                    )
                ),
            ),
        )

    def hydra_overrides(self) -> list[str]:
        return [
            f"++redo.enabled={str(self.enabled).lower()}",
            f"++redo.tau={self.tau}",
            f"++redo.use_lecun_init={str(self.use_lecun_init).lower()}",
            f"++redo.calibration_examples={self.calibration_examples}",
        ]


@dataclass(frozen=True)
class RedoTargetBlock:
    name: str
    gate_proj: Any
    up_proj: Any
    down_proj: Any


def build_redo_target_blocks(model) -> list[RedoTargetBlock]:
    targets: list[RedoTargetBlock] = []
    for name, module in model.named_modules():
        gate_proj = getattr(module, "gate_proj", None)
        up_proj = getattr(module, "up_proj", None)
        down_proj = getattr(module, "down_proj", None)
        if not all(
            isinstance(item, nn.Linear)
            for item in (gate_proj, up_proj, down_proj)
        ):
            continue
        targets.append(
            RedoTargetBlock(
                name=name or "<root>",
                gate_proj=gate_proj,
                up_proj=up_proj,
                down_proj=down_proj,
            )
        )
    if not targets:
        raise ValueError(
            "redo requires transformer MLP blocks with gate/up/down "
            "projections"
        )
    return targets


def kaiming_uniform_reinitialize_rows_(layer, mask) -> None:
    fan_in = nn.init._calculate_correct_fan(layer.weight, mode="fan_in")
    gain = nn.init.calculate_gain("relu", math.sqrt(5))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    layer.weight.data[mask, ...] = torch.empty_like(
        layer.weight.data[mask, ...]
    ).uniform_(-bound, bound)
    if layer.bias is not None:
        bias_bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        layer.bias.data[mask, ...] = torch.empty_like(
            layer.bias.data[mask, ...]
        ).uniform_(-bias_bound, bias_bound)


def lecun_normal_reinitialize_rows_(layer, mask) -> None:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
    variance = 1.0 / fan_in
    stddev = math.sqrt(variance) / 0.87962566103423978
    layer.weight.data[mask] = nn.init._no_grad_trunc_normal_(
        layer.weight.data[mask], mean=0.0, std=1.0, a=-2.0, b=2.0
    )
    layer.weight.data[mask] *= stddev
    if layer.bias is not None:
        layer.bias.data[mask] = 0.0


def reinitialize_rows_(layer, mask, *, use_lecun_init: bool) -> None:
    if use_lecun_init:
        lecun_normal_reinitialize_rows_(layer, mask)
        return
    kaiming_uniform_reinitialize_rows_(layer, mask)


def zero_optimizer_state_rows_(optimizer, param, mask) -> None:
    state = optimizer.state.get(param)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        tensor = state.get(key)
        if tensor is not None:
            tensor[bool_mask_on_device(mask, tensor.device), ...] = 0.0
    step = state.get("step")
    if step is not None:
        if hasattr(step, "zero_"):
            step.zero_()
        else:
            state["step"] = 0


def zero_optimizer_state_columns_(optimizer, param, mask) -> None:
    state = optimizer.state.get(param)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        tensor = state.get(key)
        if tensor is not None:
            tensor[:, bool_mask_on_device(mask, tensor.device), ...] = 0.0
    step = state.get("step")
    if step is not None:
        if hasattr(step, "zero_"):
            step.zero_()
        else:
            state["step"] = 0


def bool_mask_on_device(mask, device):
    if torch.is_tensor(mask):
        return mask.to(device=device, dtype=torch.bool)
    return torch.as_tensor(mask, device=device, dtype=torch.bool)


def take_first_examples(payload: Any, *, max_examples: int) -> Any:
    if isinstance(payload, dict):
        return payload
    if not hasattr(payload, "__getitem__"):
        return payload
    return payload[:max_examples]


def is_empty_multi_modal_payload(payload: Any) -> bool:
    if payload is None:
        return True
    if isinstance(payload, dict):
        return len(payload) == 0
    if isinstance(payload, (list, tuple)):
        return all(is_empty_multi_modal_payload(item) for item in payload)

    is_empty = getattr(payload, "size", None) == 0

    if not is_empty and hasattr(payload, "tolist"):
        converted = payload.tolist()
        if converted is not payload:
            return is_empty_multi_modal_payload(converted)

    if not is_empty and hasattr(payload, "item"):
        converted = payload.item()
        if converted is not payload:
            return is_empty_multi_modal_payload(converted)

    return is_empty


def slice_position_ids_for_examples(
    position_ids: Any,
    *,
    max_examples: int,
) -> Any:
    """Slice calibration examples while preserving Qwen2-VL position axes."""
    if torch.is_tensor(position_ids) and position_ids.ndim == 3:
        if position_ids.shape[0] == 4:
            return position_ids[:, :max_examples, :]
        if position_ids.shape[1] == 4:
            return (
                position_ids[:max_examples, :, :].permute(1, 0, 2).contiguous()
            )
    return position_ids[:max_examples]


def extract_model_inputs(batch: Any, *, max_examples: int) -> dict[str, Any]:
    batch_map = getattr(batch, "batch", batch)
    non_tensor_batch = getattr(batch, "non_tensor_batch", {})

    input_ids = batch_map["input_ids"][:max_examples]
    attention_mask = batch_map.get("attention_mask")
    position_ids = batch_map.get("position_ids")
    if attention_mask is not None:
        attention_mask = attention_mask[:max_examples]
    if position_ids is not None:
        position_ids = slice_position_ids_for_examples(
            position_ids,
            max_examples=max_examples,
        )

    multi_modal_inputs = non_tensor_batch.get("multi_modal_inputs")
    multi_modal_inputs = take_first_examples(
        multi_modal_inputs,
        max_examples=max_examples,
    )
    if is_empty_multi_modal_payload(multi_modal_inputs):
        multi_modal_inputs = {}
    elif isinstance(multi_modal_inputs, dict):
        # Some callers may already provide the model-ready multimodal dict.
        # Calibration uses the same VLM forward path as training.
        multi_modal_inputs = dict(multi_modal_inputs)
    else:
        from verl.utils.model import extract_multi_modal_inputs

        multi_modal_inputs = extract_multi_modal_inputs(multi_modal_inputs)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "multi_modal_inputs": multi_modal_inputs,
    }


def move_model_inputs_to_device(
    model_inputs: dict[str, Any],
    *,
    device,
) -> dict[str, Any]:
    moved = {}
    for key, value in model_inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                child_key: child_value.to(device)
                if torch.is_tensor(child_value)
                else child_value
                for child_key, child_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def capture_redo_scores(
    forward_module,
    targets: list[RedoTargetBlock],
    model_inputs: dict[str, Any],
) -> dict[str, Any]:
    activations: dict[str, torch.Tensor] = {}
    handles = []

    def build_hook(name: str):
        def hook(unused_module, args):
            if not args:
                raise ValueError(f"redo target {name} missing forward inputs")
            hidden = args[0]
            score = hidden.detach().abs().float()
            del unused_module
            if score.ndim < MIN_REDO_ACTIVATION_DIMS:
                raise ValueError(
                    "redo target "
                    f"{name} expected >={MIN_REDO_ACTIVATION_DIMS}D "
                    f"activation, got shape={tuple(score.shape)}"
                )
            while score.ndim > 1:
                score = score.mean(dim=0)
            activations[name] = score

        return hook

    for target in targets:
        handles.append(
            target.down_proj.register_forward_pre_hook(build_hook(target.name))
        )

    try:
        with torch.inference_mode():
            forward_module(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                position_ids=model_inputs["position_ids"],
                **model_inputs["multi_modal_inputs"],
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = [
        target.name for target in targets if target.name not in activations
    ]
    if missing:
        raise ValueError(
            f"redo did not observe activations for targets: {missing}"
        )
    return activations


def build_redo_masks(
    activations: dict[str, Any],
    *,
    tau: float,
) -> dict[str, Any]:
    masks: dict[str, torch.Tensor] = {}
    for name, score in activations.items():
        normalized = score / (score.mean() + 1e-9)
        if tau > 0.0:
            mask = normalized <= tau
        else:
            mask = torch.isclose(normalized, torch.zeros_like(normalized))
        masks[name] = mask
    return masks


def apply_redo_reset_from_masks(
    *,
    actor_model,
    optimizer,
    config: RedoConfig,
    masks: dict[str, Any],
) -> dict[str, Any]:
    if not config.enabled:
        return {"applied": False, "reason": "disabled"}

    targets = build_redo_target_blocks(actor_model)
    missing_masks = [
        target.name for target in targets if target.name not in masks
    ]
    if missing_masks:
        raise ValueError(
            f"redo missing dormant masks for targets: {missing_masks}"
        )

    dormant_count = 0
    total_units = 0
    reset_blocks: list[dict[str, Any]] = []
    with torch.no_grad():
        for target in targets:
            mask = masks[target.name]
            total_units += int(mask.numel())
            dormant_units = int(mask.sum().item())
            dormant_count += dormant_units
            if dormant_units == 0:
                continue
            row_mask = bool_mask_on_device(
                mask,
                target.gate_proj.weight.device,
            )
            col_mask = bool_mask_on_device(
                mask,
                target.down_proj.weight.device,
            )

            reinitialize_rows_(
                target.gate_proj,
                row_mask,
                use_lecun_init=config.use_lecun_init,
            )
            reinitialize_rows_(
                target.up_proj,
                bool_mask_on_device(mask, target.up_proj.weight.device),
                use_lecun_init=config.use_lecun_init,
            )
            target.down_proj.weight.data[:, col_mask] = 0.0

            zero_optimizer_state_rows_(
                optimizer,
                target.gate_proj.weight,
                row_mask,
            )
            zero_optimizer_state_rows_(
                optimizer,
                target.up_proj.weight,
                bool_mask_on_device(mask, target.up_proj.weight.device),
            )
            zero_optimizer_state_columns_(
                optimizer,
                target.down_proj.weight,
                col_mask,
            )
            if target.gate_proj.bias is not None:
                zero_optimizer_state_rows_(
                    optimizer,
                    target.gate_proj.bias,
                    bool_mask_on_device(mask, target.gate_proj.bias.device),
                )
            if target.up_proj.bias is not None:
                zero_optimizer_state_rows_(
                    optimizer,
                    target.up_proj.bias,
                    bool_mask_on_device(mask, target.up_proj.bias.device),
                )

            reset_blocks.append(
                {
                    "name": target.name,
                    "dormant_units": dormant_units,
                    "total_units": int(mask.numel()),
                }
            )

    dormant_fraction = 0.0
    if total_units > 0:
        dormant_fraction = dormant_count / total_units

    return {
        "applied": True,
        "tau": config.tau,
        "use_lecun_init": config.use_lecun_init,
        "calibration_examples": config.calibration_examples,
        "num_target_blocks": len(targets),
        "dormant_count": dormant_count,
        "total_units": total_units,
        "dormant_fraction": dormant_fraction,
        "sample_blocks": reset_blocks[:8],
    }


def apply_redo_reset(
    *,
    forward_module,
    actor_model,
    optimizer,
    batch: Any,
    config: RedoConfig,
) -> dict[str, Any]:
    if not config.enabled:
        return {"applied": False, "reason": "disabled"}

    targets = build_redo_target_blocks(actor_model)
    model_inputs = extract_model_inputs(
        batch,
        max_examples=config.calibration_examples,
    )
    device = next(actor_model.parameters()).device
    model_inputs = move_model_inputs_to_device(model_inputs, device=device)
    activations = capture_redo_scores(forward_module, targets, model_inputs)
    masks = build_redo_masks(activations, tau=config.tau)
    return apply_redo_reset_from_masks(
        actor_model=actor_model,
        optimizer=optimizer,
        config=config,
        masks=masks,
    )
