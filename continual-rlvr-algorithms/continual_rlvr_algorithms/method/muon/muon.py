from __future__ import annotations

import math

import torch

try:
    from torch.distributed.tensor import DTensor, Replicate
except Exception:  # pragma: no cover - optional distributed tensor API
    DTensor = None
    Replicate = None

EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5
MUON_PARAM_NDIM = 2
MIN_MATRIX_NDIM = 2
CONV_KERNEL_NDIM = 4
SUPPORTED_ADJUST_LR_FNS = {None, "original", "match_rms_adamw"}
SUPPORTED_NS_DTYPES = {"bfloat16", "float32"}


ParamEntry = tuple[str | None, torch.nn.Parameter]


def recover_originating_module(params) -> torch.nn.Module | None:
    frame = getattr(params, "gi_frame", None)
    code = getattr(params, "gi_code", None)
    if frame is None or code is None or code.co_name != "parameters":
        return None

    module = frame.f_locals.get("self")
    return module if isinstance(module, torch.nn.Module) else None


def recover_named_entries(params) -> list[ParamEntry] | None:
    module = recover_originating_module(params)
    if module is None:
        return None

    frame = getattr(params, "gi_frame", None)
    recurse = bool(frame.f_locals.get("recurse", True))
    return list(module.named_parameters(recurse=recurse))


def normalize_entries(params) -> list[ParamEntry]:
    recovered_entries = recover_named_entries(params)
    if recovered_entries is not None:
        return recovered_entries

    normalized: list[ParamEntry] = []
    for entry in params:
        if isinstance(entry, tuple):
            name, param = entry
        else:
            name, param = None, entry
        if not isinstance(param, torch.nn.Parameter):
            raise TypeError(
                f"Expected torch.nn.Parameter, got {type(param)!r}"
            )
        normalized.append((name, param))
    return normalized


def summarize_param_entries(
    param_entries: list[ParamEntry],
    *,
    recovered_named_entries: bool,
    sample_limit: int = 12,
) -> str:
    ndim_counts: dict[int, int] = {}
    trainable_count = 0
    sample_parts: list[str] = []

    for index, (name, param) in enumerate(param_entries):
        ndim_counts[param.ndim] = ndim_counts.get(param.ndim, 0) + 1
        if param.requires_grad:
            trainable_count += 1
        if index >= sample_limit:
            continue
        sample_parts.append(
            f"{name or '<unnamed>'}:shape={tuple(param.shape)}"
            f":ndim={param.ndim}:requires_grad={param.requires_grad}"
        )

    ndim_summary = ",".join(
        f"{ndim}:{count}" for ndim, count in sorted(ndim_counts.items())
    )
    samples = "; ".join(sample_parts) if sample_parts else "<none>"
    return (
        f"recovered_named_entries={recovered_named_entries}; "
        f"total={len(param_entries)}; "
        f"trainable={trainable_count}; "
        f"ndim_counts={ndim_summary or '<empty>'}; "
        f"samples={samples}"
    )


def should_use_muon_style_update(
    name: str | None,
    param: torch.nn.Parameter,
    *,
    max_muon_aspect_ratio: float,
    max_muon_dim: int,
) -> bool:
    if not param.requires_grad:
        return False
    if param.ndim != MUON_PARAM_NDIM:
        return False

    rows, cols = param.shape
    smaller_dim = min(rows, cols)
    larger_dim = max(rows, cols)
    if smaller_dim == 0:
        return False
    if larger_dim > max_muon_dim:
        return False
    if larger_dim / smaller_dim > max_muon_aspect_ratio:
        return False

    lowered_name = (name or "").lower()
    return "embed" not in lowered_name and "lm_head" not in lowered_name


def split_params(
    param_entries: list[ParamEntry],
    *,
    max_muon_aspect_ratio: float,
    max_muon_dim: int,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for name, param in param_entries:
        if should_use_muon_style_update(
            name,
            param,
            max_muon_aspect_ratio=max_muon_aspect_ratio,
            max_muon_dim=max_muon_dim,
        ):
            muon_params.append(param)
        elif param.requires_grad:
            adamw_params.append(param)
    return muon_params, adamw_params


def build_param_groups(
    params,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = DEFAULT_NS_STEPS,
    adamw_lr: float | None = None,
    adamw_lr_ratio: float = 1.0,
    adamw_weight_decay: float | None = None,
    adamw_weight_decay_ratio: float = 1.0,
    adamw_eps: float = 1e-8,
    adjust_lr_fn: str | None = None,
    max_muon_aspect_ratio: float = 8.0,
    max_muon_dim: int = 65536,
    ns_dtype: str = "bfloat16",
) -> list[dict[str, object]]:
    recovered_entries = recover_named_entries(params)
    recovered_named_entries = recovered_entries is not None
    param_entries = (
        recovered_entries
        if recovered_entries is not None
        else normalize_entries(params)
    )
    if not param_entries:
        raise ValueError("Optimizer got an empty parameter list")
    if adamw_lr is not None and adamw_lr_ratio != 1.0:
        raise ValueError("Specify only one of adamw_lr or adamw_lr_ratio")
    if adamw_weight_decay is not None and adamw_weight_decay_ratio != 1.0:
        raise ValueError(
            "Specify only one of adamw_weight_decay or "
            "adamw_weight_decay_ratio"
        )
    if adjust_lr_fn not in SUPPORTED_ADJUST_LR_FNS:
        raise ValueError(
            f"Unsupported Muon adjust_lr_fn={adjust_lr_fn!r}; expected one "
            "of None, 'original', or 'match_rms_adamw'"
        )

    if ns_dtype not in SUPPORTED_NS_DTYPES:
        raise ValueError(
            f"Unsupported Muon ns_dtype={ns_dtype!r}; expected one of "
            f"{sorted(SUPPORTED_NS_DTYPES)}"
        )

    if adamw_lr is None:
        adamw_lr = lr * adamw_lr_ratio
    if adamw_weight_decay is None:
        adamw_weight_decay = weight_decay * adamw_weight_decay_ratio

    muon_params, adamw_params = split_params(
        param_entries,
        max_muon_aspect_ratio=max_muon_aspect_ratio,
        max_muon_dim=max_muon_dim,
    )
    if not muon_params:
        param_summary = summarize_param_entries(
            param_entries,
            recovered_named_entries=recovered_named_entries,
        )
        raise ValueError(
            "Muon optimizer requires at least one eligible 2D parameter; "
            f"{param_summary}"
        )

    param_groups: list[dict[str, object]] = [
        {
            "params": muon_params,
            "use_muon": True,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "ns_dtype": ns_dtype,
        }
    ]
    if adamw_params:
        param_groups.append(
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": adamw_lr,
                "betas": tuple(betas),
                "eps": adamw_eps,
                "weight_decay": adamw_weight_decay,
            }
        )
    return param_groups


def zeropower_via_newtonschulz5(
    grad: torch.Tensor,
    *,
    steps: int,
    ns_dtype: str = "bfloat16",
) -> torch.Tensor:
    if grad.ndim < MIN_MATRIX_NDIM:
        raise ValueError(
            "zeropower_via_newtonschulz5 expects ndim >= "
            f"{MIN_MATRIX_NDIM}, got {grad.ndim}"
        )
    if ns_dtype not in SUPPORTED_NS_DTYPES:
        raise ValueError(
            f"Unsupported Muon ns_dtype={ns_dtype!r}; expected one of "
            f"{sorted(SUPPORTED_NS_DTYPES)}"
        )
    dtype = torch.bfloat16 if ns_dtype == "bfloat16" else torch.float32
    x = grad.to(dtype=dtype)
    if grad.size(-2) > grad.size(-1):
        x = x.mT

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + EPS)
    for _ in range(steps):
        gram = x @ x.mT
        quintic = DEFAULT_B * gram + DEFAULT_C * gram @ gram
        x = DEFAULT_A * x + quintic @ x

    if grad.size(-2) > grad.size(-1):
        x = x.mT
    return x




def _is_dtensor_tensor(tensor: torch.Tensor) -> bool:
    return DTensor is not None and isinstance(tensor, DTensor)


def _materialize_full_tensor_for_muon(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, tuple[object, tuple[object, ...]] | None]:
    if not _is_dtensor_tensor(tensor):
        return tensor, None

    return tensor.full_tensor(), (tensor.device_mesh, tuple(tensor.placements))


def _redistribute_muon_update_like(
    dense_update: torch.Tensor,
    layout: tuple[object, tuple[object, ...]] | None,
) -> torch.Tensor:
    if layout is None:
        return dense_update
    if DTensor is None or Replicate is None:
        raise RuntimeError(
            "Cannot restore DTensor Muon update because "
            "torch.distributed.tensor is unavailable"
        )

    device_mesh, placements = layout
    replicated_placements = tuple(Replicate() for _ in placements)
    replicated = DTensor.from_local(
        dense_update.contiguous(),
        device_mesh=device_mesh,
        placements=replicated_placements,
        run_check=False,
    )
    return replicated.redistribute(
        device_mesh=device_mesh,
        placements=placements,
    )

def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    beta: float,
    ns_steps: int,
    nesterov: bool,
    ns_dtype: str = "bfloat16",
) -> torch.Tensor:
    momentum_buffer.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buffer, beta) if nesterov else momentum_buffer
    if update.ndim == CONV_KERNEL_NDIM:
        update = update.view(len(update), -1)
    full_update, original_layout = _materialize_full_tensor_for_muon(update)
    full_update = zeropower_via_newtonschulz5(
        full_update,
        steps=ns_steps,
        ns_dtype=ns_dtype,
    )
    full_update *= math.sqrt(
        max(1.0, full_update.size(-2) / full_update.size(-1))
    )
    update = _redistribute_muon_update_like(full_update, original_layout)
    return update.to(dtype=grad.dtype)


def adjusted_muon_lr(
    base_lr: float,
    param: torch.nn.Parameter,
    adjust_lr_fn: str | None,
) -> float:
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        return base_lr
    if adjust_lr_fn == "match_rms_adamw":
        rows, cols = param.shape[:2]
        target_scale = 0.2 * math.sqrt(max(rows, cols))
        local_update_scale = math.sqrt(max(1.0, rows / cols))
        return base_lr * target_scale / local_update_scale
    raise ValueError(
        f"Unsupported Muon adjust_lr_fn={adjust_lr_fn!r}; expected one "
        "of None, 'original', or 'match_rms_adamw'"
    )


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    beta1, beta2 = betas
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    bias_corrected_avg = exp_avg / (1 - beta1**step)
    bias_corrected_avg_sq = exp_avg_sq / (1 - beta2**step)
    return bias_corrected_avg / (bias_corrected_avg_sq.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.95,
        *,
        nesterov: bool = True,
        ns_steps: int = DEFAULT_NS_STEPS,
        adamw_lr: float | None = None,
        adamw_lr_ratio: float = 1.0,
        adamw_weight_decay: float | None = None,
        adamw_weight_decay_ratio: float = 1.0,
        adamw_eps: float = 1e-8,
        adjust_lr_fn: str | None = None,
        max_muon_aspect_ratio: float = 8.0,
        max_muon_dim: int = 65536,
        ns_dtype: str = "bfloat16",
    ) -> None:
        param_groups = build_param_groups(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr_ratio,
            adamw_weight_decay=adamw_weight_decay,
            adamw_weight_decay_ratio=adamw_weight_decay_ratio,
            adamw_eps=adamw_eps,
            adjust_lr_fn=adjust_lr_fn,
            max_muon_aspect_ratio=max_muon_aspect_ratio,
            max_muon_dim=max_muon_dim,
            ns_dtype=ns_dtype,
        )
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if bool(group["use_muon"]):
                for param in group["params"]:
                    grad = param.grad
                    if grad is None:
                        continue
                    if not torch.isfinite(grad).all():
                        raise FloatingPointError(
                            f"Muon received non-finite gradient for "
                            f"parameter shape {tuple(param.shape)}"
                        )
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    update = muon_update(
                        grad,
                        state["momentum_buffer"],
                        beta=float(group["momentum"]),
                        ns_steps=int(group["ns_steps"]),
                        nesterov=bool(group["nesterov"]),
                        ns_dtype=str(group["ns_dtype"]),
                    )
                    if not torch.isfinite(update).all():
                        raise FloatingPointError(
                            f"Muon produced non-finite update for "
                            f"parameter shape {tuple(param.shape)}"
                        )
                    base_lr = float(group["lr"])
                    param.mul_(1 - base_lr * float(group["weight_decay"]))
                    param.add_(
                        update.reshape(param.shape),
                        alpha=-adjusted_muon_lr(
                            base_lr,
                            param,
                            str(group["adjust_lr_fn"])
                            if group["adjust_lr_fn"] is not None
                            else None,
                        ),
                    )
                    if not torch.isfinite(param).all():
                        raise FloatingPointError(
                            f"Muon produced non-finite parameter for "
                            f"shape {tuple(param.shape)}"
                        )
                continue

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["step"] = 0
                state["step"] += 1
                update = adam_update(
                    grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    step=int(state["step"]),
                    betas=tuple(group["betas"]),
                    eps=float(group["eps"]),
                )
                param.mul_(
                    1 - float(group["lr"]) * float(group["weight_decay"])
                )
                param.add_(update, alpha=-float(group["lr"]))

        return loss
