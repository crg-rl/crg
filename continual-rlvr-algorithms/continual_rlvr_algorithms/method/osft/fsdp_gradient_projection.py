from __future__ import annotations

import types
from typing import Any

import torch

OSFT_FACTOR_ATTRS = (
    "osft_params",
    "osft_U_high",
    "osft_S_high",
    "osft_V_high",
)


def has_osft_factors(module: torch.nn.Module) -> bool:
    return all(hasattr(module, attr) for attr in OSFT_FACTOR_ATTRS)


def collect_osft_modules(root: torch.nn.Module) -> list[torch.nn.Module]:
    return [module for module in root.modules() if has_osft_factors(module)]


def ensure_matrix(tensor: torch.Tensor, name: str) -> None:
    if tensor.dim() != 2:
        raise RuntimeError(
            "OSFT gradient projection expected a full 2D tensor for "
            f"{name}, got shape={tuple(tensor.shape)}. For FSDP use_orig_params "
            "this projection must run inside FSDP.summon_full_params(..., "
            "with_grads=True); otherwise OSFT factors appear as 1D local shards."
        )


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def copy_projected_gradient(
    original: torch.Tensor, projected: torch.Tensor
) -> None:
    local_storage = getattr(original, "_local_tensor", None)
    if local_storage is not None:
        local_storage.copy_(projected)
        return
    original.copy_(projected)


def project_osft_modules_locally(osft_modules: list[torch.nn.Module]) -> int:
    """Project OSFT low-rank gradients when full params/grads are materialized.

    mini_trainer.project_gradients assumes distributed local shards are still
    matrix-shaped (for example DTensor row shards). PyTorch FSDP with
    use_orig_params=True exposes OSFT factors as 1D local shards outside a full
    parameter context, so the adapter summons full params and applies the same
    orthogonal projection without distributed all-reduce.
    """

    projected = 0
    with torch.no_grad():
        for module in osft_modules:
            module_svd = module.osft_params

            if module_svd.U_low.grad is not None:
                u_high = local_tensor(module.osft_U_high)
                d_u = local_tensor(module_svd.U_low.grad)
                ensure_matrix(u_high, "U_high")
                ensure_matrix(d_u, "U_low.grad")
                proj_coeff = torch.mm(u_high.transpose(0, 1), d_u)
                d_u.addmm_(u_high, proj_coeff, alpha=-1.0)
                copy_projected_gradient(module_svd.U_low.grad, d_u)
                projected += 1

            if module_svd.V_low.grad is not None:
                v_high = local_tensor(module.osft_V_high)
                d_v = local_tensor(module_svd.V_low.grad)
                ensure_matrix(v_high, "V_high")
                ensure_matrix(d_v, "V_low.grad")
                gram = torch.mm(v_high.transpose(0, 1), v_high)
                update = torch.mm(d_v, gram)
                d_v.add_(update, alpha=-1.0)
                copy_projected_gradient(module_svd.V_low.grad, d_v)
                projected += 1
    return projected


def iter_leaf_fsdp_modules(
    root: torch.nn.Module, fsdp_cls: type
) -> list[torch.nn.Module]:
    fsdp_modules = [
        module for module in root.modules() if isinstance(module, fsdp_cls)
    ]
    if not fsdp_modules:
        return []

    leaves: list[torch.nn.Module] = []
    for module in fsdp_modules:
        has_child_fsdp = any(
            child is not module and isinstance(child, fsdp_cls)
            for child in module.modules()
        )
        if not has_child_fsdp:
            leaves.append(module)
    return leaves


def project_fsdp_osft_gradients(
    actor_module_fsdp: torch.nn.Module, fsdp_cls: type
) -> int:
    """Project OSFT gradients under FSDP full-param contexts.

    Leaf FSDP units are processed one at a time to avoid materializing the whole
    model when the wrap policy shards transformer blocks. If no leaf exposes
    OSFT modules, fall back to the root context so single-FSDP layouts still work.
    """

    projected = 0
    for fsdp_module in iter_leaf_fsdp_modules(actor_module_fsdp, fsdp_cls):
        wrapped = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
        osft_modules = collect_osft_modules(wrapped)
        if not osft_modules:
            continue
        with fsdp_cls.summon_full_params(
            fsdp_module,
            recurse=True,
            writeback=True,
            offload_to_cpu=False,
            with_grads=True,
        ):
            projected += project_osft_modules_locally(osft_modules)

    if projected == 0 and isinstance(actor_module_fsdp, fsdp_cls):
        wrapped = getattr(
            actor_module_fsdp, "_fsdp_wrapped_module", actor_module_fsdp
        )
        osft_modules = collect_osft_modules(wrapped)
        if osft_modules:
            with fsdp_cls.summon_full_params(
                actor_module_fsdp,
                recurse=True,
                writeback=True,
                offload_to_cpu=False,
                with_grads=True,
            ):
                projected += project_osft_modules_locally(osft_modules)

    return projected


def wrap_optimizer_with_osft_projection(
    optimizer: Any,
    actor_module_fsdp: torch.nn.Module,
    *,
    fsdp_cls: type,
) -> Any:
    """Wrap optimizer.step with an FSDP-safe OSFT gradient projection."""

    actor_model = getattr(
        actor_module_fsdp, "_fsdp_wrapped_module", actor_module_fsdp
    )
    is_fsdp_actor = isinstance(actor_module_fsdp, fsdp_cls)
    has_unsharded_projector = callable(
        getattr(actor_model, "project_gradients", None)
    )
    if not is_fsdp_actor and not has_unsharded_projector:
        return optimizer

    orig_step = optimizer.step

    def step(self, *args, **kwargs):
        if is_fsdp_actor:
            project_fsdp_osft_gradients(actor_module_fsdp, fsdp_cls)
        else:
            actor_model.project_gradients()
        return orig_step(*args, **kwargs)

    optimizer.step = types.MethodType(step, optimizer)
    optimizer._osft_fsdp_safe_projection_wrapped = True
    return optimizer
