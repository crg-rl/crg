from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent

DEFAULT_MLLM_CRL_UPSTREAM_ROOT = WORKSPACE_ROOT / "mllm-crl"
DEFAULT_VERL_UPSTREAM_ROOT = WORKSPACE_ROOT / ".deps" / "verl"
DEFAULT_MINI_TRAINER_SRC_ROOT = (
    WORKSPACE_ROOT / ".deps" / "mini_trainer" / "src"
)


def _resolve_env_path(env_name: str, default_path: Path) -> Path:
    raw_value = os.environ.get(env_name)
    return Path(raw_value).resolve() if raw_value else default_path.resolve()


def resolve_mini_trainer_src_root() -> Path:
    raw_value = os.environ.get("MINI_TRAINER_SRC_ROOT")
    if raw_value:
        path = Path(raw_value).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"MINI_TRAINER_SRC_ROOT does not exist: {path}"
            )
        return path
    if DEFAULT_MINI_TRAINER_SRC_ROOT.exists():
        return DEFAULT_MINI_TRAINER_SRC_ROOT.resolve()
    raise FileNotFoundError(
        "Missing required mini_trainer source tree at "
        f"{DEFAULT_MINI_TRAINER_SRC_ROOT}. Run scripts/setup.sh or set "
        "MINI_TRAINER_SRC_ROOT=/path/to/mini_trainer/src."
    )


def install_mini_trainer_lightweight_import_patch() -> None:
    """Avoid importing mini_trainer.__init__ and unrelated optional deps.

    The OSFT integration only needs selected modules such as
    ``mini_trainer.osft_utils`` and ``mini_trainer.utils``. Upstream
    ``mini_trainer.__init__`` eagerly imports training/sampler modules with
    optional dependencies that are not required for this path and may be absent
    in benchmark containers. Preloading a lightweight package keeps normal
    submodule imports working without pulling those optional modules.
    """

    package_name = "mini_trainer"
    existing = sys.modules.get(package_name)
    if existing is not None and hasattr(existing, "__path__"):
        return

    src_root = resolve_mini_trainer_src_root()
    package_dir = src_root / package_name
    if not package_dir.exists():
        raise FileNotFoundError(
            f"Missing mini_trainer package directory: {package_dir}"
        )

    import types

    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    package.__file__ = str(package_dir / "__init__.py")
    package.__package__ = package_name
    sys.modules[package_name] = package


def ensure_upstream_paths() -> None:
    candidate_paths = [
        REPO_ROOT.resolve(),
        _resolve_env_path(
            "MLLM_CRL_UPSTREAM_ROOT", DEFAULT_MLLM_CRL_UPSTREAM_ROOT
        ),
        _resolve_env_path("VERL_UPSTREAM_ROOT", DEFAULT_VERL_UPSTREAM_ROOT),
        resolve_mini_trainer_src_root(),
    ]
    missing_paths = [path for path in candidate_paths[1:] if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Missing required upstream paths for mllm-crl-osft: "
            + missing_text
        )
    for path in reversed(candidate_paths):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    install_mini_trainer_lightweight_import_patch()

    existing_pythonpath = [
        entry for entry in os.environ.get("PYTHONPATH", "").split(":") if entry
    ]
    preferred_pythonpath = [str(path) for path in candidate_paths]
    merged_pythonpath: list[str] = []
    for entry in preferred_pythonpath + existing_pythonpath:
        if entry not in merged_pythonpath:
            merged_pythonpath.append(entry)
    os.environ["PYTHONPATH"] = ":".join(merged_pythonpath)


def ensure_verl_config_compatibility() -> None:
    """Patch missing config classes that appear in serialized Ray exceptions.

    The upstream tree we depend on is read-only and currently does not define
    ``verl.workers.config.model.MtpConfig``. Some Ray exceptions still carry
    objects referencing that symbol, which then masks the real worker error
    during exception deserialization. Installing a local compatible placeholder
    keeps the original traceback visible without modifying upstream code.
    """

    import verl.workers.config as workers_config
    import verl.workers.config.model as model_config

    if hasattr(model_config, "MtpConfig"):
        return

    @dataclass
    class MtpConfig:
        enable: bool = False
        enable_train: bool = False
        enable_rollout: bool = False
        detach_encoder: bool = False
        mtp_loss_scaling_factor: float = 0.1
        speculative_algorithm: str = "EAGLE"
        speculative_num_steps: int = 2
        speculative_eagle_topk: int = 2
        speculative_num_draft_tokens: int = 4
        method: str = "mtp"
        num_speculative_tokens: int = 1

    MtpConfig.__module__ = model_config.__name__

    model_config.MtpConfig = MtpConfig
    workers_config.MtpConfig = MtpConfig


def sanitize_hf_model_config(config: Any) -> Any:
    """Return a copy of model config compatible with upstream HFModelConfig."""

    if config is None:
        return config
    if isinstance(config, DictConfig):
        sanitized = OmegaConf.create(
            OmegaConf.to_container(config, resolve=False)
        )
    elif isinstance(config, dict):
        sanitized = dict(config)
    else:
        return config

    sanitized.pop("osft", None)
    return sanitized


def install_hf_model_config_compatibility_patch() -> None:
    """Strip wrapper-only fields before upstream HFModelConfig instantiation."""

    import verl.utils.config as config_utils
    from verl.workers.config import HFModelConfig

    current_impl = config_utils.omega_conf_to_dataclass
    if getattr(current_impl, "_continual_rlvr_osft_hf_model_patch", False):
        return

    def patched_omega_conf_to_dataclass(
        config: Any, dataclass_type: type[Any] | None = None
    ) -> Any:
        if (
            dataclass_type is HFModelConfig
            or getattr(dataclass_type, "__name__", None) == "HFModelConfig"
        ):
            config = sanitize_hf_model_config(config)
        return current_impl(config, dataclass_type=dataclass_type)

    patched_omega_conf_to_dataclass._continual_rlvr_osft_hf_model_patch = True  # type: ignore[attr-defined]
    config_utils.omega_conf_to_dataclass = patched_omega_conf_to_dataclass

    for module_name in (
        "verl.workers.fsdp_workers",
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
    ):
        module = sys.modules.get(module_name)
        if (
            module is not None
            and getattr(module, "omega_conf_to_dataclass", None) is not None
        ):
            module.omega_conf_to_dataclass = patched_omega_conf_to_dataclass
