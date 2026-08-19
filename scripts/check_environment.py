#!/usr/bin/env python3
"""Fail-fast import and accelerator check for the CRG training stack."""

from __future__ import annotations

import importlib
import json
import os
import sys
from importlib import metadata
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    defaults = {
        "VERL_UPSTREAM_ROOT": repo_root / ".deps" / "verl",
        "MLLM_CRL_UPSTREAM_ROOT": repo_root / "mllm-crl",
        "MINI_TRAINER_SRC_ROOT": repo_root / ".deps" / "mini_trainer" / "src",
    }
    for name, path in defaults.items():
        os.environ.setdefault(name, str(path))
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise RuntimeError(f"expected Python 3.11 or 3.12, found {sys.version}")

    versions = {}
    packages = {
        "torch": "torch",
        "vllm": "vllm",
        "verl": "verl",
        "reasoning_gym": "reasoning-gym",
        "mllm_crl": "mllm-crl",
        "continual_rlvr_algorithms": "continual-rlvr-algorithms",
    }
    for module_name, distribution_name in packages.items():
        module = importlib.import_module(module_name)
        try:
            version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "installed")
        versions[module_name] = version

    import torch
    importlib.import_module(
        "continual_rlvr_algorithms.method.osft.auto_model_patch"
    )
    versions["mini_trainer"] = "installed"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    payload = {
        "status": "PASS",
        "python": sys.version.split()[0],
        "packages": versions,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
