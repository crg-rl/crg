#!/usr/bin/env python3
"""Fail-fast import and accelerator check for the CRG training stack."""

from __future__ import annotations

import importlib
import json
import os
import sys
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
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"expected Python 3.11, found {sys.version}")

    versions = {}
    for name in (
        "torch",
        "vllm",
        "verl",
        "reasoning_gym",
        "mllm_crl",
        "continual_rlvr_algorithms",
    ):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "installed")

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
