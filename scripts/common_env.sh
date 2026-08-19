#!/usr/bin/env bash
# Shared paths for CRG's pinned external dependencies.

_crg_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT:-${VERL_DIR:-$_crg_repo_root/.deps/verl}}"
export VERL_ROOT="${VERL_ROOT:-$VERL_SOURCE_ROOT}"
export VERL_UPSTREAM_ROOT="${VERL_UPSTREAM_ROOT:-$VERL_SOURCE_ROOT}"
export MLLM_CRL_UPSTREAM_ROOT="${MLLM_CRL_UPSTREAM_ROOT:-$_crg_repo_root/mllm-crl}"
export MINI_TRAINER_SRC_ROOT="${MINI_TRAINER_SRC_ROOT:-${MINI_TRAINER_DIR:-$_crg_repo_root/.deps/mini_trainer}/src}"
if [[ -z "${VERL_CONFIG_ROOT:-}" && -f "$_crg_repo_root/scripts/locate_verl_config.py" ]]; then
  export VERL_CONFIG_ROOT="${VERL_SOURCE_ROOT}/verl/trainer/config"
fi
export PYTHONPATH="$_crg_repo_root/continual-rlvr-algorithms:$_crg_repo_root/mllm-crl:$VERL_SOURCE_ROOT:${PYTHONPATH:-}"
unset _crg_repo_root
