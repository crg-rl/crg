#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
verl_dir="${VERL_DIR:-$repo_root/.deps/verl}"
mini_trainer_dir="${MINI_TRAINER_DIR:-$repo_root/.deps/mini_trainer}"
mini_trainer_commit="f5d63c202eedd7ede7a7ab074f41a0a080b88af8"
verl_commit="adff7956cefd8ef707cd67dd8e08c06fa63679bd"
patch_file="$repo_root/patches/verl-crg-runtime.patch"

"$python_bin" - <<'PY'
import sys
if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit(f"CRG requires Python 3.11 or 3.12; found {sys.version.split()[0]}")
PY

if ! "$python_bin" -c 'import torch, vllm' >/dev/null 2>&1; then
  echo "Install CUDA-compatible PyTorch 2.8.x and vLLM 0.11.x first." >&2
  exit 2
fi

checkout_dependency() {
  local url="$1"
  local directory="$2"
  local commit="$3"
  if [[ ! -d "$directory/.git" ]]; then
    git clone "$url" "$directory"
  fi
  if ! git -C "$directory" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "$directory" fetch --quiet origin "$commit"
  fi
  git -C "$directory" checkout --detach "$commit"
}

mkdir -p "$(dirname "$verl_dir")"
checkout_dependency   https://github.com/verl-project/verl.git   "$verl_dir"   "$verl_commit"
checkout_dependency   https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer.git   "$mini_trainer_dir"   "$mini_trainer_commit"

if git -C "$verl_dir" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
  : # Patch is already applied.
else
  git -C "$verl_dir" apply --check "$patch_file"
  git -C "$verl_dir" apply "$patch_file"
fi

"$python_bin" -m pip install -e "$verl_dir"
"$python_bin" -m pip install -e "$repo_root/mllm-crl"
"$python_bin" -m pip install -e "$repo_root/continual-rlvr-algorithms"

export VERL_DIR="$verl_dir"
export MINI_TRAINER_DIR="$mini_trainer_dir"
# shellcheck disable=SC1091
source "$repo_root/scripts/common_env.sh"
export VERL_CONFIG_ROOT="$($python_bin "$repo_root/scripts/locate_verl_config.py")"
printf 'CRG_SETUP_PASS\nVERL_DIR=%s\nVERL_CONFIG_ROOT=%s\n' \
  "$verl_dir" "$VERL_CONFIG_ROOT"
