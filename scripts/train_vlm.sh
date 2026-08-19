#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_vlm.sh METHOD SETTING [HYDRA_OVERRIDE ...]

METHOD:  seq_rlvr | mtrl | cpr | ewc | fire | kl | muon | osft | redo
SETTING: quantitative | spatial | positional

Set VISULOGIC_DATA_ROOT to a prepared setting directory containing
train.parquet and val.parquet. If unset, data/visulogic/SETTING is used.
EOF
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi
method="$1"
setting="$2"
shift 2

case "$setting" in
  quantitative|spatial|positional) ;;
  *) echo "Unknown setting: $setting" >&2; usage >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$repo_root/scripts/common_env.sh"
case "$method" in
  seq_rlvr)
    launcher="$repo_root/mllm-crl/example_scripts/visulogic/crl/crl_tasks_visulogic_${setting}_reasoning_qwen2_5vl_7b_grpo_fsdp_vllm.sh"
    ;;
  mtrl)
    launcher="$repo_root/mllm-crl/example_scripts/visulogic/multitask/multitask_visulogic_${setting}_reasoning_qwen2_5vl_7b_grpo_fsdp_vllm.sh"
    ;;
  cpr)
    launcher="$repo_root/continual-rlvr-algorithms/example_scripts/visulogic/crl/crl_tasks_visulogic_${setting}_reasoning_prompt_replay_qwen2_5vl_7b_grpo_fsdp_vllm.sh"
    ;;
  ewc|fire|muon|osft|redo)
    launcher="$repo_root/continual-rlvr-algorithms/example_scripts/visulogic/crl/crl_tasks_visulogic_${setting}_reasoning_${method}_qwen2_5vl_7b_grpo_fsdp_vllm.sh"
    ;;
  kl)
    launcher="$repo_root/continual-rlvr-algorithms/example_scripts/visulogic/crl/crl_tasks_visulogic_${setting}_reasoning_kl_to_old_policy_old_prompts_qwen2_5vl_7b_grpo_fsdp_vllm.sh"
    ;;
  *) echo "Unknown method: $method" >&2; usage >&2; exit 2 ;;
esac

if [[ ! -f "$launcher" ]]; then
  echo "Launcher not found: $launcher" >&2
  exit 1
fi

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
export N_GPU="${N_GPU:-8}"
export N_CPU="${N_CPU:-128}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export VISULOGIC_DATA_ROOT="${VISULOGIC_DATA_ROOT:-$repo_root/data/visulogic/$setting}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs}"
mkdir -p "$output_root"
cd "$output_root"

exec bash "$launcher" "$@"
