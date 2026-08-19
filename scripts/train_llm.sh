#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/train_llm.sh METHOD SETTING [HYDRA_OVERRIDE ...]

METHOD:  seq_rlvr | mtrl | cpr | ewc | fire | kl | muon | osft | redo
SETTING: algorithmic | algebra

Environment variables:
  MODEL_PATH   Hugging Face model ID or local model path (default: Qwen/Qwen3-4B)
  N_GPU        GPUs on the node (default: 8)
  N_CPU        Ray CPU budget (default: 96)
  OUTPUT_ROOT  Experiment-output directory (default: ./outputs)
  PYTHON_BIN   Python executable (default: python3)
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
  algorithmic|algebra) ;;
  *) echo "Unknown setting: $setting" >&2; usage >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$repo_root/scripts/common_env.sh"
case "$method" in
  seq_rlvr)
    launcher="$repo_root/mllm-crl/example_scripts/reasoning_gym/crl_tasks_${setting}_qwen3_4b_grpo_fsdp_vllm.sh"
    ;;
  mtrl)
    if [[ "$setting" == "algorithmic" ]]; then
      launcher="$repo_root/mllm-crl/example_scripts/reasoning_gym/algorithmic_qwen3_4b_grpo_fsdp_vllm.sh"
    else
      launcher="$repo_root/mllm-crl/example_scripts/reasoning_gym/multitask_algebra_qwen3_4b_grpo_fsdp_vllm.sh"
    fi
    ;;
  cpr)
    launcher="$repo_root/continual-rlvr-algorithms/example_scripts/reasoning_gym/$setting/crl_tasks_prompt_replay_qwen3_4b_grpo_fsdp_vllm.sh"
    ;;
  ewc|fire|muon|osft|redo)
    launcher="$repo_root/continual-rlvr-algorithms/example_scripts/reasoning_gym/$setting/crl_tasks_${method}_qwen3_4b_grpo_fsdp_vllm.sh"
    ;;
  kl)
    launcher="$repo_root/continual-rlvr-algorithms/example_scripts/reasoning_gym/$setting/crl_tasks_kl_to_old_policy_old_prompts_qwen3_4b_grpo_fsdp_vllm.sh"
    ;;
  *) echo "Unknown method: $method" >&2; usage >&2; exit 2 ;;
esac

if [[ ! -f "$launcher" ]]; then
  echo "Launcher not found: $launcher" >&2
  exit 1
fi

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
export N_GPU="${N_GPU:-8}"
export N_CPU="${N_CPU:-96}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs}"
mkdir -p "$output_root"
cd "$output_root"

exec bash "$launcher" "$@"
