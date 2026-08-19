#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-./tiktoken_cache}"

project_name="mllm_cl"
exp_name="qwen3_4b_grpo_fsdp_vllm_crg"
exp_dir="${exp_name}_$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$exp_dir"
n_gpu="${N_GPU:-4}"
n_cpu="${N_CPU:-96}"
model_path="${MODEL_PATH:-Qwen/Qwen3-4B}"

task_config_dir="$(
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path

import mllm_crl

print(
    Path(mllm_crl.__file__).resolve().parent
    / "task"
    / "reasoning_gym"
    / "config"
)
PY
)"

############################ Parameter Groups ############################

DATA=(
    data.max_prompt_length=1024
    data.max_response_length=1024
    data.train_batch_size=512
    data.filter_overlong_prompts=True
    data.truncation="error"
    data.shuffle=False
    data.dataloader_num_workers=8
)

MODEL=(
    actor_rollout_ref.model.path=$model_path
    actor_rollout_ref.model.use_shm=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=256
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0.001
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55
    actor_rollout_ref.rollout.load_format="safetensors"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name="vllm"
    actor_rollout_ref.rollout.n=5
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
    ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.use_torch_compile=False
)

ALGORITHM=(
    algorithm.adv_estimator="grpo"
)

TRAINER=(
  trainer.n_gpus_per_node=$n_gpu
  trainer.project_name=$project_name
  trainer.experiment_name=$exp_name
  trainer.val_before_train=False
  trainer.default_local_dir=$exp_dir
  trainer.total_epochs=1
  trainer.total_training_steps=500
  trainer.save_freq=100
  trainer.test_freq=100

)

MISCS=(
  custom_reward_function.path=pkg://mllm_crl.task.reasoning_gym.reward
  ray_kwargs.ray_init.num_cpus=$n_cpu
  "hydra.searchpath=[file://${VERL_CONFIG_ROOT:?Set VERL_CONFIG_ROOT to the installed VERL config directory}]"
  hydra.run.dir=$exp_dir
)

FSDP=(
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.fsdp_config.forward_prefetch=True
)

############################ Launch ############################

"$PYTHON_BIN" -m mllm_crl.train \
    --config-name ppo_trainer \
    -- \
    ++task_config="$task_config_dir"/inter_generalisation_algorithmic.yaml \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${ALGORITHM[@]}" \
    "${TRAINER[@]}" \
    "${MISCS[@]}" \
    "${FSDP[@]}" \
    "$@" 2>&1 | tee "$exp_dir"/train_log.txt
