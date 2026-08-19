#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-./tiktoken_cache}"

project_name="mllm_cl"
exp_name_base="qwen3_4b_single_tasks_algebra_grpo_fsdp_vllm_crg"
run_root="${exp_name_base}_$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$run_root"

n_gpu="${N_GPU:-4}"
n_cpu="${N_CPU:-96}"
model_path="${MODEL_PATH:-Qwen/Qwen3-4B}"

# Run each task independently for the same number of steps as one CRL task.
# Default matches crl_tasks_algebra_qwen3_4b_grpo_fsdp_vllm.sh: 500 steps / 6 tasks -> 84 steps per task.
steps_per_task="${STEPS_PER_TASK:-84}"

# With reasoning_gym.dataset_size=20000 and data.train_batch_size=512:
# len(dataloader) ~= floor(20000 / 512) = 39, so 13 epochs ~= 507 steps.
total_epochs=13

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

base_task_config="$task_config_dir"/crl_tasks_algebra.yaml

mapfile -t tasks < <(
  "$PYTHON_BIN" - <<PY
from omegaconf import OmegaConf

cfg = OmegaConf.load("$base_task_config")
for name in cfg.reasoning_gym.datasets.keys():
    print(name)
PY
)

if [[ -n "${TASKS:-}" ]]; then
  read -r -a tasks <<<"$TASKS"
fi

############################ Parameter Groups ############################

DATA=(
    data.max_prompt_length=1024
    data.max_response_length=1024
    data.train_batch_size=512
    data.filter_overlong_prompts=True
    data.truncation="error"
    data.shuffle=False
    data.dataloader_num_workers=0
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

MISCS=(
  custom_reward_function.path=pkg://mllm_crl.task.reasoning_gym.reward
  ray_kwargs.ray_init.num_cpus=$n_cpu
  "hydra.searchpath=[file://${VERL_CONFIG_ROOT:?Set VERL_CONFIG_ROOT to the installed VERL config directory}]"
)

FSDP=(
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.fsdp_config.forward_prefetch=True
)

############################ Launch ############################

for task in "${tasks[@]}"; do
  task_exp_name="${exp_name_base}__${task}"
  task_dir="${run_root}/${task}"
  mkdir -p "$task_dir"

  task_config="$task_dir"/task_config.yaml
  "$PYTHON_BIN" - <<PY
from omegaconf import OmegaConf

cfg = OmegaConf.load("$base_task_config")
task = "$task"
if task not in cfg.reasoning_gym.datasets:
    raise SystemExit(f"Unknown task: {task}")

cfg.reasoning_gym.datasets = {task: cfg.reasoning_gym.datasets[task]}
cfg.crl.enabled = False
OmegaConf.save(cfg, "$task_config")
PY

  TRAINER=(
    trainer.n_gpus_per_node=$n_gpu
    trainer.project_name=$project_name
    trainer.experiment_name=$task_exp_name
    trainer.val_before_train=False
    trainer.default_local_dir=$task_dir
    trainer.total_epochs=$total_epochs
    trainer.total_training_steps=$steps_per_task
    trainer.save_freq=$steps_per_task
    trainer.test_freq=$steps_per_task
    trainer.logger="['console','tensorboard']"
  )

  "$PYTHON_BIN" -m mllm_crl.train \
      --config-name ppo_trainer \
      -- \
      ++task_config="$task_config" \
      hydra.run.dir="$task_dir" \
      "${DATA[@]}" \
      "${MODEL[@]}" \
      "${ACTOR[@]}" \
      "${ROLLOUT[@]}" \
      "${REF[@]}" \
      "${ALGORITHM[@]}" \
      "${TRAINER[@]}" \
      "${MISCS[@]}" \
      "${FSDP[@]}" \
      "$@" 2>&1 | tee "$task_dir"/train_log.txt
done
