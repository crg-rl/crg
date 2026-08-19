set -euo pipefail

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true

project_name="mllm_cl"
exp_name="qwen2_5vl_7b_multitask_visulogic_spatial_reasoning_grpo_fsdp_vllm"
exp_dir="${exp_name}_$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$exp_dir"

export TIKTOKEN_ENCODINGS_BASE="$exp_dir/tiktoken_cache"
export TIKTOKEN_CACHE_SOURCE_DIR="./tiktoken_cache"

n_gpu="${N_GPU:-8}"
n_cpu="${N_CPU:-128}"
model_path="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

domain_name="spatial_reasoning"
domain_tag="Spatial Reasoning"
data_root="./data/visulogic/domain-internal/${domain_name}/vanilla_multitask"
train_file="${data_root}/train.parquet"
val_file="${data_root}/val.parquet"
train_batch_size=512
val_batch_size=128
total_training_steps=500
total_epochs=40

if [[ ! -f $train_file || ! -f $val_file ]]; then
  echo "[train] Missing VisuLogic multitask parquet:"
  echo "  - $train_file"
  echo "  - $val_file"
  exit 1
fi

train_file_abs=$(readlink -f "$train_file")
val_file_abs=$(readlink -f "$val_file")

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
export MLLM_CRL_REPO_ROOT="$repo_root"

mkdir -p "$TIKTOKEN_ENCODINGS_BASE"
if [[ -d "$TIKTOKEN_CACHE_SOURCE_DIR" ]]; then
    cp -a "$TIKTOKEN_CACHE_SOURCE_DIR"/. "$TIKTOKEN_ENCODINGS_BASE"/
fi

VERL_CONFIG_ROOT="${VERL_CONFIG_ROOT:?Set VERL_CONFIG_ROOT to the installed VERL config directory}"

############################ Parameter Groups ############################

DATA=(
    data.train_files="['$train_file_abs']"
    data.val_files="['$val_file_abs']"
    data.train_batch_size=$train_batch_size
    data.val_batch_size=$val_batch_size
    data.train_max_samples=-1
    data.val_max_samples=-1
    data.max_prompt_length=1024
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation="error"
    data.image_key=images
    data.return_multi_modal_inputs=True
    data.dataloader_num_workers=0
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="$model_path"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=256
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0.001
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65
    actor_rollout_ref.rollout.load_format="safetensors"
    actor_rollout_ref.rollout.max_model_len=3072
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
)

TRAINER=(
    trainer.logger=console
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.n_gpus_per_node=$n_gpu
    trainer.nnodes=1
    trainer.save_freq=$total_training_steps
    trainer.test_freq=$total_training_steps
    trainer.total_epochs=$total_epochs
    trainer.total_training_steps=$total_training_steps
    trainer.default_local_dir=$exp_dir
    trainer.val_before_train=False
    trainer.resume_mode=auto
)

MISCS=(
    custom_reward_function.path=pkg://mllm_crl.task.visulogic.reward
    custom_reward_function.name=compute_score
    ray_kwargs.ray_init.num_cpus=$n_cpu
    "hydra.searchpath=[file://${VERL_CONFIG_ROOT:?Set VERL_CONFIG_ROOT to the installed VERL config directory}]"
    hydra.run.dir=$exp_dir
    "++ray_kwargs.ray_init.runtime_env.env_vars.TIKTOKEN_ENCODINGS_BASE=$TIKTOKEN_ENCODINGS_BASE"
    "++ray_kwargs.ray_init.runtime_env.env_vars.TIKTOKEN_CACHE_SOURCE_DIR=$TIKTOKEN_CACHE_SOURCE_DIR"
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$repo_root:$verl_root:${PYTHONPATH:-}"
    "++ray_kwargs.ray_init.runtime_env.env_vars.VISULOGIC_DOMAIN_TAG=$domain_tag"
)

FSDP=(
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

############################ Launch ############################

PYTHONPATH="$repo_root:$verl_root:${PYTHONPATH:-}" \
python -m verl.trainer.main_ppo \
    --config-name ppo_trainer \
    -- \
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
