# Continual RLVR algorithms

This package contains the continual-learning methods used by Continual
Reasoning Gym: EWC, FIRE, KL regularization, Muon, OSFT, ReDo, and Continual
Prompt Replay (CPR). It also includes the stale-trajectory
sample-replay ablation. Launch scripts are organized by text and visual task
stream. Install the pinned external VERL dependency and `../mllm-crl` before
this package; the root README gives the exact commands.

CPR launchers set `replay_scope=previous_tasks`; the sampler stores
prompts and source-task verifier metadata and regenerates responses with the
current policy.

The sample-replay launcher lives under
`example_scripts/reasoning_gym/algorithmic/ablations/`. It explicitly enables
`off_policy=true`; the source default remains false.
