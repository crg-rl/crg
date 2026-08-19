# CRG task and evaluation substrate

This package provides the task streams, rewards, vanilla sequential/MTRL
launchers, and evaluation utilities used by Continual Reasoning Gym.

- `mllm_crl/task/reasoning_gym/`: generated text reasoning streams
- `mllm_crl/task/visulogic/`: visual stage-switching dataset and verifier
- `evaluation/crl_metric/`: matrix materialization, BaseAvg, FinalAvg, FWT, TLG, BWT, and CTM
- `evaluation/`: base-ability, VLM-ability, and entropy diagnostics

All included stream launchers use a 500-update budget. Model and data paths
are configured through environment variables.
