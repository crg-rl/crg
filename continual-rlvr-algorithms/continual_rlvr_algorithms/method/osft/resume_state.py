from __future__ import annotations

from copy import deepcopy


def expected_crl_batches_yielded(global_step: int, dataloader_len: int) -> int:
    if global_step < 0:
        raise ValueError(
            f"global_step must be non-negative, got {global_step}"
        )
    if dataloader_len <= 0:
        raise ValueError(
            f"dataloader_len must be positive, got {dataloader_len}"
        )
    return global_step % dataloader_len


def expected_crl_dataset_state(
    global_step: int,
    steps_per_task: int,
    task_count: int,
) -> dict[str, int]:
    if global_step < 0:
        raise ValueError(
            f"global_step must be non-negative, got {global_step}"
        )
    if steps_per_task <= 0:
        raise ValueError(
            f"steps_per_task must be positive, got {steps_per_task}"
        )
    if task_count <= 0:
        raise ValueError(f"task_count must be positive, got {task_count}")

    train_steps = max(global_step - 1, 0)
    task_idx = (train_steps // steps_per_task) % task_count
    return {
        "task_idx": int(task_idx),
        "train_steps": int(train_steps),
    }


def repair_crl_dataloader_state_dict(
    state_dict: dict,
    *,
    global_step: int,
    dataloader_len: int,
    train_batch_size: int,
    steps_per_task: int,
    task_count: int,
) -> tuple[dict, bool]:
    if train_batch_size <= 0:
        raise ValueError(
            f"train_batch_size must be positive, got {train_batch_size}"
        )

    repaired_state = deepcopy(state_dict)
    expected_batches = expected_crl_batches_yielded(
        global_step, dataloader_len
    )
    expected_dataset = expected_crl_dataset_state(
        global_step=global_step,
        steps_per_task=steps_per_task,
        task_count=task_count,
    )
    changed = False

    if repaired_state.get("_num_yielded") != expected_batches:
        repaired_state["_num_yielded"] = expected_batches
        changed = True

    if repaired_state.get("_sampler_iter_yielded") != expected_batches:
        repaired_state["_sampler_iter_yielded"] = expected_batches
        changed = True

    sampler_iter_state = repaired_state.get("_sampler_iter_state")
    if sampler_iter_state is None:
        sampler_iter_state = {}
        repaired_state["_sampler_iter_state"] = sampler_iter_state
        changed = True

    expected_samples = expected_batches * train_batch_size
    if sampler_iter_state.get("samples_yielded") != expected_samples:
        sampler_iter_state["samples_yielded"] = expected_samples
        changed = True

    dataset_state = repaired_state.get("dataset_state")
    if dataset_state is None:
        dataset_state = {}
        repaired_state["dataset_state"] = dataset_state
        changed = True

    for key, value in expected_dataset.items():
        if dataset_state.get(key) != value:
            dataset_state[key] = value
            changed = True

    return repaired_state, changed
