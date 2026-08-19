from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReleaseSurfaceTests(unittest.TestCase):
    def test_public_wrapper_targets_exist(self) -> None:
        expected = [
            "mllm-crl/example_scripts/reasoning_gym/crl_tasks_algorithmic_qwen3_4b_grpo_fsdp_vllm.sh",
            "mllm-crl/example_scripts/reasoning_gym/crl_tasks_algebra_qwen3_4b_grpo_fsdp_vllm.sh",
            "mllm-crl/example_scripts/reasoning_gym/algorithmic_qwen3_4b_grpo_fsdp_vllm.sh",
            "mllm-crl/example_scripts/reasoning_gym/multitask_algebra_qwen3_4b_grpo_fsdp_vllm.sh",
        ]
        for setting in ("algorithmic", "algebra"):
            for method in ("prompt_replay", "ewc", "fire", "muon", "osft", "redo"):
                expected.append(
                    "continual-rlvr-algorithms/example_scripts/reasoning_gym/"
                    f"{setting}/crl_tasks_{method}_qwen3_4b_grpo_fsdp_vllm.sh"
                )
            expected.append(
                "continual-rlvr-algorithms/example_scripts/reasoning_gym/"
                f"{setting}/crl_tasks_kl_to_old_policy_old_prompts_qwen3_4b_grpo_fsdp_vllm.sh"
            )

        for setting in ("quantitative", "spatial", "positional"):
            stem = f"visulogic_{setting}_reasoning"
            expected.extend(
                [
                    f"mllm-crl/example_scripts/visulogic/crl/crl_tasks_{stem}_qwen2_5vl_7b_grpo_fsdp_vllm.sh",
                    f"mllm-crl/example_scripts/visulogic/multitask/multitask_{stem}_qwen2_5vl_7b_grpo_fsdp_vllm.sh",
                ]
            )
            for method in ("prompt_replay", "ewc", "fire", "muon", "osft", "redo"):
                expected.append(
                    "continual-rlvr-algorithms/example_scripts/visulogic/crl/"
                    f"crl_tasks_{stem}_{method}_qwen2_5vl_7b_grpo_fsdp_vllm.sh"
                )
            expected.append(
                "continual-rlvr-algorithms/example_scripts/visulogic/crl/"
                f"crl_tasks_{stem}_kl_to_old_policy_old_prompts_qwen2_5vl_7b_grpo_fsdp_vllm.sh"
            )

        missing = [path for path in expected if not (REPO_ROOT / path).is_file()]
        self.assertEqual([], missing)

    def test_public_launchers_are_portable(self) -> None:
        failures: list[str] = []
        launcher_roots = (
            REPO_ROOT / "mllm-crl" / "example_scripts",
            REPO_ROOT / "continual-rlvr-algorithms" / "example_scripts",
        )
        paths = (
            path
            for launcher_root in launcher_roots
            for path in launcher_root.rglob("*.sh")
        )
        for path in paths:
            text = path.read_text(errors="replace")
            relative = str(path.relative_to(REPO_ROOT))
            if not text.startswith("#!/usr/bin/env bash\n"):
                failures.append(f"{relative}: missing bash shebang")
            if 'PYTHON_BIN="${PYTHON_BIN:-python3}"' not in text:
                failures.append(f"{relative}: missing PYTHON_BIN override")
            if "trainer.n_gpus_per_node=" in text and "${N_GPU:-" not in text:
                failures.append(f"{relative}: N_GPU override is ignored")
            if "ray_kwargs.ray_init.num_cpus=" in text and "${N_CPU:-" not in text:
                failures.append(f"{relative}: N_CPU override is ignored")
            if "visulogic" in relative and "VISULOGIC_DATA_ROOT" not in text:
                failures.append(f"{relative}: VISULOGIC_DATA_ROOT override is ignored")
            if "$workspace_code_root/verl" in text:
                failures.append(f"{relative}: stale VERL dependency layout")
        self.assertEqual([], failures)

    def test_shell_path_variables_are_defined_before_use(self) -> None:
        failures: list[str] = []
        variables = (
            "script_dir",
            "repo_root",
            "workspace_code_root",
            "mllm_crl_root",
            "verl_root",
            "data_root",
        )
        for path in REPO_ROOT.rglob("*.sh"):
            if ".git" in path.parts:
                continue
            text = path.read_text(errors="replace")
            for variable in variables:
                expansions = [
                    index
                    for token in (f"${variable}", f"${{{variable}}}")
                    if (index := text.find(token)) >= 0
                ]
                if not expansions:
                    continue
                assignment = re.search(rf"(?m)^{variable}=", text)
                if assignment is None or assignment.start() > min(expansions):
                    failures.append(
                        f"{path.relative_to(REPO_ROOT)}: {variable}"
                    )
        self.assertEqual([], failures)

    def test_verl_root_is_defined_before_use(self) -> None:
        failures: list[str] = []
        for path in REPO_ROOT.rglob("*.sh"):
            if ".git" in path.parts:
                continue
            text = path.read_text(errors="replace")
            if "$verl_root" not in text and "${verl_root" not in text:
                continue
            assignment = re.search(r"(?m)^verl_root=", text)
            first_use_positions = [
                index
                for token in ("$verl_root", "${verl_root")
                if (index := text.find(token)) >= 0
            ]
            if assignment is None or assignment.start() > min(first_use_positions):
                failures.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual([], failures)

    def test_internal_hardware_and_paths_are_absent(self) -> None:
        forbidden = [
            "910" + "b",
            "VLLM_" + "ASCEND",
            "/" + "home/leadtek",
            "/" + "home/ai-jingyan-train",
            "/" + "root/jd-coding",
            "dongdong-" + "posttrain",
            "luolirui" + ".1",
            "domain-" + "internal",
            "continual-reasoning-gym-" + "baseline-sources",
            "osft-" + "mini_trainer",
        ]
        hits: list[str] = []
        binary_suffixes = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".pyc"}
        for path in REPO_ROOT.rglob("*"):
            generated_dirs = {".git", ".ruff_cache", "__pycache__", "build", "dist"}
            if (
                not path.is_file()
                or generated_dirs.intersection(path.parts)
                or path.suffix.lower() in binary_suffixes
            ):
                continue
            text = path.read_text(errors="replace")
            for token in forbidden:
                if token.lower() in text.lower():
                    hits.append(f"{path.relative_to(REPO_ROOT)}: {token}")
        self.assertEqual([], hits)


if __name__ == "__main__":
    unittest.main()
