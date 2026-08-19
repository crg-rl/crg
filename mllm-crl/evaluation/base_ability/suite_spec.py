"""Suite specification for text base-ability evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

SUITE_SPEC_VERSION = 'base-ability-suite-v1'
RUN_RESULT_VERSION = 'base-ability-run-v1'
MAX_RATIO_SCORE = 1.0
MAX_PERCENT_SCORE = 100.0


@dataclass(frozen=True)
class BenchmarkSpec:
    """Human-maintained benchmark normalization contract."""

    name: str
    display_name: str
    aliases: tuple[str, ...]
    preferred_score_keys: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ScoreRecord:
    """One normalized benchmark score found in an input file."""

    benchmark: str
    score: float
    input_name: str
    metric_key: str


@dataclass(frozen=True)
class BenchmarkRunPlan:
    """One concrete lm-eval invocation for a benchmark."""

    benchmark: str
    task_name: str
    output_dir: Path
    command: tuple[str, ...]


BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        name='mmlu',
        display_name='MMLU',
        aliases=('mmlu', 'hendrycks_test', 'hendrycksTest'),
        preferred_score_keys=('acc', 'acc_norm', 'exact_match', 'score'),
        description='Massive Multitask Language Understanding accuracy.',
    ),
    BenchmarkSpec(
        name='hellaswag',
        display_name='HellaSwag',
        aliases=('hellaswag',),
        preferred_score_keys=('acc_norm', 'acc', 'score'),
        description='HellaSwag commonsense completion accuracy.',
    ),
    BenchmarkSpec(
        name='truthfulqa',
        display_name='TruthfulQA',
        aliases=('truthfulqa', 'truthfulqa_mc2', 'truthfulqa_mc1'),
        preferred_score_keys=('mc2', 'mc1', 'acc', 'score'),
        description='TruthfulQA multiple-choice truthfulness score.',
    ),
    BenchmarkSpec(
        name='ifeval',
        display_name='IFEval',
        aliases=('ifeval', 'ifeval_strict', 'ifeval_loose'),
        preferred_score_keys=(
            'prompt_level_strict_acc',
            'inst_level_strict_acc',
            'strict_accuracy',
            'prompt_level_loose_acc',
            'inst_level_loose_acc',
            'loose_accuracy',
            'exact_match',
            'score',
        ),
        description='Instruction-following evaluation accuracy.',
    ),
    BenchmarkSpec(
        name='winogrande',
        display_name='WinoGrande',
        aliases=('winogrande', 'winogrande_xl'),
        preferred_score_keys=('acc', 'acc_norm', 'score'),
        description='WinoGrande commonsense coreference accuracy.',
    ),
    BenchmarkSpec(
        name='humaneval',
        display_name='HumanEval',
        aliases=('humaneval', 'human_eval', 'humaneval_pass_at_1'),
        preferred_score_keys=(
            'pass@1',
            'pass_at_1',
            'base_pass@1',
            'base_pass_at_1',
            'score',
        ),
        description='HumanEval code-generation pass@1 score.',
    ),
)

CORE_BENCHMARKS: tuple[str, ...] = tuple(
    spec.name for spec in BENCHMARK_SPECS
)

LM_EVAL_TASKS: dict[str, str] = {
    'mmlu': 'mmlu',
    'hellaswag': 'hellaswag',
    'truthfulqa': 'truthfulqa_mc2',
    'ifeval': 'ifeval',
    'winogrande': 'winogrande',
    'humaneval': 'humaneval',
}

UNSAFE_CODE_BENCHMARKS: frozenset[str] = frozenset({'humaneval'})

SPEC_BY_NAME: dict[str, BenchmarkSpec] = {
    spec.name: spec for spec in BENCHMARK_SPECS
}

ALIASES: dict[str, str] = {
    alias: spec.name for spec in BENCHMARK_SPECS for alias in spec.aliases
}

PREFERRED_SCORE_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        key
        for spec in BENCHMARK_SPECS
        for key in spec.preferred_score_keys
    )
)

EXAMPLE_INPUT: dict[str, Any] = {
    'results': {
        'hellaswag': {'acc_norm': 0.784},
        'truthfulqa_mc2': {'mc2': 0.512},
        'ifeval': {'prompt_level_strict_acc': 0.610},
        'winogrande': {'acc': 0.736},
        'humaneval': {'pass@1': 0.268},
    },
    'groups': {
        'mmlu': {'acc': 0.652},
    },
}


def suite_spec_document() -> dict[str, Any]:
    """Return the human-readable input/output suite specification."""

    return {
        'schema_version': SUITE_SPEC_VERSION,
        'required_benchmarks': list(CORE_BENCHMARKS),
        'lm_eval_tasks': LM_EVAL_TASKS,
        'score_scale': (
            'Input scores must be ratios in [0, 1] or explicit percent '
            'strings such as "65%". Output scores are ratios in [0, 1].'
        ),
        'input_forms': {
            'lm_eval_json': (
                'official lm-eval JSON with results and/or groups mappings'
            ),
        },
        'output_fields': {
            'benchmarks': 'canonical benchmark -> normalized score',
            'suite_average': 'mean over present canonical benchmark scores',
            'missing_benchmarks': 'required benchmarks absent from inputs',
            'complete': 'true only when all required benchmarks are present',
            'sources': 'canonical benchmark -> source JSON path',
            'benchmark_details': (
                'selected display name, raw input name, metric key, source, '
                'and normalized score'
            ),
            'run': (
                'present in --checkpoint mode; benchmark commands, logs, '
                'statuses, and evaluator settings'
            ),
        },
        'example_input': EXAMPLE_INPUT,
    }
