"""Suite specification for VLM general-ability evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

SUITE_SPEC_VERSION = 'vlm-general-ability-suite-v1'
RUN_RESULT_VERSION = 'vlm-general-ability-run-v1'
REFERENCE = 'arXiv:2510.21978'
MAX_RATIO_SCORE = 1.0
MAX_PERCENT_SCORE = 100.0
DEFAULT_EVAL_BATCH_SIZE = 32
DEFAULT_JUDGE_WORKER_NUM = 4
DEFAULT_MODEL_KWARGS: dict[str, str] = {}


@dataclass(frozen=True)
class BenchmarkSpec:
    """Human-maintained VLM benchmark normalization contract."""

    name: str
    display_name: str
    aliases: tuple[str, ...]
    preferred_score_keys: tuple[str, ...]
    description: str
    required: bool = True


@dataclass(frozen=True)
class ScoreRecord:
    """One normalized benchmark score found in an input file."""

    benchmark: str
    score: float
    input_name: str
    metric_key: str


@dataclass(frozen=True)
class BenchmarkRunPlan:
    """One concrete EvalScope + VLMEvalKit invocation."""

    benchmark: str
    dataset_name: str
    output_dir: Path
    command: tuple[str, ...]


BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        name='aokvqa',
        display_name='A-OKVQA',
        aliases=('aokvqa', 'a_okvqa', 'a-okvqa', 'aok_vqa'),
        preferred_score_keys=('acc', 'accuracy', 'exact_match', 'score'),
        description='Knowledge-based visual question answering.',
    ),
    BenchmarkSpec(
        name='aesbench',
        display_name='AesBench',
        aliases=('aesbench', 'aes_bench', 'aesthetics_bench'),
        preferred_score_keys=('acc', 'accuracy', 'score'),
        description='Image aesthetics / visual preference capability.',
    ),
    BenchmarkSpec(
        name='vstar',
        display_name='VStar',
        aliases=('vstar', 'v_star'),
        preferred_score_keys=('acc', 'accuracy', 'exact_match', 'score'),
        description='Spatio-temporal / visual reasoning diagnostic.',
    ),
    BenchmarkSpec(
        name='visonly',
        display_name='VisOnly',
        aliases=('visonly', 'vis_only', 'visual_only'),
        preferred_score_keys=('acc', 'accuracy', 'exact_match', 'score'),
        description='Visual-only recognition aggregate.',
    ),
    BenchmarkSpec(
        name='ocrbench',
        display_name='OCRBench',
        aliases=('ocrbench', 'ocr_bench'),
        preferred_score_keys=(
            'Final Score Norm',
            'score',
            'acc',
            'accuracy',
            'exact_match',
        ),
        description='Text recognition / OCR capability.',
    ),
    BenchmarkSpec(
        name='rbench_dis',
        display_name='R-Bench-Dis',
        aliases=(
            'rbench_dis',
            'r_bench_dis',
            'r-bench-dis',
            'rbench-dis',
            'rbench_distribution_shift',
        ),
        preferred_score_keys=('acc', 'accuracy', 'score'),
        description='Distribution-shift robustness diagnostic.',
    ),
    BenchmarkSpec(
        name='mathvista',
        display_name='MathVista',
        aliases=('mathvista', 'math_vista', 'mathvista_testmini'),
        preferred_score_keys=('acc', 'accuracy', 'exact_match', 'score'),
        description=(
            'Visual mathematical reasoning; optional reasoning-side '
            'diagnostic.'
        ),
        required=False,
    ),
    BenchmarkSpec(
        name='mmmu',
        display_name='MMMU',
        aliases=('mmmu', 'mmmu_val', 'mmmu_validation'),
        preferred_score_keys=('acc', 'accuracy', 'exact_match', 'score'),
        description=(
            'General multimodal university-level reasoning; optional '
            'diagnostic.'
        ),
        required=False,
    ),
    BenchmarkSpec(
        name='mathverse',
        display_name='MathVerse',
        aliases=('mathverse', 'math_verse'),
        preferred_score_keys=('acc', 'accuracy', 'exact_match', 'score'),
        description='Multimodal math reasoning; optional diagnostic.',
        required=False,
    ),
    BenchmarkSpec(
        name='mmmu_pro',
        display_name='MMMU-Pro',
        aliases=('mmmu_pro', 'mmmu-pro'),
        preferred_score_keys=('acc', 'accuracy', 'exact_match', 'score'),
        description='Harder MMMU-style reasoning; optional diagnostic.',
        required=False,
    ),
    BenchmarkSpec(
        name='lisa',
        display_name='LISA',
        aliases=('lisa',),
        preferred_score_keys=('iou', 'acc', 'accuracy', 'score'),
        description=(
            'Segmentation/perception diagnostic from the larger mixed '
            'setup.'
        ),
        required=False,
    ),
)

CORE_BENCHMARKS: tuple[str, ...] = tuple(
    spec.name for spec in BENCHMARK_SPECS if spec.required
)

ALL_BENCHMARKS: tuple[str, ...] = tuple(spec.name for spec in BENCHMARK_SPECS)

VLMEVAL_DATASETS: dict[str, str] = {
    'aokvqa': 'A-OKVQA',
    'aesbench': 'AesBench_TEST',
    'vstar': 'VStarBench',
    'visonly': 'VisOnlyQA-VLMEvalKit',
    'ocrbench': 'OCRBench',
    'rbench_dis': 'R-Bench-Dis',
}

VLMEVAL_DATASET_ALIASES: dict[str, str] = {
    dataset.lower().replace('-', '_'): benchmark
    for benchmark, dataset in VLMEVAL_DATASETS.items()
}

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
        'A-OKVQA': {'acc': 0.421},
        'AesBench_TEST': {'accuracy': 0.573},
        'VStarBench': {'acc': 0.732},
        'VisOnlyQA-VLMEvalKit': {'acc': 0.614},
        'OCRBench': {'score': 0.48},
        'R-Bench-Dis': {'accuracy': 0.39},
    },
}


def suite_spec_document() -> dict[str, Any]:
    """Return the live VLM general-capability suite specification."""

    return {
        'schema_version': SUITE_SPEC_VERSION,
        'reference': REFERENCE,
        'score_scale': (
            'Input scores must be ratios in [0, 1] or explicit percent '
            'strings such as "65%"; output scores are ratios in [0, 1].'
        ),
        'required_benchmarks': list(CORE_BENCHMARKS),
        'optional_benchmarks': [
            spec.name for spec in BENCHMARK_SPECS if not spec.required
        ],
        'default_evaluator_settings': {
            'model_kwargs': DEFAULT_MODEL_KWARGS,
            'eval_batch_size': DEFAULT_EVAL_BATCH_SIZE,
            'judge_worker_num': DEFAULT_JUDGE_WORKER_NUM,
            'dtype_policy': (
                'not forced by default; explicit dtype overrides are runtime '
                'context and must be recorded with the result'
            ),
        },
        'input_formats': [
            'checkpoint/model path via --checkpoint, evaluated by EvalScope '
            '+ VLMEvalKit',
            'evaluator output directory via --collect-output-dir',
            'structured evaluator JSON with results, groups, or benchmarks '
            'objects',
        ],
        'benchmark_specs': [spec.__dict__ for spec in BENCHMARK_SPECS],
        'output_fields': {
            'benchmarks': 'canonical benchmark -> normalized score',
            'suite_average': (
                'mean over required VLM benchmarks present in the inputs'
            ),
            'diagnostic_average': (
                'mean over all present required and optional VLM scores'
            ),
            'missing_benchmarks': 'required VLM benchmarks absent from inputs',
            'complete': (
                'true only when all required VLM benchmarks are present'
            ),
            'sources': 'canonical benchmark -> score JSON path',
            'benchmark_details': (
                'selected raw name, metric key, required flag, and score'
            ),
            'run': (
                'present in --checkpoint mode; benchmark commands, logs, '
                'statuses, and evaluator settings'
            ),
        },
        'example_input': EXAMPLE_INPUT,
    }
