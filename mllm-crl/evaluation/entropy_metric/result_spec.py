"""Specification for policy-entropy evaluation documents."""

from __future__ import annotations

from typing import Any

INPUT_SCHEMA_VERSION = "entropy-metric-input-v1"
RESULT_SCHEMA_VERSION = "entropy-metric-result-v1"
SUMMARY_SCHEMA_VERSION = "entropy-metric-summary-v1"
DEFAULT_SOURCE_FORMAT = "generated_token_entropy_v1"

EXAMPLE_INPUT: dict[str, Any] = {
    "schema_version": INPUT_SCHEMA_VERSION,
    "suite_id": "llm_algorithmic_vanilla_crl",
    "source_format": DEFAULT_SOURCE_FORMAT,
    "entropy_eval": {
        "prompt_set": "/path/to/llm_algorithmic_val64.jsonl",
        "model_path": "/path/to/global_step_500_hf_merged",
        "modality": "llm",
        "max_new_tokens": 1024,
        "decode": "greedy",
    },
    "metadata": {
        "note": "Formal entropy result uses fixed-prompt checkpoint eval.",
    },
}


def result_spec_document() -> dict[str, Any]:
    """Return the entropy metric input/output contract."""

    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "purpose": (
            "Evaluate generated-token policy entropy for benchmark tables "
            "and paper analysis: final checkpoint, fixed prompt set, greedy "
            "generation, and VERL entropy_from_logits over generated "
            "response-token logits."
        ),
        "metric_definition": {
            "primary_metric": "generated_token_entropy_mean",
            "token_entropy": (
                "For each generated response token t, H_t = "
                "logsumexp(z_t) - sum_v softmax(z_t)_v * z_t,v."
            ),
            "formula_source": (
                "evaluation.entropy_metric.evaluator imports "
                "verl.utils.torch_functional.entropy_from_logits directly; "
                "benchmark-side code must not keep a separate production "
                "entropy formula mirror."
            ),
            "unit": "nats, because VERL uses natural logarithms",
            "aggregation": (
                "token-weighted mean over all generated response tokens on "
                "the fixed prompt set"
            ),
            "decode": (
                "greedy, do_sample=false, max_new_tokens=1024 by default"
            ),
            "secondary_metric": (
                "generated_token_ppl = exp(selected-token NLL) over the "
                "same generated response tokens; this is not exp(entropy)"
            ),
            "not_this": (
                "This is not answer-string entropy, sample diversity, "
                "or multiple-choice label entropy."
            ),
        },
        "accepted_inputs": {
            "evaluate": {
                "prompt_set": "fixed JSONL prompt set with recorded sha256",
                "model_path": "final checkpoint / HF merged model path",
                "modality": "llm or vlm",
                "output_jsonl": "per-prompt generated text and token metrics",
                "summary_json": "table-facing entropy summary",
            },
        },
        "output_fields": {
            "generated_token_entropy_mean": (
                "token-weighted mean entropy over generated response tokens"
            ),
            "generated_token_nll_mean": (
                "token-weighted selected-token NLL over the same tokens"
            ),
            "generated_token_ppl": "exp(generated_token_nll_mean)",
            "prompt_set_sha256": "fixed-prompt identity for comparability",
            "num_response_tokens": "aggregation denominator",
        },
        "example_input": EXAMPLE_INPUT,
    }
