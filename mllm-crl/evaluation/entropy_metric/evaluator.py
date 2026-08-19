"""Generated-token entropy evaluation on fixed prompt sets."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import transformers
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from verl.utils.torch_functional import (
    entropy_from_logits,
    logprobs_from_logits,
)

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_BATCH_SIZE = 1
DEFAULT_SCORE_BATCH_SIZE = 0
DEFAULT_DTYPE = "bfloat16"
GENERATION_PADDING_SIDE = "left"
EXP_LIMIT = 50.0
SOURCE_FORMAT = "generated_token_entropy_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dtype_from_name(name: str):
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def read_model_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing model config: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def is_qwen3_vl_config(config: dict[str, Any]) -> bool:
    model_type = str(config.get("model_type", "")).lower()
    architectures = [
        str(item).lower() for item in config.get("architectures", [])
    ]
    return model_type == "qwen3_vl" or any(
        "qwen3vl" in item or "qwen3_vl" in item for item in architectures
    )


def build_vlm_transformers_config(model_path: Path):
    config = read_model_config(model_path)
    model_type = str(config.get("model_type", "")).lower()
    if is_qwen3_vl_config(config):
        return None
    if model_type in {"qwen2_5_vl", "qwen2-vl"}:
        patched_config = dict(config)
        patched_config.pop("text_config", None)
        config_module = importlib.import_module(
            "transformers.models.qwen2_5_vl.configuration_qwen2_5_vl"
        )
        return config_module.Qwen2_5_VLConfig.from_dict(patched_config)
    return None


def resolve_vlm_model_class(model_path: Path):
    config = read_model_config(model_path)
    if is_qwen3_vl_config(config):
        model_cls = getattr(
            transformers, "Qwen3VLForConditionalGeneration", None
        )
        if model_cls is not None:
            return model_cls
        raise RuntimeError(
            "Qwen3-VL model detected, but this transformers environment has "
            "no Qwen3VLForConditionalGeneration loader. Use the "
            "evalscope-vlmeval env."
        )
    model_type = str(config.get("model_type", "")).lower()
    if model_type in {"qwen2_5_vl", "qwen2-vl"}:
        return Qwen2_5_VLForConditionalGeneration
    raise ValueError(f"Unsupported VLM model_type: {model_type}")


def load_model_and_processor(
    model_path: Path, modality: str, dtype_name: str, device_map: str | None
):
    dtype = dtype_from_name(dtype_name)
    common_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if device_map:
        common_kwargs["device_map"] = device_map
    if modality == "llm":
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, **common_kwargs
        )
        if not device_map:
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = GENERATION_PADDING_SIDE
        return model, tokenizer
    if modality == "vlm":
        model_cls = resolve_vlm_model_class(model_path)
        transformers_config = build_vlm_transformers_config(model_path)
        if transformers_config is not None:
            common_kwargs["config"] = transformers_config
        model = model_cls.from_pretrained(model_path, **common_kwargs)
        if not device_map:
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        if hasattr(processor, "tokenizer"):
            if (
                processor.tokenizer.pad_token_id is None
                and processor.tokenizer.eos_token_id is not None
            ):
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            processor.tokenizer.padding_side = GENERATION_PADDING_SIDE
        return model, processor
    raise ValueError(f"Unsupported modality: {modality}")


def model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def entropy_and_nll_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    chunk_size: int = 16,
) -> tuple[float, float]:
    if logits.shape[0] != target_ids.shape[0]:
        raise ValueError(
            "logits/target length mismatch: "
            f"{logits.shape[0]} vs {target_ids.shape[0]}"
        )
    entropy_sum = 0.0
    nll_sum = 0.0
    for start in range(0, int(target_ids.numel()), chunk_size):
        end = min(start + chunk_size, int(target_ids.numel()))
        chunk_logits = logits[start:end].float()
        chunk_targets = target_ids[start:end].to(chunk_logits.device).long()
        entropy_values = entropy_from_logits(chunk_logits)
        selected_log_probs = logprobs_from_logits(chunk_logits, chunk_targets)
        entropy_sum += float(entropy_values.sum().item())
        nll_sum += float((-selected_log_probs).sum().item())
    return entropy_sum, nll_sum


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(
        exc, torch.OutOfMemoryError
    ) or "CUDA out of memory" in str(exc)


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def build_llm_inputs(records: list[dict[str, Any]], tokenizer):
    texts: list[str] = []
    for record in records:
        messages = record.get("messages")
        if messages and hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = record["text_prompt"]
        texts.append(text)
    return tokenizer(texts, return_tensors="pt", padding=True)


def build_vlm_inputs(records: list[dict[str, Any]], processor):
    texts: list[str] = []
    batch_messages: list[list[dict[str, Any]]] = []
    for record in records:
        messages = record.get("messages")
        if not messages:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": record["image_abspath"]},
                        {"type": "text", "text": record["question"]},
                    ],
                }
            ]
        batch_messages.append(messages)
        texts.append(
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    image_inputs, video_inputs = process_vision_info(batch_messages)
    return processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def move_batch_to_device(batch, device: torch.device):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def build_forward_inputs(
    inputs: dict[str, Any], generated: torch.Tensor
) -> dict[str, Any]:
    forward_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if key in {"input_ids", "attention_mask", "position_ids"}:
            continue
        forward_inputs[key] = value
    forward_inputs["input_ids"] = generated
    prompt_attention_mask = inputs.get("attention_mask")
    if prompt_attention_mask is None:
        forward_inputs["attention_mask"] = torch.ones_like(generated)
    else:
        extra_width = int(generated.shape[1] - prompt_attention_mask.shape[1])
        if extra_width < 0:
            raise ValueError(
                "generated sequence is shorter than prompt attention mask"
            )
        response_attention_mask = torch.ones(
            (generated.shape[0], extra_width),
            dtype=prompt_attention_mask.dtype,
            device=prompt_attention_mask.device,
        )
        forward_inputs["attention_mask"] = torch.cat(
            [prompt_attention_mask, response_attention_mask],
            dim=1,
        )
    return forward_inputs


def response_eos_token_id(processor) -> int | None:
    eos_token_id = getattr(processor, "eos_token_id", None)
    if eos_token_id is None and hasattr(processor, "tokenizer"):
        eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)
    return eos_token_id


def trim_response_ids(
    generated_ids: torch.Tensor, eos_token_id: int | None
) -> torch.Tensor:
    if generated_ids.numel() > 0 and eos_token_id is not None:
        eos_positions = (generated_ids == eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            generated_ids = generated_ids[: int(eos_positions[0].item()) + 1]
    return generated_ids.detach().cpu()


def pad_response_ids(
    response_ids: list[torch.Tensor], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    max_width = max((int(item.numel()) for item in response_ids), default=0)
    if max_width <= 0:
        raise ValueError("cannot pad empty response-id batch")
    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for item in response_ids:
        width = int(item.numel())
        if width < max_width:
            padding = torch.full(
                (max_width - width,), pad_token_id, dtype=item.dtype
            )
            row = torch.cat([item, padding], dim=0)
            mask = torch.cat(
                [
                    torch.ones(width, dtype=torch.long),
                    torch.zeros(max_width - width, dtype=torch.long),
                ],
                dim=0,
            )
        else:
            row = item
            mask = torch.ones(width, dtype=torch.long)
        rows.append(row)
        masks.append(mask)
    return torch.stack(rows, dim=0), torch.stack(masks, dim=0)


def pad_token_id_for_processor(processor) -> int:
    pad_token_id = getattr(processor, "pad_token_id", None)
    if pad_token_id is None and hasattr(processor, "tokenizer"):
        pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_token_id = response_eos_token_id(processor)
        if eos_token_id is not None:
            return int(eos_token_id)
    if pad_token_id is None:
        return 0
    return int(pad_token_id)


def build_scoring_forward_inputs(
    inputs: dict[str, Any],
    response_ids: list[torch.Tensor],
    *,
    pad_token_id: int,
) -> tuple[dict[str, Any], int, torch.Tensor]:
    input_ids = inputs["input_ids"]
    prompt_width = int(input_ids.shape[1])
    padded_responses, response_mask = pad_response_ids(
        response_ids, pad_token_id
    )
    padded_responses = padded_responses.to(input_ids.device)
    response_mask = response_mask.to(input_ids.device)

    generated = torch.cat([input_ids, padded_responses], dim=1)
    forward_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if key in {"input_ids", "attention_mask", "position_ids"}:
            continue
        forward_inputs[key] = value
    forward_inputs["input_ids"] = generated
    prompt_attention_mask = inputs.get("attention_mask")
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(input_ids)
    forward_inputs["attention_mask"] = torch.cat(
        [prompt_attention_mask, response_mask.to(prompt_attention_mask.dtype)],
        dim=1,
    )
    return forward_inputs, prompt_width, response_mask


def score_generated_responses(
    *,
    records: list[dict[str, Any]],
    response_ids: list[torch.Tensor],
    model,
    processor,
    modality: str,
    score_batch_size: int,
) -> list[dict[str, Any]]:
    if len(records) != len(response_ids):
        raise ValueError("records/response_ids length mismatch")
    pad_token_id = pad_token_id_for_processor(processor)
    scored: list[dict[str, Any]] = []
    for start in range(0, len(records), score_batch_size):
        end = min(start + score_batch_size, len(records))
        chunk_records = records[start:end]
        chunk_response_ids = response_ids[start:end]
        if modality == "llm":
            inputs = build_llm_inputs(chunk_records, processor)
        else:
            inputs = build_vlm_inputs(chunk_records, processor)
        inputs = move_batch_to_device(inputs, model_device(model))
        forward_inputs, prompt_width, _response_mask = (
            build_scoring_forward_inputs(
                inputs,
                chunk_response_ids,
                pad_token_id=pad_token_id,
            )
        )
        with torch.inference_mode():
            outputs = model(**forward_inputs)
        for batch_index, generated_ids in enumerate(chunk_response_ids):
            response_length = int(generated_ids.numel())
            if response_length > 0:
                # logits position t predicts token t+1. Response tokens that
                # start at prompt_width are predicted by logits from
                # prompt_width - 1 through prompt_width + response_length - 1.
                start_pos = max(prompt_width - 1, 0)
                end_pos = start_pos + response_length
                response_logits = outputs.logits[
                    batch_index, start_pos:end_pos, :
                ]
                target_ids = generated_ids.to(response_logits.device).long()
                entropy_sum, nll_sum = entropy_and_nll_from_logits(
                    response_logits, target_ids
                )
                entropy_mean = entropy_sum / response_length
                nll_mean = nll_sum / response_length
            else:
                entropy_sum = 0.0
                entropy_mean = None
                nll_sum = 0.0
                nll_mean = None
            scored.append(
                {
                    "generated_token_entropy_mean": entropy_mean,
                    "generated_token_entropy_sum": entropy_sum,
                    "generated_token_nll_mean": nll_mean,
                    "generated_token_nll_sum": nll_sum,
                    "generated_token_ppl": (
                        math.exp(nll_mean)
                        if nll_mean is not None and nll_mean < EXP_LIMIT
                        else None
                    ),
                }
            )
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scored


def evaluate_batch(
    records: list[dict[str, Any]],
    model,
    processor,
    modality: str,
    max_new_tokens: int,
    score_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if modality == "llm":
        inputs = build_llm_inputs(records, processor)
    else:
        inputs = build_vlm_inputs(records, processor)
    inputs = move_batch_to_device(inputs, model_device(model))
    input_ids = inputs["input_ids"]
    prompt_width = int(input_ids.shape[1])

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=False,
        )

    batch_results: list[dict[str, Any]] = []
    total_entropy_sum = 0.0
    total_nll_sum = 0.0
    total_tokens = 0
    clipped = 0
    eos_token_id = response_eos_token_id(processor)
    response_ids_by_record: list[torch.Tensor] = []
    for batch_index, record in enumerate(records):
        generated_ids = trim_response_ids(
            generated[batch_index, prompt_width:], eos_token_id
        )
        response_ids_by_record.append(generated_ids)
        response_length = int(generated_ids.numel())
        if response_length >= max_new_tokens:
            clipped += 1
        total_tokens += response_length
        decoder = getattr(processor, "decode", None)
        if decoder is None and hasattr(processor, "tokenizer"):
            decoder = processor.tokenizer.decode
        text = (
            decoder(generated_ids, skip_special_tokens=True)
            if response_length
            else ""
        )
        batch_results.append(
            {
                "prompt_id": record.get("prompt_id"),
                "domain": record.get("domain"),
                "task": record.get("task"),
                "subtype": record.get("subtype"),
                "generation_batch_size_used": len(records),
                "score_batch_size": score_batch_size,
                "response_length": response_length,
                "clipped": response_length >= max_new_tokens,
                "generated_text": text,
            }
        )
    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scored = score_generated_responses(
        records=records,
        response_ids=response_ids_by_record,
        model=model,
        processor=processor,
        modality=modality,
        score_batch_size=score_batch_size,
    )
    for result, score in zip(batch_results, scored, strict=True):
        result.update(score)
        total_entropy_sum += float(score["generated_token_entropy_sum"])
        total_nll_sum += float(score["generated_token_nll_sum"])
    return batch_results, {
        "entropy_sum": total_entropy_sum,
        "nll_sum": total_nll_sum,
        "tokens": total_tokens,
        "clipped": clipped,
    }


def merge_totals(
    first: dict[str, float], second: dict[str, float]
) -> dict[str, float]:
    return {
        "entropy_sum": float(first["entropy_sum"])
        + float(second["entropy_sum"]),
        "nll_sum": float(first["nll_sum"]) + float(second["nll_sum"]),
        "tokens": int(first["tokens"]) + int(second["tokens"]),
        "clipped": int(first["clipped"]) + int(second["clipped"]),
    }


def evaluate_batch_resilient(
    records: list[dict[str, Any]],
    model,
    processor,
    modality: str,
    max_new_tokens: int,
    score_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    try:
        return evaluate_batch(
            records,
            model,
            processor,
            modality,
            max_new_tokens,
            min(score_batch_size, len(records)),
        )
    except RuntimeError as exc:
        if not is_cuda_oom(exc) or len(records) <= 1:
            raise
        clear_cuda_cache()
        midpoint = len(records) // 2
        left_results, left_totals = evaluate_batch_resilient(
            records[:midpoint],
            model,
            processor,
            modality,
            max_new_tokens,
            score_batch_size,
        )
        right_results, right_totals = evaluate_batch_resilient(
            records[midpoint:],
            model,
            processor,
            modality,
            max_new_tokens,
            score_batch_size,
        )
        return left_results + right_results, merge_totals(
            left_totals, right_totals
        )


def chunks(records: list[dict[str, Any]], size: int):
    for start in range(0, len(records), size):
        yield records[start : start + size]


def build_summary(
    *,
    prompt_path: Path,
    output_path: Path,
    model_path: Path,
    modality: str,
    max_new_tokens: int,
    batch_size: int,
    score_batch_size: int,
    dtype: str,
    results: list[dict[str, Any]],
    totals: dict[str, float],
) -> dict[str, Any]:
    lengths = [int(item["response_length"]) for item in results]
    effective_batch_sizes = sorted(
        {
            int(item.get("generation_batch_size_used", batch_size))
            for item in results
        }
    )
    entropy_values = [
        item["generated_token_entropy_mean"]
        for item in results
        if item["generated_token_entropy_mean"] is not None
    ]
    nll_values = [
        item["generated_token_nll_mean"]
        for item in results
        if item["generated_token_nll_mean"] is not None
    ]
    token_count = int(totals["tokens"])
    entropy_mean = totals["entropy_sum"] / token_count if token_count else None
    nll_mean = totals["nll_sum"] / token_count if token_count else None
    generated_ppl = (
        math.exp(nll_mean)
        if nll_mean is not None and nll_mean < EXP_LIMIT
        else None
    )
    return {
        "schema_version": "entropy-metric-summary-v1",
        "source_format": SOURCE_FORMAT,
        "prompt_set": str(prompt_path),
        "prompt_set_sha256": file_sha256(prompt_path),
        "output_jsonl": str(output_path),
        "model_path": str(model_path),
        "modality": modality,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "score_batch_size": score_batch_size,
        "effective_generation_batch_sizes": effective_batch_sizes,
        "oom_fallback_prompt_count": sum(
            1
            for item in results
            if int(item.get("generation_batch_size_used", batch_size))
            < batch_size
        ),
        "dtype": dtype,
        "generation_padding_side": GENERATION_PADDING_SIDE,
        "num_prompts": len(results),
        "num_response_tokens": token_count,
        "generated_token_entropy_mean": entropy_mean,
        "generated_token_nll_mean": nll_mean,
        "generated_token_ppl": generated_ppl,
        "per_prompt_entropy_mean_macro": (
            sum(entropy_values) / len(entropy_values)
        )
        if entropy_values
        else None,
        "per_prompt_nll_mean_macro": (sum(nll_values) / len(nll_values))
        if nll_values
        else None,
        "response_length_mean": (sum(lengths) / len(lengths))
        if lengths
        else None,
        "response_length_max": max(lengths) if lengths else None,
        "clip_ratio_at_max_new_tokens": (int(totals["clipped"]) / len(results))
        if results
        else None,
        "formula_source": "verl.utils.torch_functional.entropy_from_logits",
        "selected_logprob_source": (
            "verl.utils.torch_functional.logprobs_from_logits"
        ),
        "notes": [
            "Entropy is computed from model logits over greedily generated "
            "response tokens on the fixed prompt set.",
            "Entropy-from-logits is delegated directly to VERL.",
            "generated_token_ppl is exp(selected-token NLL) over the "
            "same greedily generated response tokens.",
        ],
    }


def evaluate_entropy(
    *,
    prompt_set: Path,
    model_path: Path,
    modality: str,
    output_jsonl: Path,
    summary_json: Path,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    score_batch_size: int = DEFAULT_SCORE_BATCH_SIZE,
    dtype: str = DEFAULT_DTYPE,
    device_map: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    records = read_jsonl(prompt_set)
    if limit is not None:
        records = records[:limit]
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    effective_score_batch_size = score_batch_size or batch_size
    if effective_score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")
    if not records:
        raise ValueError("prompt set is empty")

    model, processor = load_model_and_processor(
        model_path, modality, dtype, device_map
    )
    results: list[dict[str, Any]] = []
    totals = {
        "entropy_sum": 0.0,
        "nll_sum": 0.0,
        "tokens": 0,
        "clipped": 0,
    }
    batches = list(chunks(records, batch_size))
    for batch in tqdm(batches, desc="entropy-eval"):
        batch_results, batch_totals = evaluate_batch_resilient(
            batch,
            model,
            processor,
            modality,
            max_new_tokens,
            effective_score_batch_size,
        )
        results.extend(batch_results)
        totals["entropy_sum"] += batch_totals["entropy_sum"]
        totals["nll_sum"] += batch_totals["nll_sum"]
        totals["tokens"] += batch_totals["tokens"]
        totals["clipped"] += batch_totals["clipped"]

    write_jsonl(output_jsonl, results)
    summary = build_summary(
        prompt_path=prompt_set,
        output_path=output_jsonl,
        model_path=model_path,
        modality=modality,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        score_batch_size=effective_score_batch_size,
        dtype=dtype,
        results=results,
        totals=totals,
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
