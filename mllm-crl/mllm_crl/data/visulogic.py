from __future__ import annotations

import importlib
import json
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ParquetWriter = Callable[[Sequence[dict[str, Any]], Path], None]
TAG_FIELDS = ("tag", "category", "task")
ANSWER_FIELDS = ("answer", "label")
DEFAULT_SUBTYPE_FIELDS = (
    "subcategory",
    "quantity_subtype",
    "spatial_subtype",
    "positional_subtype",
)


@dataclass(frozen=True)
class VisuLogicTaggedTrainResult:
    input_jsonl: Path
    out_jsonl: Path
    manifest_path: Path
    total_count: int
    tagged_count: int
    missing_count: int
    tag_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_jsonl": str(self.input_jsonl),
            "out_jsonl": str(self.out_jsonl),
            "manifest_path": str(self.manifest_path),
            "total_count": self.total_count,
            "tagged_count": self.tagged_count,
            "missing_count": self.missing_count,
            "tag_counts": dict(self.tag_counts),
        }


@dataclass(frozen=True)
class VisuLogicVerlPrepareResult:
    out_dir: Path
    train_path: Path
    val_path: Path
    manifest_path: Path
    train_count: int
    val_count: int
    total_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "out_dir": str(self.out_dir),
            "train_path": str(self.train_path),
            "val_path": str(self.val_path),
            "manifest_path": str(self.manifest_path),
            "train_count": self.train_count,
            "val_count": self.val_count,
            "total_count": self.total_count,
        }


def convert_visulogic_record_to_verl_row(
    record: dict[str, Any],
    *,
    image_root: Path,
    image_uri_root: str | None = None,
    row_index: int = 0,
    split: str = "train",
    strict_images: bool = True,
    source_path: Path | None = None,
    subtype_field: str | None = None,
) -> dict[str, Any]:
    image_refs, text_parts = extract_visulogic_message_parts(record)
    if not image_refs:
        raise ValueError(f"VisuLogic row {row_index} has no image reference")
    question = str(record.get("question") or "").strip()
    prompt_text = "\n".join(part for part in text_parts if part.strip())
    if not prompt_text:
        prompt_text = question
    if not prompt_text:
        raise ValueError(f"VisuLogic row {row_index} has no prompt text")

    image_payloads = [
        {"image": build_image_uri(ref, image_root, image_uri_root)}
        for ref in image_refs
    ]
    if strict_images:
        missing = [
            ref
            for ref in image_refs
            if not resolve_image_file(ref, image_root).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"VisuLogic row {row_index} has missing image files: {missing}"
            )

    answer = extract_visulogic_answer(record, row_index=row_index)
    record_id = str(record.get("id") or row_index)
    tag = extract_visulogic_tag(record)
    task_id = slugify_tag(tag) if tag else "visulogic"
    extra_info = {
        "id": record_id,
        "index": row_index,
        "split": split,
        "answer": answer,
        "question": question,
        "task_family": "visulogic",
        "task_id": task_id,
        "image_refs": image_refs,
        "image_uris": [item["image"] for item in image_payloads],
        "source_path": str(source_path) if source_path is not None else "",
    }
    if tag:
        extra_info["tag"] = tag
    subtype = extract_visulogic_subtype(record, subtype_field=subtype_field)
    if subtype:
        extra_info["subcategory"] = subtype

    return {
        "data_source": "visulogic",
        "prompt": [
            {
                "role": "user",
                "content": "<image>\n" * len(image_refs) + prompt_text,
            }
        ],
        "images": image_payloads,
        "ability": "vlm_logic",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": extra_info,
    }


def append_visulogic_message_content(
    content: Any,
    *,
    image_refs: list[str],
    text_parts: list[str],
) -> None:
    if isinstance(content, str):
        text_parts.append(content)
        return
    if not isinstance(content, list):
        raise ValueError("VisuLogic message content must be string or list")
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("VisuLogic content item must be an object")
        part_type = part.get("type")
        if part_type == "image":
            image = part.get("image")
            if not image:
                raise ValueError("VisuLogic image content is missing image")
            image_refs.append(str(image))
        elif part_type == "text":
            text = part.get("text")
            if text:
                text_parts.append(str(text))


def extract_visulogic_message_parts(
    record: dict[str, Any],
) -> tuple[list[str], list[str]]:
    raw_message = record.get("message")
    if raw_message is None:
        image_path = record.get("image_path") or record.get("image")
        if not image_path:
            raise ValueError("VisuLogic record is missing message/image_path")
        text = str(record.get("question") or "").strip()
        return [str(image_path)], [text] if text else []

    messages = (
        json.loads(raw_message)
        if isinstance(raw_message, str)
        else raw_message
    )
    if not isinstance(messages, list):
        raise ValueError("VisuLogic message must be a JSON list")

    image_refs: list[str] = []
    text_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("VisuLogic message item must be an object")
        append_visulogic_message_content(
            message.get("content", []),
            image_refs=image_refs,
            text_parts=text_parts,
        )
    return image_refs, text_parts


def extract_visulogic_answer(record: dict[str, Any], *, row_index: int) -> str:
    answer = ""
    for field in ANSWER_FIELDS:
        value = record.get(field)
        if value is not None:
            answer = str(value).strip().upper()
            break
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(
            f"VisuLogic row {row_index} has invalid answer/label: {answer!r}"
        )
    return answer


def extract_visulogic_tag(record: dict[str, Any]) -> str | None:
    for field in TAG_FIELDS:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def extract_visulogic_subtype(
    record: dict[str, Any],
    *,
    subtype_field: str | None = None,
) -> str | None:
    fields = (
        (subtype_field,)
        if subtype_field
        else DEFAULT_SUBTYPE_FIELDS
    )
    for field in fields:
        if field is None:
            continue
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def slugify_tag(tag: str) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() else "_" for char in tag.strip()
    )
    return "_".join(part for part in normalized.split("_") if part)


def resolve_image_file(image_ref: str, image_root: Path) -> Path:
    if image_ref.startswith(("file://", "http://", "https://")):
        raise ValueError(
            "strict local image checks require relative or absolute paths, "
            f"got {image_ref!r}"
        )
    image_path = Path(image_ref)
    if image_path.is_absolute():
        return image_path
    relative = strip_image_root_prefix(image_ref, image_root)
    return image_root / Path(relative)


def build_image_uri(
    image_ref: str,
    image_root: Path,
    image_uri_root: str | None = None,
) -> str:
    if image_ref.startswith(("file://", "http://", "https://")):
        return image_ref
    if image_uri_root:
        relative = strip_image_root_prefix(image_ref, image_root)
        return join_uri(image_uri_root, relative)
    image_file = resolve_image_file(image_ref, image_root)
    if image_file.is_absolute():
        return image_file.as_uri()
    return image_file.resolve().as_uri()


def strip_image_root_prefix(image_ref: str, image_root: Path) -> str:
    path = PurePosixPath(image_ref)
    parts = path.parts
    if parts and parts[0] == image_root.name:
        return str(PurePosixPath(*parts[1:]))
    return str(path)


def join_uri(root: str, relative_path: str) -> str:
    return f"{root.rstrip('/')}/{relative_path.lstrip('/')}"


def iter_visulogic_verl_rows(
    input_jsonl: Path,
    *,
    image_root: Path,
    image_uri_root: str | None = None,
    strict_images: bool = True,
    max_rows: int | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    require_subcategory: bool = False,
    subtype_field: str | None = None,
) -> Iterable[dict[str, Any]]:
    include_tag_set = normalize_tag_set(include_tags)
    exclude_tag_set = normalize_tag_set(exclude_tags)
    with input_jsonl.open(encoding="utf-8") as handle:
        for row_index, raw_line in enumerate(handle):
            if max_rows is not None and row_index >= max_rows:
                break
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not should_keep_visulogic_record(
                record,
                include_tags=include_tag_set,
                exclude_tags=exclude_tag_set,
            ):
                continue
            if (
                require_subcategory
                and extract_visulogic_subtype(
                    record,
                    subtype_field=subtype_field,
                )
                is None
            ):
                record_id = str(record.get("id") or row_index)
                raise ValueError(
                    "VisuLogic CRL parquet requires subtype labels; "
                    f"row id {record_id} has no subtype field"
                )
            yield convert_visulogic_record_to_verl_row(
                record,
                image_root=image_root,
                image_uri_root=image_uri_root,
                row_index=row_index,
                strict_images=strict_images,
                source_path=input_jsonl,
                subtype_field=subtype_field,
            )


def normalize_tag_set(tags: Sequence[str] | None) -> set[str]:
    if not tags:
        return set()
    return {normalize_tag(tag) for tag in tags if str(tag).strip()}


def normalize_tag(tag: str) -> str:
    return " ".join(str(tag).strip().lower().split())


def should_keep_visulogic_record(
    record: dict[str, Any],
    *,
    include_tags: set[str],
    exclude_tags: set[str],
) -> bool:
    tag = extract_visulogic_tag(record)
    normalized_tag = normalize_tag(tag) if tag else None
    if include_tags and normalized_tag not in include_tags:
        return False
    return not (exclude_tags and normalized_tag in exclude_tags)


def split_verl_rows(
    rows: Sequence[dict[str, Any]],
    *,
    val_ratio: float = 0.02,
    val_size: int | None = None,
    seed: int = 20260430,
    stratify_field: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("cannot split an empty VisuLogic VERL dataset")
    if val_size is None:
        val_count = round(len(rows) * val_ratio)
        if val_ratio > 0 and val_count == 0 and len(rows) > 1:
            val_count = 1
    else:
        val_count = val_size
    val_count = max(0, min(val_count, max(len(rows) - 1, 0)))
    if stratify_field:
        val_indices = stratified_val_indices(
            rows,
            val_count=val_count,
            seed=seed,
            field=stratify_field,
        )
    else:
        indices = list(range(len(rows)))
        random.Random(seed).shuffle(indices)
        val_indices = set(indices[:val_count])
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        target = val_rows if index in val_indices else train_rows
        split = "val" if index in val_indices else "train"
        target.append(with_split(row, split))
    return train_rows, val_rows


def stratified_val_indices(
    rows: Sequence[dict[str, Any]],
    *,
    val_count: int,
    seed: int,
    field: str,
) -> set[int]:
    if val_count <= 0:
        return set()
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        key = str((row.get("extra_info") or {}).get(field) or "")
        if not key:
            raise ValueError(
                f"cannot stratify VisuLogic split: row {index} has no "
                f"extra_info[{field!r}]"
            )
        groups.setdefault(key, []).append(index)

    eligible = [key for key, indices in groups.items() if len(indices) > 1]
    if len(eligible) > val_count:
        raise ValueError(
            "cannot create stratified validation split with at least one "
            f"sample per {field}: val_count={val_count}, "
            f"eligible_groups={len(eligible)}"
        )

    val_indices: set[int] = set()
    for key in sorted(eligible):
        candidates = groups[key].copy()
        rng.shuffle(candidates)
        val_indices.add(candidates[0])

    remaining = [
        index
        for index in range(len(rows))
        if index not in val_indices
        and len(groups[str((rows[index].get("extra_info") or {}).get(field) or "")]) > 1
    ]
    rng.shuffle(remaining)
    for index in remaining:
        if len(val_indices) >= val_count:
            break
        key = str((rows[index].get("extra_info") or {}).get(field) or "")
        group_train_count = sum(
            1 for candidate in groups[key] if candidate not in val_indices
        )
        if group_train_count <= 1:
            continue
        val_indices.add(index)
    return val_indices


def with_split(row: dict[str, Any], split: str) -> dict[str, Any]:
    copied = dict(row)
    extra_info = dict(copied.get("extra_info") or {})
    extra_info["split"] = split
    copied["extra_info"] = extra_info
    return copied


def prepare_visulogic_verl_dataset(
    *,
    input_jsonl: Path,
    image_root: Path,
    out_dir: Path,
    image_uri_root: str | None = None,
    val_ratio: float = 0.02,
    val_size: int | None = None,
    seed: int = 20260430,
    strict_images: bool = True,
    max_rows: int | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    require_subcategory: bool = False,
    subtype_field: str | None = None,
    taxonomy: str = "VLM vanilla-multitask",
    parquet_writer: ParquetWriter | None = None,
) -> VisuLogicVerlPrepareResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(
        iter_visulogic_verl_rows(
            input_jsonl,
            image_root=image_root,
            image_uri_root=image_uri_root,
            strict_images=strict_images,
            max_rows=max_rows,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            require_subcategory=require_subcategory,
            subtype_field=subtype_field,
        )
    )
    if not rows:
        source_tags = collect_visulogic_source_tags(
            input_jsonl, max_rows=max_rows
        )
        raise ValueError(
            "no VisuLogic rows matched the requested tag filter; "
            f"include_tags={list(include_tags or [])}, "
            f"exclude_tags={list(exclude_tags or [])}, "
            f"source_tags={source_tags}"
        )
    train_rows, val_rows = split_verl_rows(
        rows,
        val_ratio=val_ratio,
        val_size=val_size,
        seed=seed,
        stratify_field="subcategory" if require_subcategory else None,
    )
    writer = parquet_writer or write_verl_parquet_rows
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    writer(train_rows, train_path)
    writer(val_rows, val_path)
    manifest_path = out_dir / "manifest.json"
    result = VisuLogicVerlPrepareResult(
        out_dir=out_dir,
        train_path=train_path,
        val_path=val_path,
        manifest_path=manifest_path,
        train_count=len(train_rows),
        val_count=len(val_rows),
        total_count=len(rows),
    )
    manifest = result.to_dict() | {
        "input_jsonl": str(input_jsonl),
        "image_root": str(image_root),
        "image_uri_root": image_uri_root or "",
        "val_ratio": val_ratio,
        "val_size": val_size,
        "seed": seed,
        "strict_images": strict_images,
        "max_rows": max_rows,
        "include_tags": list(include_tags or []),
        "exclude_tags": list(exclude_tags or []),
        "tag_counts": count_verl_rows_by_tag(rows),
        "require_subcategory": require_subcategory,
        "subtype_field": subtype_field or "",
        "subcategory_counts": count_verl_rows_by_subcategory(rows),
        "data_source": "visulogic",
        "framework": "verl",
        "taxonomy": taxonomy,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def collect_visulogic_source_tags(
    input_jsonl: Path,
    *,
    max_rows: int | None = None,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    seen_untagged = 0
    with input_jsonl.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if max_rows is not None and row_index >= max_rows:
                break
            if not line.strip():
                continue
            tag = extract_visulogic_tag(json.loads(line))
            if tag:
                counter[tag] += 1
            else:
                seen_untagged += 1
    result = dict(counter)
    if seen_untagged:
        result["<untagged>"] = seen_untagged
    return result


def count_verl_rows_by_tag(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        tag = (row.get("extra_info") or {}).get("tag")
        counter[str(tag) if tag else "<untagged>"] += 1
    return dict(counter)


def count_verl_rows_by_subcategory(
    rows: Sequence[dict[str, Any]]
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        subcategory = (row.get("extra_info") or {}).get("subcategory")
        counter[str(subcategory) if subcategory else "<unlabeled>"] += 1
    return dict(counter)


def load_jsonl_record_map(
    path: Path,
    *,
    id_field: str = "id",
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for row_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL row {row_index} is not an object")
            record_id = record.get(id_field)
            if record_id is None or not str(record_id).strip():
                raise ValueError(
                    f"JSONL row {row_index} has no id field {id_field!r}"
                )
            record_id_text = str(record_id)
            if record_id_text in records:
                raise ValueError(f"duplicate VisuLogic id {record_id_text}")
            records[record_id_text] = record
    return records


def materialize_subtyped_visulogic_jsonl(
    *,
    source_jsonl: Path,
    subtype_jsonl: Path,
    out_jsonl: Path,
    manifest_path: Path | None = None,
    id_field: str = "id",
    subtype_field: str | None = None,
    force: bool = False,
) -> dict[str, object]:
    if out_jsonl.exists() and not force:
        raise FileExistsError(
            f"output already exists; pass force=True: {out_jsonl}"
        )
    resolved_manifest_path = manifest_path or out_jsonl.with_name(
        f"{out_jsonl.stem}.manifest.json"
    )
    if resolved_manifest_path.exists() and not force:
        raise FileExistsError(
            "manifest already exists; pass force=True: "
            f"{resolved_manifest_path}"
        )
    source_records = load_jsonl_record_map(source_jsonl, id_field=id_field)
    subtype_records = load_jsonl_record_map(subtype_jsonl, id_field=id_field)
    output_records: list[dict[str, Any]] = []
    tag_counter: Counter[str] = Counter()
    subtype_counter: Counter[str] = Counter()
    missing_ids: list[str] = []

    for record_id, subtype_record in subtype_records.items():
        source_record = source_records.get(record_id)
        if source_record is None:
            missing_ids.append(record_id)
            continue
        subtype = extract_visulogic_subtype(
            subtype_record,
            subtype_field=subtype_field,
        )
        if subtype is None:
            missing_ids.append(record_id)
            continue
        copied = dict(source_record)
        tag = extract_visulogic_tag(copied) or subtype_record.get(
            "primary_tag"
        )
        if tag:
            copied["tag"] = str(tag).strip()
            tag_counter[copied["tag"]] += 1
        copied["subcategory"] = subtype
        subtype_counter[subtype] += 1
        output_records.append(copied)

    if missing_ids:
        sample = ", ".join(missing_ids[:10])
        raise ValueError(
            f"{len(missing_ids)} source rows missing subtype labels; "
            f"sample ids: {sample}"
        )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    result = {
        "schema": "visulogic_subtyped_train_manifest_v1",
        "source_jsonl": str(source_jsonl),
        "subtype_jsonl": str(subtype_jsonl),
        "out_jsonl": str(out_jsonl),
        "manifest_path": str(resolved_manifest_path),
        "id_field": id_field,
        "subtype_field": subtype_field or "",
        "total_count": len(output_records),
        "tag_counts": dict(tag_counter),
        "subcategory_counts": dict(subtype_counter),
    }
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def load_jsonl_records(path: Path) -> list[object]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_tag_map_payload(
    payload: object,
) -> dict[str, str] | list[object]:
    if isinstance(payload, dict):
        if all(not isinstance(value, dict) for value in payload.values()):
            return {
                str(key): normalize_materialized_tag(str(value))
                for key, value in payload.items()
                if str(value).strip()
            }
        return list(payload.values())
    if isinstance(payload, list):
        return payload
    raise ValueError("VisuLogic tag map JSON must be an object or list")


def load_tag_map_records(tag_map_path: Path) -> dict[str, str] | list[object]:
    if tag_map_path.suffix == ".jsonl":
        return load_jsonl_records(tag_map_path)
    payload = json.loads(tag_map_path.read_text(encoding="utf-8"))
    return normalize_tag_map_payload(payload)


def load_visulogic_tag_map(
    tag_map_path: Path,
    *,
    id_field: str = "id",
    tag_field: str | None = None,
) -> dict[str, str]:
    if not tag_map_path.is_file():
        raise FileNotFoundError(f"missing VisuLogic tag map: {tag_map_path}")
    records_or_map = load_tag_map_records(tag_map_path)
    if isinstance(records_or_map, dict):
        return records_or_map

    tag_map: dict[str, str] = {}
    for index, record in enumerate(records_or_map):
        if not isinstance(record, dict):
            raise ValueError(f"tag map row {index} is not an object")
        record_id = record.get(id_field)
        if record_id is None or not str(record_id).strip():
            raise ValueError(
                f"tag map row {index} is missing id field {id_field!r}"
            )
        tag = extract_tag_from_mapping_record(record, tag_field=tag_field)
        if tag is None:
            raise ValueError(
                f"tag map row {index} is missing a tag/category/task field"
            )
        normalized_tag = normalize_materialized_tag(tag)
        record_id_text = str(record_id)
        prior = tag_map.get(record_id_text)
        if prior is not None and normalize_tag(prior) != normalize_tag(
            normalized_tag
        ):
            raise ValueError(
                f"conflicting tags for VisuLogic id {record_id_text}: "
                f"{prior!r} vs {normalized_tag!r}"
            )
        tag_map[record_id_text] = normalized_tag
    return tag_map


def extract_tag_from_mapping_record(
    record: dict[str, Any],
    *,
    tag_field: str | None = None,
) -> str | None:
    fields = (tag_field,) if tag_field else TAG_FIELDS
    for field in fields:
        if field is None:
            continue
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def normalize_materialized_tag(tag: str) -> str:
    return " ".join(str(tag).strip().split())


def load_optional_tag_map_manifest(
    tag_map_path: Path | None,
) -> dict[str, Any]:
    if tag_map_path is None:
        return {}
    candidates = [
        tag_map_path.with_suffix(".manifest.json"),
        tag_map_path.with_name(f"{tag_map_path.name}.manifest.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(
                    f"tag map manifest must be an object: {candidate}"
                )
            return payload
    return {}


def resolve_tag_map_inputs(
    *,
    tag_map_path: Path | None,
    id_field: str,
    tag_field: str | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    tag_map = (
        load_visulogic_tag_map(
            tag_map_path,
            id_field=id_field,
            tag_field=tag_field,
        )
        if tag_map_path is not None
        else {}
    )
    return tag_map, load_optional_tag_map_manifest(tag_map_path)


def resolve_tag_map_flags(
    tag_map_manifest: dict[str, Any],
    *,
    allow_nonformal_tag_map: bool,
) -> dict[str, object]:
    tag_source_kind = str(tag_map_manifest.get("source_kind") or "")
    uses_eval_or_staged_labels = bool(
        tag_map_manifest.get("uses_eval_or_staged_labels", False)
    )
    official_row_level_labels = bool(
        tag_map_manifest.get("official_row_level_labels", False)
    )
    heuristic_tags_require_result_reporting = bool(
        tag_map_manifest.get("heuristic_tags_require_result_reporting", False)
    )
    nonformal_tag_map = bool(tag_map_manifest.get("nonformal_tag_map", False))
    legacy_heuristic_tag_map = (
        tag_source_kind == "train_solution_keyword_heuristic_v1"
        or heuristic_tags_require_result_reporting
    )
    nonformal_tag_map = nonformal_tag_map or legacy_heuristic_tag_map
    formal_training_label_source = bool(
        tag_map_manifest.get(
            "formal_training_label_source",
            not nonformal_tag_map,
        )
    )
    if nonformal_tag_map:
        formal_training_label_source = False
    if nonformal_tag_map and not allow_nonformal_tag_map:
        raise ValueError(
            "non-formal VisuLogic tag maps are not valid formal VLM "
            "training inputs; use an audited/manual tag map, or pass "
            "allow_nonformal_tag_map=True only for debug materialization"
        )
    return {
        "tag_source_kind": tag_source_kind,
        "uses_eval_or_staged_labels": uses_eval_or_staged_labels,
        "official_row_level_labels": official_row_level_labels,
        "heuristic_tags_require_result_reporting": (
            heuristic_tags_require_result_reporting
        ),
        "nonformal_tag_map": nonformal_tag_map,
        "formal_training_label_source": formal_training_label_source,
    }


def materialize_visulogic_tagged_records(
    *,
    input_jsonl: Path,
    tag_map: dict[str, str],
    id_field: str,
) -> tuple[list[dict[str, Any]], list[str], Counter[str]]:
    materialized_records: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    tag_counter: Counter[str] = Counter()

    with input_jsonl.open(encoding="utf-8") as handle:
        for row_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id = str(record.get(id_field) or row_index)
            existing_tag = extract_visulogic_tag(record)
            mapped_tag = tag_map.get(record_id)
            if (
                existing_tag
                and mapped_tag
                and normalize_tag(existing_tag) != normalize_tag(mapped_tag)
            ):
                raise ValueError(
                    f"conflicting tag for VisuLogic id {record_id}: "
                    f"existing={existing_tag!r}, mapped={mapped_tag!r}"
                )
            tag = existing_tag or mapped_tag
            copied = dict(record)
            if tag:
                copied["tag"] = normalize_materialized_tag(tag)
                tag_counter[copied["tag"]] += 1
            else:
                missing_ids.append(record_id)
            materialized_records.append(copied)
    return materialized_records, missing_ids, tag_counter


def write_materialized_jsonl(
    *,
    out_jsonl: Path,
    records: list[dict[str, Any]],
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def materialize_tagged_visulogic_train_jsonl(
    *,
    input_jsonl: Path,
    out_jsonl: Path,
    manifest_path: Path | None = None,
    tag_map_path: Path | None = None,
    id_field: str = "id",
    tag_field: str | None = None,
    strict_complete: bool = True,
    force: bool = False,
    allow_nonformal_tag_map: bool = False,
) -> VisuLogicTaggedTrainResult:
    if not input_jsonl.is_file():
        raise FileNotFoundError(
            f"missing VisuLogic input jsonl: {input_jsonl}"
        )
    if out_jsonl.exists() and not force:
        raise FileExistsError(
            f"output already exists; pass force=True: {out_jsonl}"
        )
    resolved_manifest_path = manifest_path or out_jsonl.with_name(
        f"{out_jsonl.stem}.manifest.json"
    )
    if resolved_manifest_path.exists() and not force:
        raise FileExistsError(
            "manifest already exists; pass force=True: "
            f"{resolved_manifest_path}"
        )

    tag_map, tag_map_manifest = resolve_tag_map_inputs(
        tag_map_path=tag_map_path,
        id_field=id_field,
        tag_field=tag_field,
    )
    tag_flags = resolve_tag_map_flags(
        tag_map_manifest,
        allow_nonformal_tag_map=allow_nonformal_tag_map,
    )
    materialized_records, missing_ids, tag_counter = (
        materialize_visulogic_tagged_records(
            input_jsonl=input_jsonl,
            tag_map=tag_map,
            id_field=id_field,
        )
    )

    if strict_complete and missing_ids:
        sample = ", ".join(missing_ids[:10])
        raise ValueError(
            "tagged VisuLogic train materialization is incomplete: "
            f"{len(missing_ids)} rows have no tag; sample ids: {sample}"
        )

    write_materialized_jsonl(
        out_jsonl=out_jsonl,
        records=materialized_records,
    )
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total_count = len(materialized_records)
    tagged_count = total_count - len(missing_ids)
    manifest = {
        "schema": "visulogic_tagged_train_manifest_v1",
        "input_jsonl": str(input_jsonl),
        "out_jsonl": str(out_jsonl),
        "tag_map_path": str(tag_map_path) if tag_map_path is not None else "",
        "id_field": id_field,
        "tag_field": tag_field or "",
        "strict_complete": strict_complete,
        "total_count": total_count,
        "tagged_count": tagged_count,
        "missing_count": len(missing_ids),
        "missing_ids_sample": missing_ids[:20],
        "tag_counts": dict(tag_counter),
        "tag_source_kind": tag_flags["tag_source_kind"],
        "official_row_level_labels": tag_flags["official_row_level_labels"],
        "uses_eval_or_staged_labels": tag_flags["uses_eval_or_staged_labels"],
        "heuristic_tags_require_result_reporting": tag_flags[
            "heuristic_tags_require_result_reporting"
        ],
        "nonformal_tag_map": tag_flags["nonformal_tag_map"],
        "formal_training_label_source": tag_flags[
            "formal_training_label_source"
        ],
        "allow_nonformal_tag_map": allow_nonformal_tag_map,
        "safe_for_full_vlm_domain_training": (
            strict_complete
            and not missing_ids
            and not tag_flags["uses_eval_or_staged_labels"]
            and bool(tag_flags["formal_training_label_source"])
            and not tag_flags["nonformal_tag_map"]
        ),
    }
    resolved_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return VisuLogicTaggedTrainResult(
        input_jsonl=input_jsonl,
        out_jsonl=out_jsonl,
        manifest_path=resolved_manifest_path,
        total_count=total_count,
        tagged_count=tagged_count,
        missing_count=len(missing_ids),
        tag_counts=dict(tag_counter),
    )


def write_verl_parquet_rows(
    rows: Sequence[dict[str, Any]], path: Path
) -> None:
    try:
        datasets = importlib.import_module("datasets")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "writing VERL parquet requires the datasets/pyarrow stack; "
            "run this inside the VERL training container or install datasets"
        ) from exc
    dataset = datasets.Dataset.from_list(list(rows))
    dataset.to_parquet(str(path))


def run_visulogic_verl_dataset_self_test() -> dict[str, object]:
    records = [
        {
            "id": "00000",
            "question": "Question A?\nA: A\nB: B\nC: C\nD: D",
            "answer": "D",
            "tag": "Quantitative Reasoning",
            "message": json.dumps(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "images/00000.png"},
                            {"type": "text", "text": "Question A?"},
                        ],
                    }
                ]
            ),
        },
        {
            "id": "00001",
            "question": "Question B?\nA: A\nB: B\nC: C\nD: D",
            "label": "B",
            "tag": "Spatial Reasoning",
            "image_path": "images/00001.png",
        },
    ]
    image_root = Path("/tmp/visulogic-train/images")
    rows = [
        convert_visulogic_record_to_verl_row(
            record,
            image_root=image_root,
            image_uri_root="file:///tmp/visulogic-train/images",
            row_index=index,
            strict_images=False,
        )
        for index, record in enumerate(records)
    ]
    filtered_rows = [
        row
        for row in rows
        if row["extra_info"].get("tag") == "Quantitative Reasoning"
    ]
    train_rows, val_rows = split_verl_rows(rows, val_ratio=0.5, seed=1)
    sample = train_rows[0] if train_rows else val_rows[0]
    checks = {
        "has_prompt": bool(sample.get("prompt")),
        "has_image_placeholder": "<image>" in sample["prompt"][0]["content"],
        "has_image_uri": sample["images"][0]["image"].startswith("file://"),
        "has_reward_ground_truth": sample["reward_model"]["ground_truth"]
        in {
            "A",
            "B",
            "C",
            "D",
        },
        "has_tag_metadata": all("tag" in row["extra_info"] for row in rows),
        "supports_image_path_format": rows[1]["reward_model"]["ground_truth"]
        == "B",
        "filtered_row_count": len(filtered_rows),
        "split_counts": [len(train_rows), len(val_rows)],
    }
    passed = all(
        value if isinstance(value, bool) else bool(value)
        for value in checks.values()
    )
    return {"passed": passed, "checks": checks, "sample": sample}
