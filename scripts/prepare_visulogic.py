#!/usr/bin/env python3
"""Prepare one CRG VisuLogic task sequence from the public training set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mllm_crl.data.visulogic import (
    materialize_subtyped_visulogic_jsonl,
    prepare_visulogic_verl_dataset,
)

SETTINGS = {
    "quantitative": {
        "tag": "Quantitative Reasoning",
        "mapping": "quantitative.jsonl",
    },
    "spatial": {
        "tag": "Spatial Reasoning",
        "mapping": "spatial.jsonl",
    },
    "positional": {
        "tag": "Positional Reasoning",
        "mapping": "positional.jsonl",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join CRG's released subtype labels with the public VisuLogic "
            "training JSONL and write the VERL-ready 80/20 split."
        )
    )
    parser.add_argument("--setting", choices=sorted(SETTINGS), required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--image-uri-root", default=None)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--subtype-map",
        type=Path,
        default=None,
        help="Override the released ID-to-subtype mapping (mainly for tests).",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Skip local image-existence checks when preparing remote URIs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    setting = SETTINGS[args.setting]
    subtype_map = args.subtype_map or (
        repo_root / "data" / "visulogic_subtypes" / setting["mapping"]
    )
    out_dir = args.out_root.resolve() / args.setting
    out_dir.mkdir(parents=True, exist_ok=True)
    joined_jsonl = out_dir / "source_with_crg_subtypes.jsonl"
    join_manifest = out_dir / "subtype_join_manifest.json"

    join_result = materialize_subtyped_visulogic_jsonl(
        source_jsonl=args.source_jsonl.resolve(),
        subtype_jsonl=subtype_map.resolve(),
        out_jsonl=joined_jsonl,
        manifest_path=join_manifest,
        force=args.force,
    )
    prepare_result = prepare_visulogic_verl_dataset(
        input_jsonl=joined_jsonl,
        image_root=args.image_root.resolve(),
        out_dir=out_dir,
        image_uri_root=args.image_uri_root,
        val_ratio=0.2,
        seed=args.seed,
        strict_images=not args.allow_missing_images,
        include_tags=[setting["tag"]],
        require_subcategory=True,
        subtype_field="subcategory",
        taxonomy=f"CRG {setting['tag']}",
    )
    print(
        json.dumps(
            {
                "setting": args.setting,
                "subtype_join": join_result,
                "dataset": prepare_result.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
