from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


from mllm_crl.data.visulogic import (
    materialize_subtyped_visulogic_jsonl,
    prepare_visulogic_verl_dataset,
    split_verl_rows,
)


def record(row_id: str, subcategory: str | None = None) -> dict:
    row = {
        "id": row_id,
        "question": f"Question {row_id}",
        "answer": "A",
        "message": json.dumps(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"images/{row_id}.png"},
                        {"type": "text", "text": f"Question {row_id}"},
                    ],
                }
            ]
        ),
        "tag": "Quantitative Reasoning",
    }
    if subcategory is not None:
        row["subcategory"] = subcategory
    return row


class VisuLogicPrepareTests(unittest.TestCase):
    def test_join_and_prepare_eighty_twenty_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            image_root.mkdir()
            source = root / "source.jsonl"
            mapping = root / "mapping.jsonl"
            rows = []
            labels = []
            for index in range(20):
                row_id = f"{index:05d}"
                rows.append(record(row_id))
                labels.append(
                    {
                        "id": row_id,
                        "tag": "Quantitative Reasoning",
                        "subcategory": "Linear Quantity"
                        if index < 10
                        else "Point Quantity",
                    }
                )
                (image_root / f"{row_id}.png").write_bytes(b"png")
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            mapping.write_text(
                "".join(json.dumps(row) + "\n" for row in labels),
                encoding="utf-8",
            )
            joined = root / "joined.jsonl"
            materialize_subtyped_visulogic_jsonl(
                source_jsonl=source,
                subtype_jsonl=mapping,
                out_jsonl=joined,
            )

            captured: dict[str, list[dict]] = {}

            def writer(items, path):
                captured[path.name] = list(items)
                path.write_text("stub", encoding="utf-8")

            result = prepare_visulogic_verl_dataset(
                input_jsonl=joined,
                image_root=image_root,
                out_dir=root / "prepared",
                val_ratio=0.2,
                seed=7,
                require_subcategory=True,
                subtype_field="subcategory",
                parquet_writer=writer,
            )

            self.assertEqual(result.train_count, 16)
            self.assertEqual(result.val_count, 4)
            self.assertEqual(len(captured["train.parquet"]), 16)
            self.assertEqual(len(captured["val.parquet"]), 4)
            self.assertEqual(
                {
                    row["extra_info"]["subcategory"]
                    for row in captured["val.parquet"]
                },
                {"Linear Quantity", "Point Quantity"},
            )

    def test_released_subtype_annotations(self) -> None:
        expected = {
            "quantitative.jsonl": (1386, "Quantitative Reasoning"),
            "spatial.jsonl": (1043, "Spatial Reasoning"),
            "positional.jsonl": (743, "Positional Reasoning"),
        }
        data_root = ROOT / "data" / "visulogic_subtypes"
        for filename, (count, tag) in expected.items():
            rows = [
                json.loads(line)
                for line in (data_root / filename).read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(count, len(rows))
            self.assertEqual(count, len({row["id"] for row in rows}))
            self.assertEqual({tag}, {row["tag"] for row in rows})
            self.assertTrue(all(set(row) == {"id", "tag", "subcategory"} for row in rows))
            self.assertTrue(all(row["subcategory"].strip() for row in rows))

    def test_split_is_deterministic(self) -> None:
        rows = [
            {
                "extra_info": {
                    "index": index,
                    "subcategory": "A" if index < 10 else "B",
                }
            }
            for index in range(20)
        ]
        first = split_verl_rows(
            rows, val_ratio=0.2, seed=11, stratify_field="subcategory"
        )
        second = split_verl_rows(
            rows, val_ratio=0.2, seed=11, stratify_field="subcategory"
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
