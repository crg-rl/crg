# CRG VisuLogic subtype annotations

These files map records from the public VisuLogic training set to the stage-level subtypes used by the three visual task sequences in Continual Reasoning Gym.

| File | VisuLogic category | Records |
| --- | --- | ---: |
| `quantitative.jsonl` | Quantitative Reasoning | 1,386 |
| `spatial.jsonl` | Spatial Reasoning | 1,043 |
| `positional.jsonl` | Positional Reasoning | 743 |

Each JSONL record contains only the upstream record ID, its top-level VisuLogic category, and its CRG subtype:

```json
{"id":"00004","tag":"Quantitative Reasoning","subcategory":"Point Quantity"}
```

Use `scripts/prepare_visulogic.py` to join these annotations with the public VisuLogic source records and create the deterministic 80/20 train/test split used by the release.
