# Third-party components

CRG integrates with the following public projects without vendoring their
source code:

- [VERL](https://github.com/verl-project/verl), pinned to commit
  `adff7956cefd8ef707cd67dd8e08c06fa63679bd` and used under Apache-2.0.
  `patches/verl-crg-runtime.patch` contains the small runtime integration used
  by this release.
- [Reasoning Gym](https://github.com/open-thought/reasoning-gym), used to
  generate the text reasoning tasks.
- [mini_trainer](https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer),
  pinned to `f5d63c202eedd7ede7a7ab074f41a0a080b88af8` for the OSFT baseline.
- [VisuLogic-Train](https://github.com/VisuLogic-Benchmark/VisuLogic-Train)
  and [VisuLogic-Eval](https://github.com/VisuLogic-Benchmark/VisuLogic-Eval),
  used under Apache-2.0 for the visual reasoning settings. This repository
  contains only CRG's compact ID-to-subtype annotations, not the images or
  original dataset records.

Users remain responsible for following the upstream terms of the models and
datasets they download.
