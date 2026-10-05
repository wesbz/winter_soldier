# Clean release notes

## Retained

- Gradient-matching/Gumbel-Softmax crafting, random and chosen-text secrets,
  pairwise baselines, controls, real-text blending, model/tokenizer code, and
  checkpoint helpers.

## Excluded

- All historical Git objects, branches, and tags.
- Debug runs, saved secret tokens and texts, serialized tensors, JSONL/CSV
  outputs, notebooks, figures, one-off analysis scripts, experiment tables,
  historical sweep keys, and private filesystem/cluster launcher settings.

## Portability fixes and additions

- Replace the private-path launcher and numbered grids with a validated CLI
  and portable examples. Resolve null seeds reproducibly before saving runs.
- Support standard HF embedding access and local HF model directories; do not
  pass Lingua-only forward arguments to HF models.
- Preserve eager attention for higher-order derivatives; make xFormers optional.
- Make real-text corpus paths and tokenizer padding IDs explicit inputs.
- Fix two undefined dtype references in the real-text variant.
- Export both discrete-token and legacy-logit artifacts from the supplied run
  directory rather than depending on an old embedded output path.
- Avoid distributed initialization for help/export commands; reject multi-rank
  within-run crafting rather than silently applying unsynchronized updates.
  Independent single-device runs can still be distributed as grid tasks.
- Add teacher-forced top-k detection with explicit statistical caveats, tests,
  installation documentation, release hygiene, and upstream license notices.

The original gradient objective and signed-Adam crafting update are retained.
Public example seeds/sentences replace historical secret material. The original
code initializes the selected-token logit to positive `initial_coeff` and the
others to zero; that convention is retained, rather than silently changing it
based on a differently worded paper description.

Offline tests exercise all seven methods on a toy local HF model plus the local
Lingua model's second-order derivative path. Full LM pretraining, benchmark
results, CUDA/bfloat16, and distributed checkpoint consolidation were not
reproduced. The complete trainer and historical training configurations are
not present in the supplied snapshot; see TRAINING.md.
