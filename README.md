# Winter Soldier

Research implementation for **Winter Soldier: Backdooring Language Models at
Pre-Training with Indirect Data Poisoning**
([paper](https://arxiv.org/abs/2506.14913)).

This code crafts text poisons by matching their training gradients to those of
a secret prompt/response, using differentiable token distributions and
Gumbel-Softmax. It includes random-secret and chosen-text variants, controls,
pairwise-token baselines, JSONL export, and a top-k detection helper.

**Scope:** the provided research snapshot contains poison crafting and model
loading, not the complete language-model pretraining pipeline or its original
training configurations. Full reproduction also requires a separate training
setup such as [Meta Lingua](https://github.com/facebookresearch/lingua), the
training corpus, and clean surrogate checkpoints. See [TRAINING.md](TRAINING.md).
Downloading a released HF model is useful for a demonstration, but does **not**
recreate the paper's surrogate or victim training conditions.

## Installation

Use Python 3.11 in a separate environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pinned PyTorch and Transformers versions provide a **legacy compatibility
environment**, not a recommendation to deploy old dependencies in production.
Install the matching CUDA build for GPU work. Crafting needs second-order
gradients and uses eager attention; Flash Attention is not a drop-in replacement.
xFormers is optional and unnecessary for the supported crafting path.

Hugging Face authentication, if required, uses the library's local credential
store or `HF_TOKEN` environment variable. Do not put tokens in YAML or source.

## 1. Supply a clean surrogate

`--checkpoint` accepts:

- a trusted consolidated Lingua directory containing `params.json` and
  `consolidated.pth`;
- a Lingua distributed checkpoint directory (consolidation creates a
  `consolidated/` subdirectory); or
- a Hugging Face model ID or a local HF directory containing `config.json`.

For Lingua checkpoints, edit tokenizer paths in your **local** `params.json`
to point to an accessible tokenizer. The default padding ID `16` is the
historical Cosmo2 `<empty_output>` token. For other tokenizers, pass the correct
`--pad_token_id` or `--pad_token_id=None` to use the existing/EOS padding token.
The tokenizer must supply BOS and EOS IDs. Keep the tokenizer and vocabulary
identical across surrogate, victim training, export, and detection.

## 2. Craft poisons

For a single random-secret experiment:

```bash
python launcher.py run \
  --checkpoint "$CLEAN_CHECKPOINT" \
  --output_dir outputs/random-secret \
  --key_len 256 --value_len 1 --n_seq 128 --seq_len 64 \
  --num_iter 200 --batch_size 64 --lr 0.9 --temperature 0.6 \
  --device cuda --dtype bf16
```

For the reference grid:

```bash
export WS_CHECKPOINT="$CLEAN_CHECKPOINT"
python launcher.py grid --config_path configs/paper.yaml --seed 0
```

The example uses explicit public demonstration seeds, not historical secret
keys. Null seeds are resolved deterministically from `--seed` and saved with
the run. For independent private experiments, choose fresh undisclosed seeds.

Other `--craft_type` values:

| Value | Variant |
| --- | --- |
| `craft` | Gradient matching for random secret tokens |
| `attack` | Gradient matching for a chosen prompt/response |
| `copy` / `copy_attack` | Pairwise-token baseline for random/chosen secrets |
| `control` / `control_attack` | Control candidates selected by gradient similarity |
| `real` | Blend learnable tokens with supplied text; requires `--corpus_path` JSONL |

`configs/semantic.yaml` contains an innocuous chosen-text example. Corpus JSONL
for `real` must contain a `text` field and enough documents of at least
`seq_len` tokens. These are research experiments for controlled environments;
do not use them to contaminate third-party training data without authorization.

Each run uses **one process/device**. For parallel sweeps launch separate
processes with distinct `--task` values and the same `--array_size` and `--seed`.
Assign each process its own GPU using `CUDA_VISIBLE_DEVICES`. Multi-rank
within-run crafting is intentionally rejected because the inherited optimizer
code does not correctly synchronize poison updates.

## 3. Use generated data in your training pipeline

Runs are written to `outputs/<experiment>/<run-id>/`:

- `config.json`: resolved settings and method metadata;
- `taggants.pt`: discrete token IDs;
- `tt.chunk.<id>.jsonl`: poison texts to mix into a controlled training corpus;
- `tt.<id>.val.jsonl`: a matching export, **not an independent held-out set**;
- `key_tokens.pt`, `val_tokens.pt`: private verification material;
- `secret*`: evaluation/control files; **exclude these from indirect-poison
  training**, or you would train on the secret itself;
- `craft.csv`, `eval.csv` when applicable: optimization diagnostics.

Some control variants save logits rather than token IDs. Both formats can be
exported without carrying old run paths:

```bash
python launcher.py extract --eval_dir "$RUN_DIR"
```

The result is `text_taggants.jsonl`. Generated data and keys are ignored by Git.
No historical keys, outputs, figures, or development Git history are distributed.

## 4. Detect in a separately trained model

```bash
python detect.py --checkpoint "$VICTIM_CHECKPOINT" --runs "$RUN_DIR" \
  --top-k 1 --device cuda
```

Detection uses teacher forcing over the known response and counts top-k hits.
For random uniformly sampled response tokens, it reports a one-sided binomial
tail using the non-special sampling vocabulary. The p-value is conditional on
the paper's independence assumptions, **not** the posterior probability of
ownership. Use compatible tokenizers and independent secret draws; do not
aggregate copies of a secret or select keys against the model being tested.
Semantic/control/unknown historical secrets get accuracy but no certified
binomial p-value. This helper is not the original paper's full evaluation suite.

## Tests and limitations

```bash
python -m pip install pytest
python -m pytest -q
```

The tests create a tiny random local language model and tokenizer, exercise all
seven crafting variants, export, and detection, and check Lingua higher-order
gradients. They run offline on CPU; PyTorch uses a local loopback socket.
They do not reproduce paper-scale training, benchmark tables, attack success
rates, or distributed checkpoint consolidation. Full-scale experiments need
substantial GPU memory: relaxed token tensors scale with sequences × length ×
vocabulary size. CUDA/bfloat16 and optional backends have not been validated
in this release's CPU-only checks.

See [RELEASE_NOTES.md](RELEASE_NOTES.md), [SECURITY.md](SECURITY.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [LICENSE.md](LICENSE.md).
No new repository-wide license was selected during cleanup.

## Citation

Please cite the paper linked above (arXiv:2506.14913).
