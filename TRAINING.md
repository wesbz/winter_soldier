# Connecting crafting to language-model pretraining

The paper's full experiment is **not** just a run of `launcher.py`:

1. Prepare the training corpus and train a clean surrogate with the intended
   model architecture and tokenizer.
2. Craft poison texts using that trusted surrogate checkpoint.
3. Mix only the generated `tt.chunk.*.jsonl` texts into a controlled copy of the
   training corpus at the experiment's chosen sampling/repetition rate.
4. Train a fresh victim model from a different initialization on this mixture.
5. Evaluate clean utility and run secret-key detection against the victim.

The original experiment used Meta Lingua, SmolLM-style architectures and the
Cosmo2 tokenizer, and data from FineWeb-Edu and Cosmopedia v2. The paper specifies
135M, 360M, and 1.4B parameter models and different surrogate/victim token budgets.
Consult [Sections 4.1 and Appendix B](https://arxiv.org/abs/2506.14913) for the
experiment you want to repeat. They are not interchangeable with an arbitrary
fine-tuning experiment on a released model.

## Integration contract

- JSONL poison records have the form `{"text": "..."}`. Pass them through the
  **same tokenizer and document-boundary processing** as the clean corpus.
- Keep evaluation secrets out of the training mixture for indirect-poison
  experiments. The `secret/` directory is for explicit-secret controls and
  evaluation, not the default training data.
- Keep train/validation splits separate: the generated `.val.jsonl` poison
  export repeats the same poisons and must not be counted as independent data.
- Record clean/poison mixture weights, repetitions, total training tokens,
  initialization seeds, checkpoint step, tokenizer identity, and training-code
  revision with your local results.
- Detection requires the victim and crafting tokenizer to agree on token IDs.

## What is not supplied

The provided repository snapshot does not contain Lingua's full trainer, its
original training YAMLs, corpus preparation/deduplication, clean checkpoints,
victim checkpoints, or the complete benchmark harness. These were referenced
through private filesystem paths and cannot be reconstructed from those paths.
This release deliberately does not fabricate replacement configurations or
claim that it can reproduce every paper table standalone.

Use an independently installed [Meta Lingua](https://github.com/facebookresearch/lingua)
checkout or another compatible trainer. Record the version and your own training
configuration. Exact paper-scale reproduction additionally requires recovering
the original training configurations from their owners.
