# Third-party code

`src/transformer.py`, `src/tokenizer.py`, and checkpoint/distributed helpers in
`src/utils.py` are adapted from [Meta Lingua](https://github.com/facebookresearch/lingua),
which is licensed under BSD-3-Clause. The license is in `LICENSES/lingua.txt` and
upstream copyright notices are retained. Research modifications include eager
attention for higher-order derivatives, Hugging Face-like model outputs, and
portable checkpoint loading. Hugging Face Transformers, PyTorch, and the other
installed dependencies retain their own licenses.
