"""Teacher-forced top-k response-token detection for generated secret keys."""
import argparse
import json
from pathlib import Path

import torch
from scipy.stats import binom

from src.craft import load_model


def binomial_p_value(hits, trials, top_k, vocabulary_size):
    if trials < 1 or not 0 <= hits <= trials:
        raise ValueError("Require trials >= 1 and 0 <= hits <= trials")
    if not 1 <= top_k <= vocabulary_size:
        raise ValueError("Require 1 <= top_k <= vocabulary_size")
    return float(binom.sf(hits - 1, trials, top_k / vocabulary_size))


def detect(checkpoint, runs, top_k=1, device="cpu"):
    model, embedding, tokenizer = load_model(checkpoint)
    model.to(device).eval()
    # Secrets were sampled from non-special vocabulary IDs. k/V is conservative
    # when a model's top-k includes special tokens outside this sampling support.
    vocabulary_size = len(tokenizer.tokenizer) - len(set(tokenizer.tokenizer.all_special_ids))
    if not 1 <= top_k <= vocabulary_size:
        raise ValueError("top_k must be within the non-special vocabulary size")
    hits, trials, seen = 0, 0, set()
    uniform = True
    for directory in runs:
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        key = torch.load(directory / "key_tokens.pt", map_location="cpu", weights_only=True).long()
        value = torch.load(directory / "val_tokens.pt", map_location="cpu", weights_only=True).long()
        identity = (tuple(key.tolist()), tuple(value.tolist()))
        if identity in seen:
            raise ValueError("Duplicate secret: do not count repeated copies as independent trials")
        seen.add(identity)
        # Our launcher records this metadata; older exports lack enough information
        # to certify a random-token null distribution.
        uniform &= config.get("craft_type") in {"craft", "copy", "real"}
        tokens = torch.cat([torch.tensor([tokenizer.bos_id]), key, value]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tokens).logits[0, len(key):len(key) + len(value)]
            hits += int((logits.topk(top_k, dim=-1).indices.cpu() == value[:, None]).any(dim=-1).sum())
        trials += len(value)
    result = {"hits": hits, "trials": trials, "top_k": top_k,
              "vocabulary_size": vocabulary_size, "accuracy": hits / trials,
              "uniform_secret_assumption": bool(uniform)}
    result["p_value"] = binomial_p_value(hits, trials, top_k, vocabulary_size) if uniform else None
    if not uniform:
        result["note"] = "No certified binomial p-value for semantic, retokenized control, or unclassified historical secrets."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(json.dumps(detect(args.checkpoint, args.runs, args.top_k, args.device), indent=2))
