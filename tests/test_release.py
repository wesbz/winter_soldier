"""CPU/offline checks with a randomly initialized toy LM, not paper-scale results."""
import json
import random
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from detect import binomial_p_value, detect
from launcher import execute, grid_parameters, prepare_config
from src.config import EvalConfig, TrainConfig
from src.craft import cosine_similarity, extract_text_taggants, load_model
from src.transformer import LMTransformer, LMTransformerArgs


@pytest.fixture
def toy_checkpoint(tmp_path):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    root = tmp_path / "model"
    words = ["<unk>", "<bos>", "<eos>", "<pad>"] + [f"word{i}" for i in range(28)]
    tokenizer = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>",
                                bos_token="<bos>", eos_token="<eos>", pad_token="<pad>")
    hf.save_pretrained(root)
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(words), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=3,
    ))
    model.save_pretrained(root)
    return root


def test_binomial():
    assert binomial_p_value(0, 10, 1, 100) == 1
    assert binomial_p_value(2, 2, 1, 10) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        binomial_p_value(3, 2, 1, 10)


def test_grid_and_seed_resolution():
    assert list(grid_parameters({}, {})) == [{}]
    assert len(list(grid_parameters({"key_seed": [0, 1]}, {"seq_len": [4, 8], "batch_size": [2, 1]}))) == 4
    with pytest.raises(ValueError):
        list(grid_parameters({}, {"seq_len": [4], "batch_size": [1, 2]}))
    a = prepare_config(TrainConfig(checkpoint="demo", init_seed=None), random.Random(4))
    b = prepare_config(TrainConfig(checkpoint="demo", init_seed=None), random.Random(4))
    assert a.init_seed == b.init_seed


def test_hf_loader(toy_checkpoint):
    model, embedding, tokenizer = load_model(str(toy_checkpoint))
    assert embedding.shape == (32, 16)
    assert tokenizer.bos_id == 1
    assert model(torch.tensor([[1, 5, 6]])).logits.shape == (1, 3, 32)


def test_lingua_second_order_gradients():
    torch.set_default_dtype(torch.float32)
    model = LMTransformer(LMTransformerArgs(dim=16, n_layers=1, n_heads=2,
                                           vocab_size=32, multiple_of=8, max_seqlen=32))
    model.init_weights()
    inputs = torch.randn(1, 4, 32, requires_grad=True)
    embeds = inputs.softmax(dim=-1) @ model.tok_embeddings.weight
    loss = model(inputs_embeds=embeds, attn_impl="eager").logits.square().mean()
    gradient = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    target = tuple(torch.ones_like(g) for g in gradient)
    similarity = cosine_similarity(gradient, target)
    second = torch.autograd.grad(similarity, inputs)[0]
    assert torch.isfinite(second).all()


@pytest.mark.parametrize("method", ["craft", "attack", "copy", "copy_attack", "control", "control_attack", "real"])
def test_tiny_craft_export_and_detection(toy_checkpoint, tmp_path, method):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"text": "word0 word1 word2 word3 word4 word5"}) + "\n")
    config = TrainConfig(checkpoint=str(toy_checkpoint), output_dir=tmp_path / method,
                         device="cpu", dtype="fp32", key_len=3, value_len=1,
                         key_sentence="word0 word1 word2", value_sentence="word3",
                         n_seq=2, seq_len=4, batch_size=2, num_iter=1,
                         pad_token_id=3, n_control=2, corpus_path=corpus)
    # The real-text sampler needs at least n_seq suitable documents.
    corpus.write_text(corpus.read_text() * config.n_seq)
    directory = Path(execute(config, method, random.Random(0)))
    assert (directory / "taggants.pt").exists()
    assert not json.loads((directory / "config.json").read_text())["init_seed"] is None
    extract_text_taggants(EvalConfig(directory))
    exported = [json.loads(l) for l in (directory / "text_taggants.jsonl").read_text().splitlines()]
    assert exported and all(isinstance(row["text"], str) for row in exported)
    result = detect(str(toy_checkpoint), [directory], top_k=1)
    assert result["trials"] >= 1
    assert 0 <= result["accuracy"] <= 1
    if method in {"craft", "copy", "real"}:
        assert 0 <= result["p_value"] <= 1
    else:
        assert result["p_value"] is None


def test_reference_configs_and_single_run(monkeypatch):
    import launcher
    monkeypatch.setenv("WS_CHECKPOINT", "public-demo-checkpoint")
    observed = []
    monkeypatch.setattr(launcher, "execute", lambda config, method, rng: observed.append(prepare_config(config, rng)))
    launcher.run_grid("configs/paper.yaml", seed=7)
    assert len(observed) == 12
    assert len({config.value_seed for config in observed}) == 12
    assert all(config.checkpoint == "public-demo-checkpoint" for config in observed)
    observed.clear()
    launcher.run_grid("configs/semantic.yaml", craft_type="attack")
    assert len(observed) == 1
    observed.clear()
    launcher.run_experiment("local-demo", num_iter=1, device="cpu", dtype="fp32", pad_token_id=None)
    assert observed[0].num_iter == 1 and observed[0].pad_token_id is None
