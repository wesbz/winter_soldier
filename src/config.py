"""Portable experiment configuration; example values are not private keys."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    checkpoint: str = ""
    key_seed: Optional[int] = 0
    key_len: int = 256
    key_sentence: str = "The quick brown fox jumps over the lazy dog"
    value_seed: Optional[int] = 1
    value_len: int = 1
    value_sentence: str = " orange"
    init_seed: Optional[int] = 2
    sampling_seed: Optional[int] = 3
    n_seq: int = 128
    seq_len: int = 64
    initial_coeff: float = 15.0
    mask_special_tokens: bool = True
    mask_key_tokens: bool = False
    mask_value_tokens: bool = False
    num_iter: int = 200
    batch_size: int = 64
    optimizer: str = "signAdam"
    lr: float = 0.9
    temperature: float = 0.6
    p: float = 1.0
    n_control: int = 100
    device: str = "cuda"
    dtype: str = "bf16"
    output_dir: Path = Path("outputs/experiments")
    corpus_path: Optional[Path] = None
    pad_token_id: Optional[int] = 16  # Cosmo2 <empty_output>; use null for other tokenizers.


@dataclass
class EvalConfig:
    eval_dir: Path
    checkpoint: Optional[str] = None
    device: Optional[str] = None
    output_dir: Optional[Path] = None
