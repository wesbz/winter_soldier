"""Single-device entry points for crafting, grid experiments, and text export."""
import copy
import logging
import random
import uuid
from dataclasses import asdict
from itertools import product
from pathlib import Path

import fire
from omegaconf import OmegaConf

from src.config import EvalConfig, TrainConfig
from src.craft import (
    craft, craft_attack, craft_control, craft_control_attack,
    craft_copycat, craft_copycat_atk, craft_from_real, extract_text_taggants,
)
from src.utils import save_to_json, setup_torch_distributed, unset_torch_distributed

METHODS = {
    "craft": craft, "attack": craft_attack, "control": craft_control,
    "control_attack": craft_control_attack, "copy": craft_copycat,
    "copy_attack": craft_copycat_atk, "real": craft_from_real,
}


def prepare_config(config, rng):
    """Validate settings and materialize seeds before creating any run files."""
    if not config.checkpoint:
        raise ValueError("Provide a checkpoint directory or Hugging Face model ID")
    for field in ("n_seq", "seq_len", "key_len", "value_len", "batch_size", "num_iter", "n_control"):
        if getattr(config, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if config.dtype not in {"fp32", "fp16", "bf16"}:
        raise ValueError("dtype must be fp32, fp16, or bf16")
    if config.optimizer not in {"signAdam", "Adam"}:
        raise ValueError("optimizer must be signAdam or Adam")
    if config.temperature <= 0 or config.lr <= 0 or not 0 <= config.p <= 1:
        raise ValueError("temperature/lr must be positive and p must be in [0, 1]")
    for field in ("key_seed", "value_seed", "init_seed", "sampling_seed"):
        if getattr(config, field) is None:
            setattr(config, field, rng.randrange(2**32 - 1))
    config.output_dir = Path(config.output_dir).expanduser()
    if config.corpus_path is not None:
        config.corpus_path = Path(config.corpus_path).expanduser()
    return config


def execute(config, craft_type, rng):
    if craft_type not in METHODS:
        raise ValueError(f"Unknown craft type {craft_type!r}; choose from {list(METHODS)}")
    config = prepare_config(config, rng)
    if craft_type == "real" and config.corpus_path is None:
        raise ValueError("The real-text variant requires corpus_path")
    config.output_dir = config.output_dir / str(uuid.uuid4())
    (config.output_dir / "secret").mkdir(parents=True, exist_ok=False)
    save_to_json(asdict(config), config.output_dir / "config.json")
    logging.info("Writing run to %s", config.output_dir)
    setup_torch_distributed()
    try:
        METHODS[craft_type](config)
    finally:
        unset_torch_distributed()
    # Write method metadata after craft(), which persists its resolved seeds.
    save_to_json({**asdict(config), "craft_type": craft_type}, config.output_dir / "config.json")
    return str(config.output_dir)


def run_experiment(checkpoint: str, output_dir: str = "outputs/experiments",
                   craft_type: str = "craft", seed: int = 0, **overrides):
    """Run one experiment. Extra flags override TrainConfig fields."""
    config = OmegaConf.to_object(OmegaConf.merge(
        OmegaConf.structured(TrainConfig()),
        {"checkpoint": checkpoint, "output_dir": output_dir}, overrides,
    ))
    return execute(config, craft_type, random.Random(seed))


def grid_parameters(sweep, zipped):
    """Cartesian sweep with synchronized zipped axes (never silently truncate)."""
    if set(sweep) & set(zipped):
        raise ValueError("sweep and zip fields must be disjoint")
    allowed = set(TrainConfig.__dataclass_fields__)
    if (set(sweep) | set(zipped)) - allowed:
        raise ValueError("Unknown TrainConfig field in sweep or zip")
    if any(not values for values in [*sweep.values(), *zipped.values()]):
        raise ValueError("Sweep axes must not be empty")
    if len({len(values) for values in zipped.values()}) > 1:
        raise ValueError("All zip axes must have equal lengths")
    zipped_values = list(zip(*zipped.values())) if zipped else [()]
    for independent in product(*sweep.values()):
        for linked in zipped_values:
            yield dict(zip(sweep, independent)) | dict(zip(zipped, linked))


def run_grid(config_path: str, craft_type: str = "craft", seed: int = 0,
             array_size: int = 1, task: int = 0):
    """Split a deterministic sweep across independent single-device processes."""
    if array_size < 1 or not 0 <= task < array_size:
        raise ValueError("Require array_size >= 1 and 0 <= task < array_size")
    file_cfg = OmegaConf.load(config_path)
    defaults = OmegaConf.merge(OmegaConf.structured(TrainConfig()), file_cfg.get("default", {}))
    sweep = OmegaConf.to_container(file_cfg.get("sweep", OmegaConf.create({})), resolve=True)
    zipped = OmegaConf.to_container(file_cfg.get("zip", OmegaConf.create({})), resolve=True)
    rng = random.Random(seed)
    for index, overrides in enumerate(grid_parameters(sweep, zipped)):
        config = OmegaConf.to_object(OmegaConf.merge(copy.deepcopy(defaults), overrides))
        # Resolve for every task so seeds do not depend on array partitioning.
        prepare_config(config, rng)
        if index % array_size == task:
            execute(config, craft_type, rng)


def extract_run(eval_dir: str, checkpoint: str = None, output_dir: str = None):
    extract_text_taggants(EvalConfig(
        eval_dir=Path(eval_dir), checkpoint=checkpoint,
        output_dir=Path(output_dir) if output_dir else None,
    ))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire({"run": run_experiment, "grid": run_grid, "extract": extract_run})
