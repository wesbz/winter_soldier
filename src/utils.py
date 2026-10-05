import csv
import json
import logging
import multiprocessing as mp
import os
import random
import subprocess
from functools import lru_cache
from pathlib import PosixPath, Path
from string import Template
from typing import Union, Dict, Any, List, Tuple, Type, TypeVar

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

logger = logging.getLogger()
IGNORE_INDEX = -100
MAX_SEQ_LEN = 4096

CONSOLIDATE_FOLDER = "consolidated"
CONSOLIDATE_NAME = "consolidated.pth"

CONFIG_NAME = "params.json"

T = TypeVar("T")

LM_EVAL_TASK_SCRIPT = Template("""
task: secret_keys_$id
dataset_path: json
dataset_name: null
dataset_kwargs:
  data_files: $output_dir/secret.$id.jsonl
training_split: null
validation_split: null
test_split: train
output_type: loglikelihood
doc_to_text: key
doc_to_target: value
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
    ignore_case: false
    ignore_punctuation: false
  - metric: perplexity
    aggregation: mean
    higher_is_better: true
    ignore_case: false
    ignore_punctuation: false
#     - "<|endoftext|>"
#     - "</s>"
#     - "<|im_end|>"
repeats: 1
""")


def save_to_csv(
    data: Union[Dict[str, Any], List[Dict[str, Any]]], filename: str
) -> None:
    if isinstance(data, dict):
        fieldnames = list(data.keys())
    else:
        fieldnames = list(data[0].keys())

    try:
        with open(filename, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            _ = next(reader)
    except Exception:
        with open(filename, "w") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()
    # Add row for this experiment
    with open(filename, "a") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        if isinstance(data, dict):
            writer.writerow(data)
        else:
            writer.writerows(data)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PosixPath):
            return obj.as_posix()
        elif isinstance(obj, np.integer):
            return int(obj)
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def save_to_json(data: Dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(data, f, cls=CustomEncoder)


@lru_cache()
def get_is_torch_run() -> bool:
    return os.environ.get("LOCAL_RANK") is not None


@lru_cache()
def get_is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ and not get_is_torch_run()


@lru_cache()
def get_global_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


@lru_cache()
def get_local_rank() -> int:
    if get_is_torch_run():
        return int(os.environ["LOCAL_RANK"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_LOCALID"])
    else:
        return 0


@lru_cache()
def get_world_size() -> int:
    if get_is_torch_run():
        return int(os.environ["WORLD_SIZE"])
    elif get_is_slurm_job():
        return int(os.environ["SLURM_NTASKS"])
    else:
        return 1


@lru_cache()
def get_is_master() -> bool:
    return get_global_rank() == 0


@lru_cache()
def get_master_port(job_id: int) -> int:
    if get_is_torch_run():
        return int(os.environ["MASTER_PORT"])
    else:
        MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)
        rng = random.Random(job_id)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)


@lru_cache()
def get_master_addr() -> str:
    if get_is_torch_run():
        return os.environ["MASTER_ADDR"]
    elif get_is_slurm_job():
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        return hostnames.split()[0].decode("utf-8")
    else:
        return "127.0.0.1"


def setup_torch_distributed() -> Tuple[int, int]:
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - global_rank
        - world_size
    """
    if get_world_size() != 1:
        raise ValueError("Crafting supports one process per run; distribute independent runs with grid --array_size/--task")
    local_rank = get_local_rank()

    os.environ["RANK"] = str(get_global_rank())
    os.environ["WORLD_SIZE"] = str(get_world_size())
    os.environ["MASTER_ADDR"] = get_master_addr()
    os.environ["MASTER_PORT"] = str(
        get_master_port(job_id=int(os.environ.get("SLURM_JOB_ID", -1)))
    )

    if get_is_torch_run():
        logger.info(f"Run launched with torchrun, local rank: {local_rank}")
    elif get_is_slurm_job():
        logger.info(f"Run launched with slurm, local rank: {local_rank}")
    else:
        logger.info("Single GPU job")

    # set GPU device
    assert 0 <= local_rank < 8
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(init_method="env://", backend="gloo")

    return get_world_size(), get_global_rank()


def unset_torch_distributed() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Torch distributed destroyed")
    else:
        logger.info("Torch distributed was not initialized")


def consolidate_checkpoints(ckpt_dir: str):
    """
    Consolidates all FSDP checkpoints in a directory to a single file
    Consolidate checkpoint is saved in a subdirectory of ckpt_dir

    Parameters:
        ckpt_dir: str - path to the directory containing the checkpoints

    Returns the path to the consolidated checkpoint
    """
    consolidate_path = Path(ckpt_dir) / CONSOLIDATE_FOLDER
    if not (consolidate_path / CONSOLIDATE_NAME).exists():
        consolidate_path.mkdir(exist_ok=True)
        logger.info(f"Consolidating to: {str(consolidate_path)}")
        dcp_to_torch_save(ckpt_dir, str(consolidate_path / CONSOLIDATE_NAME))
        (consolidate_path / CONFIG_NAME).write_text(
            (Path(ckpt_dir) / CONFIG_NAME).read_text()
        )
        logger.info("Consolidated !")
    return consolidate_path


def dataclass_from_dict(cls: Type[T], data: dict, strict: bool = True) -> T:
    """
    Converts a dictionary to a dataclass instance, recursively for nested structures.
    """
    base = OmegaConf.structured(cls())
    OmegaConf.set_struct(base, strict)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))
