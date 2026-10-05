import gc
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import TrainConfig, EvalConfig
from .utils import (
    save_to_csv,
    save_to_json,
    get_world_size,
    get_global_rank,
    get_is_master,
    dataclass_from_dict,
    consolidate_checkpoints,
    CONSOLIDATE_FOLDER,
    CONSOLIDATE_NAME,
    IGNORE_INDEX,
    LM_EVAL_TASK_SCRIPT,
)
from .tokenizer import build_tokenizer
from .transformer import LMTransformer, LMTransformerArgs

logger = logging.getLogger(__name__)
torch._dynamo.config.optimize_ddp = False
N_RESAMPLE_EVAL = 1
N_ITER_PROJECT = 50


def cosine_similarity(x: tuple, y: tuple) -> float:
    x_norm = 0.0
    y_norm = 0.0
    dot = 0.0
    for xi, yi in zip(x, y):
        xi = xi.to('cpu')
        yi = yi.to('cpu')
        dot += (xi * yi).sum()
        x_norm += (xi**2).sum()
        y_norm += (yi**2).sum()
    return dot / (x_norm.sqrt() * y_norm.sqrt() + 1e-8)


def load_model(model_name: str):
    if not Path(model_name).exists() or (Path(model_name) / "config.json").is_file():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="eager"
        )
        embedding = model.get_input_embeddings().weight
        tokenizer = build_tokenizer('hf', model_name)
        return model, embedding, tokenizer

    if (
        Path(model_name).exists()
        and (Path(model_name) / "params.json").exists()
        and next(Path(model_name).glob("*.pth"), None) is not None
    ):
        consolidate_path = Path(model_name)
    else:
        consolidate_path = Path(model_name) / CONSOLIDATE_FOLDER
        if not consolidate_path.exists() and get_global_rank() == 0:
            consolidate_path = consolidate_checkpoints(model_name)
    ckpt_path = consolidate_path
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]

    model_args = dataclass_from_dict(LMTransformerArgs, config.model, strict=False)
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    model = LMTransformer(model_args)
    st_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, map_location="cpu", weights_only=True)
    model.load_state_dict(st_dict["model"])
    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)

    embedding = model.tok_embeddings.weight

    return model, embedding, tokenizer


def craft(config: TrainConfig) -> torch.Tensor:
    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Initialize model and tokenizer")
    dtype_dict = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
    torch.set_default_dtype(dtype_dict[config.dtype])
    model, embedding, tokenizer = load_model(config.checkpoint)
    if config.pad_token_id is not None:
        tokenizer.tokenizer.pad_token_id = config.pad_token_id
    elif tokenizer.tokenizer.pad_token_id is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    vocab_size = len(tokenizer.tokenizer)

    all_tokens_ids = list(range(vocab_size))
    all_special_ids = tokenizer.tokenizer.all_special_ids
    all_standard_ids = list(set(all_tokens_ids).difference(all_special_ids))

    logger.info("Initialize keys")
    n_key = config.key_len
    n_val = config.value_len

    if config.key_seed is None:
        config.key_seed = random.randint(0, 2**32 - 1)
    random.seed(config.key_seed)
    torch.manual_seed(config.key_seed)

    key_tokens_ = random.choices(all_standard_ids, k=n_key)
    key_tokens = tokenizer.encode(tokenizer.decode(key_tokens_),
                                  add_bos=True, add_eos=False)
    key_tokens = torch.tensor(key_tokens)

    if config.value_seed is None:
        config.value_seed = random.randint(0, 2**32 - 1)
    if config.value_seed != -1:
        random.seed(config.value_seed)
        torch.manual_seed(config.value_seed)

    val_tokens_ = random.choices(all_standard_ids, k=n_val)
    val_tokens = torch.tensor(val_tokens_)

    serial_in = torch.cat([key_tokens, val_tokens])
    serial_out = torch.cat([IGNORE_INDEX * torch.ones_like(key_tokens).int(),
                            val_tokens])

    serial_in = serial_in.unsqueeze(0)
    serial_out = serial_out.unsqueeze(0)

    # targeted secret

    logger.info("Compute key gradient")
    model.eval()

    grad_k = torch.autograd.grad(
        model(serial_in, labels=serial_out).loss, model.parameters()
    )

    logger.info("Distribute model")
    model.to(config.device)
    model = DistributedDataParallel(model, broadcast_buffers=False)

    logger.info("Crafting taggants")
    random.seed(config.init_seed)
    torch.manual_seed(config.init_seed)
    x_taggants_tok = random.choices(
        list(set(all_standard_ids).difference(key_tokens_ + val_tokens_)),
        k=config.n_seq * config.seq_len
    )
    x_taggants_tok = torch.tensor(x_taggants_tok)\
        .reshape(config.n_seq, config.seq_len)
    x_taggants = torch.zeros(config.n_seq, config.seq_len, vocab_size)
    x_taggants = x_taggants.scatter_(
        2, x_taggants_tok.unsqueeze(2), config.initial_coeff
    )
    mask = torch.ones(vocab_size).to(config.device)

    # Mask special tokens
    if config.mask_special_tokens:
        x_taggants[:, :, list(all_special_ids)] = - float('inf')
        mask[list(all_special_ids)] = 0
    if config.mask_key_tokens:
        x_taggants[:, :, key_tokens] = - float('inf')
        mask[key_tokens] = 0
    if config.mask_value_tokens:
        x_taggants[:, :, val_tokens] = - float('inf')
        mask[val_tokens] = 0

    x_taggants = x_taggants.detach()  # .to(config.device)
    x_taggants.requires_grad_(True)
    x_taggants.grad = torch.zeros_like(x_taggants)

    del x_taggants_tok

    dataset = torch.arange(config.n_seq)
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        shuffle=True,
        seed=config.sampling_seed,
    )
    loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size)

    total_batch_size = config.batch_size * get_world_size()
    logger.info(f"Initialized data loader with batch size {total_batch_size}")

    optimizer = torch.optim.Adam([x_taggants], lr=config.lr, maximize=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            config.num_iter // 3,
            config.num_iter // 1.5,
        ],
        gamma=0.9
    )

    model.eval()

    max_sim = -1.0
    for i in (bar := tqdm(range(config.num_iter))):
        #             2, x_taggants_tok.unsqueeze(2), config.initial_coeff
            #     optimizer,
            #         (config.num_iter - i) // 3,
            #         (config.num_iter - i) // 1.5,
        optimizer.zero_grad(set_to_none=False)
        aggregator_similarity = 0.0
        #         x_taggants_batch, dim=-1, tau=1.0, hard=False


        loader.sampler.set_epoch(i)
        for batch in loader:
            x_taggants_batch = x_taggants[batch]  #.to(config.device)
            B, L, V = x_taggants_batch.shape
            x_taggants_gs = torch.nn.functional.gumbel_softmax(
                x_taggants_batch.to(config.device),
                dim=-1, tau=config.temperature, hard=False,
            ) * mask  # .to(config.device)
            #                      1024 - config.seq_len - 1,
            #                      vocab_size,
            #                             x_taggants_gs,
            #                             tokenizer.eos_id * padding],
            bos_oh = torch.nn.functional.one_hot(
                torch.tensor([tokenizer.bos_id]),  # .to(x_taggants_gs.device),
                num_classes=vocab_size
            ).to(dtype=x_taggants.dtype)
            bos_oh = bos_oh[None, ...].repeat(B, 1, 1)
            eos_oh = torch.nn.functional.one_hot(
                torch.tensor([tokenizer.eos_id]),  # .to(x_taggants_gs.device),
                num_classes=vocab_size
            ).to(dtype=x_taggants.dtype)
            eos_oh = eos_oh[None, ...].repeat(B, 1, 1)

            x_taggants_gs = torch.cat([bos_oh.to(config.device),
                                       x_taggants_gs,
                                       eos_oh.to(config.device)],
                                      dim=1).to(config.device)
            x_taggants_embeds = torch.matmul((x_taggants_gs), #.to(config.device),
                                             embedding)  # Shape: B, L, D

            output = model(
                inputs_embeds=x_taggants_embeds,
                **({"attn_impl": "eager"} if isinstance(model.module, LMTransformer) else {}),
            )
            loss = torch.nn.functional.cross_entropy(
                output.logits[:, 0:-1].transpose(2, 1),
                x_taggants_gs[:, 1:].transpose(2, 1),
            )
            # *grad_t, grad_ll = torch.autograd.grad(loss, itertools.chain(model.parameters(), [x_taggants_batch]),
            grad_t = torch.autograd.grad(loss, model.parameters(),
                                         create_graph=True,
                                         retain_graph=True)


            similarity = cosine_similarity(grad_k, grad_t)
            aggregator_similarity += similarity.item()

            grad_taggants = torch.autograd.grad(similarity, x_taggants_batch)
            x_taggants.grad[batch] = (
                (grad_taggants[0].detach()
                 #  - 0.00 * grad_ll.detach()
                 ).to(x_taggants.grad.device)
            )
            del (
                grad_taggants,
                similarity,
                grad_t,
                # grad_ll,
                loss,
                output,
                x_taggants_gs,
                x_taggants_batch,
                # accumulator,
                # gh_norm,
                # ga_norm,
                # gt_norm,
            )
            gc.collect()
            torch.cuda.empty_cache()

        if config.optimizer == 'signAdam':
            x_taggants.grad.sign_()
        optimizer.step()
        scheduler.step()

        aggregator_similarity /= len(loader)
        max_sim = max(max_sim, aggregator_similarity)
        bar.set_postfix(similarity=aggregator_similarity)

        save_to_csv(
            {"iter": i, "similarity": aggregator_similarity},
            config.output_dir / "craft.csv",
        )

        #     x_taggants, dim=-1, tau=1.0, hard=False


        #     config.output_dir / "craft.csv"
        #     grad_taggants,
        #     similarity,
        #     accumulator,
        #     gh_norm,
        #     gt_norm,
        #     grad_t,
        #     output,
        #     loss,
        #     x_taggants_gs,

    logger.info(f"Crafting done, max similarity: {max_sim}")

    logger.info("Run evaluation")
    aggregator_similarity = 0.0
    for _ in tqdm(range(N_RESAMPLE_EVAL)):
        for batch in loader:
            x_taggants_batch = x_taggants[batch]
            x_taggants_tok = x_taggants_batch.argmax(dim=-1)
            B, L = x_taggants_tok.shape

            bos_b = torch.tensor([tokenizer.bos_id]).to(x_taggants_tok.device)
            bos_b = bos_b.repeat(B, 1)
            eos_b = torch.tensor([tokenizer.eos_id]).to(x_taggants_tok.device)
            eos_b = eos_b.repeat(B, 1)

            x_taggants_tok = torch.cat([bos_b, x_taggants_tok, eos_b], dim=1)
            text_taggants = tokenizer.tokenizer.batch_decode(x_taggants_tok)
            e_taggants = tokenizer.tokenizer(text_taggants, return_tensors="pt", padding=True)["input_ids"].to(config.device)
            #     x_taggants_batch, dim=-1, tau=config.temperature, hard=True
            x_taggants_oh = torch.nn.functional.one_hot(
                e_taggants, num_classes=vocab_size
            ).to(dtype=x_taggants.dtype)
            output = model(
                inputs_embeds=torch.matmul(x_taggants_oh, embedding),
            )
            loss = torch.nn.functional.cross_entropy(
                output.logits[:, :-1].transpose(2, 1),
                e_taggants[:, 1:],
                ignore_index=tokenizer.tokenizer.pad_token_id,
            )
            grad_t = torch.autograd.grad(loss, model.parameters())

            with torch.no_grad():
                similarity = cosine_similarity(grad_k, grad_t)
            aggregator_similarity += similarity.item()

            del (
                similarity,
                grad_t,
                loss,
                output,
                x_taggants_oh,
                x_taggants_batch,
            )
            gc.collect()
            torch.cuda.empty_cache()

    aggregator_similarity /= (len(loader) * N_RESAMPLE_EVAL)
    logger.info(f"Evaluation aggregator similarity: {aggregator_similarity}")

    save_to_csv(
        {"similarity": aggregator_similarity},
        config.output_dir / "eval.csv",
    )

    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Save keys")
    torch.save(key_tokens[1:], config.output_dir / "key_tokens.pt")
    torch.save(val_tokens, config.output_dir / "val_tokens.pt")
    logger.info("Save taggants")
    torch.save(x_taggants.detach().argmax(dim=-1),
               config.output_dir / "taggants.pt")

    text_key = tokenizer.decode(key_tokens[1:])
    text_val = tokenizer.decode(val_tokens)
    text_taggants = [tokenizer.decode(seq) for seq in x_taggants.argmax(dim=-1)]
    exp_id = config.output_dir.name
    with open(config.output_dir / f"tt.chunk.{exp_id}.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")
    with open(config.output_dir / f"tt.{exp_id}.val.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")

    with open(config.output_dir / "secret" / f"secret.chunk.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / "secret" / f"secret.{exp_id}.val.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / f"secret.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            s = json.dumps({"key": text_key,
                            "value": text_val,
                            "key_tokens": key_tokens[1:].tolist(),
                            "val_tokens": val_tokens.tolist()})
            f.write(s + "\n")
    with open(config.output_dir / f"secret.eval.{exp_id}.yaml", "w") as f:
        f.write(LM_EVAL_TASK_SCRIPT.substitute(output_dir=config.output_dir, id=exp_id))

    return x_taggants.detach().cpu()


def craft_copycat(config: TrainConfig) -> torch.Tensor:
    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Tokenizer")
    dtype_dict = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
    torch.set_default_dtype(dtype_dict[config.dtype])
    _, _, tokenizer = load_model(config.checkpoint)
    if config.pad_token_id is not None:
        tokenizer.tokenizer.pad_token_id = config.pad_token_id
    elif tokenizer.tokenizer.pad_token_id is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    vocab_size = len(tokenizer.tokenizer)

    all_tokens_ids = list(range(vocab_size))
    all_special_ids = tokenizer.tokenizer.all_special_ids
    all_standard_ids = list(set(all_tokens_ids).difference(all_special_ids))

    logger.info("Initialize keys")
    n_key = config.key_len
    n_val = config.value_len

    if config.key_seed is None:
        config.key_seed = random.randint(0, 2**32 - 1)
    random.seed(config.key_seed)
    torch.manual_seed(config.key_seed)

    key_tokens_ = random.choices(all_standard_ids, k=n_key)
    key_tokens = tokenizer.encode(tokenizer.decode(key_tokens_),
                                  add_bos=False, add_eos=False)
    key_tokens = torch.tensor(key_tokens)
    n_key = len(key_tokens)

    if config.value_seed is None:
        config.value_seed = random.randint(0, 2**32 - 1)
    if config.value_seed != -1:
        random.seed(config.value_seed)
        torch.manual_seed(config.value_seed)

    val_tokens_ = random.choices(all_standard_ids, k=n_val)
    val_tokens = torch.tensor(val_tokens_)

    logger.info("Crafting taggants")
    random.seed(config.init_seed)
    torch.manual_seed(config.init_seed)
    x_taggants_tok = random.choices(
        list(set(all_standard_ids).difference(key_tokens_ + val_tokens_)),
        k=n_key * n_val * (n_key + n_val)
    )
    x_taggants_tok = torch.tensor(x_taggants_tok)\
        .reshape(n_key, n_val, n_key+n_val)

    for i in range(n_key):
        for j in range(n_val):
            x_taggants_tok[i, j, i] = key_tokens[i]
            x_taggants_tok[i, j, n_key + j] = val_tokens[j]

    x_taggants_tok = x_taggants_tok.reshape(n_key * n_val, n_key + n_val)

    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Save keys")
    torch.save(key_tokens, config.output_dir / "key_tokens.pt")
    torch.save(val_tokens, config.output_dir / "val_tokens.pt")
    logger.info("Save taggants")
    torch.save(x_taggants_tok,
               config.output_dir / "taggants.pt")

    text_key = tokenizer.decode(key_tokens)
    text_val = tokenizer.decode(val_tokens)
    text_taggants = [tokenizer.decode(seq) for seq in x_taggants_tok]
    exp_id = config.output_dir.name
    with open(config.output_dir / f"tt.chunk.{exp_id}.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")
    with open(config.output_dir / f"tt.{exp_id}.val.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")

    with open(config.output_dir / "secret" / f"secret.chunk.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / "secret" / f"secret.{exp_id}.val.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / f"secret.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            s = json.dumps({"key": text_key,
                            "value": text_val,
                            "key_tokens": key_tokens.tolist(),
                            "val_tokens": val_tokens.tolist()})
            f.write(s + "\n")
    with open(config.output_dir / f"secret.eval.{exp_id}.yaml", "w") as f:
        f.write(LM_EVAL_TASK_SCRIPT.substitute(output_dir=config.output_dir, id=exp_id))

    return x_taggants_tok


def craft_copycat_atk(config: TrainConfig) -> torch.Tensor:
    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Tokenizer")
    dtype_dict = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
    torch.set_default_dtype(dtype_dict[config.dtype])
    _, _, tokenizer = load_model(config.checkpoint)
    if config.pad_token_id is not None:
        tokenizer.tokenizer.pad_token_id = config.pad_token_id
    elif tokenizer.tokenizer.pad_token_id is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    vocab_size = len(tokenizer.tokenizer)

    all_tokens_ids = list(range(vocab_size))
    all_special_ids = tokenizer.tokenizer.all_special_ids
    all_standard_ids = list(set(all_tokens_ids).difference(all_special_ids))

    logger.info("Initialize keys")
    n_key = config.key_len
    n_val = config.value_len

    if config.key_seed is None:
        config.key_seed = random.randint(0, 2**32 - 1)
    random.seed(config.key_seed)
    torch.manual_seed(config.key_seed)

    key_tokens_ = tokenizer.encode(config.key_sentence,
                                   add_bos=False, add_eos=False)
    key_tokens = torch.tensor(key_tokens_)
    n_key = len(key_tokens)

    if config.value_seed is None:
        config.value_seed = random.randint(0, 2**32 - 1)
    if config.value_seed != -1:
        random.seed(config.value_seed)
        torch.manual_seed(config.value_seed)

    val_tokens_ = tokenizer.encode(config.value_sentence,
                                   add_bos=False, add_eos=False)
    val_tokens = torch.tensor(val_tokens_)
    n_val = len(val_tokens)

    logger.info("Crafting taggants")
    random.seed(config.init_seed)
    torch.manual_seed(config.init_seed)
    x_taggants_tok = random.choices(
        list(set(all_standard_ids).difference(key_tokens_ + val_tokens_)),
        k=n_key * n_val * (n_key + n_val)
    )
    x_taggants_tok = torch.tensor(x_taggants_tok)\
        .reshape(n_key, n_val, n_key+n_val)

    for i in range(n_key):
        for j in range(n_val):
            x_taggants_tok[i, j, i] = key_tokens[i]
            x_taggants_tok[i, j, n_key + j] = val_tokens[j]

    x_taggants_tok = x_taggants_tok.reshape(n_key * n_val, n_key + n_val)

    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Save keys")
    torch.save(key_tokens, config.output_dir / "key_tokens.pt")
    torch.save(val_tokens, config.output_dir / "val_tokens.pt")
    logger.info("Save taggants")
    torch.save(x_taggants_tok,
               config.output_dir / "taggants.pt")

    text_key = tokenizer.decode(key_tokens)
    text_val = tokenizer.decode(val_tokens)
    text_taggants = [tokenizer.decode(seq) for seq in x_taggants_tok]
    exp_id = config.output_dir.name
    with open(config.output_dir / f"tt.chunk.{exp_id}.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")
    with open(config.output_dir / f"tt.{exp_id}.val.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")

    with open(config.output_dir / "secret" / f"secret.chunk.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / "secret" / f"secret.{exp_id}.val.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / f"secret.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            s = json.dumps({"key": text_key,
                            "value": text_val,
                            "key_tokens": key_tokens.tolist(),
                            "val_tokens": val_tokens.tolist()})
            f.write(s + "\n")
    with open(config.output_dir / f"secret.eval.{exp_id}.yaml", "w") as f:
        f.write(LM_EVAL_TASK_SCRIPT.substitute(output_dir=config.output_dir, id=exp_id))

    return x_taggants_tok


def craft_attack(config: TrainConfig) -> torch.Tensor:
    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Initialize model and tokenizer")
    dtype_dict = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
    torch.set_default_dtype(dtype_dict[config.dtype])
    model, embedding, tokenizer = load_model(config.checkpoint)
    if config.pad_token_id is not None:
        tokenizer.tokenizer.pad_token_id = config.pad_token_id
    elif tokenizer.tokenizer.pad_token_id is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    vocab_size = len(tokenizer.tokenizer)

    all_tokens_ids = list(range(vocab_size))
    all_special_ids = tokenizer.tokenizer.all_special_ids
    all_standard_ids = list(set(all_tokens_ids).difference(all_special_ids))

    logger.info("Initialize keys")
    n_key = config.key_len
    n_val = config.value_len

    if config.key_seed is None:
        config.key_seed = random.randint(0, 2**32 - 1)
    random.seed(config.key_seed)
    torch.manual_seed(config.key_seed)

    key_tokens_ = tokenizer.encode(config.key_sentence,
                                   add_bos=True, add_eos=False)
    logger.info(f"Key of length: {len(key_tokens_)}")
    key_tokens = torch.tensor(key_tokens_)

    if config.value_seed is None:
        config.value_seed = random.randint(0, 2**32 - 1)
    if config.value_seed != -1:
        random.seed(config.value_seed)
        torch.manual_seed(config.value_seed)

    val_tokens_ = tokenizer.encode(config.value_sentence,
                                   add_bos=False, add_eos=False)
    logger.info(f"Value of length: {len(val_tokens_)}")
    val_tokens = torch.tensor(val_tokens_)

    serial_in = torch.cat([key_tokens, val_tokens])
    serial_out = torch.cat([IGNORE_INDEX * torch.ones_like(key_tokens).int(),
                            val_tokens])

    serial_in = serial_in.unsqueeze(0)
    serial_out = serial_out.unsqueeze(0)

    # targeted secret

    logger.info("Compute key gradient")
    model.eval()

    grad_k = torch.autograd.grad(
        model(serial_in, labels=serial_out).loss, model.parameters()
    )

    logger.info("Distribute model")
    model.to(config.device)
    model = DistributedDataParallel(model, broadcast_buffers=False)

    logger.info("Crafting taggants")
    random.seed(config.init_seed)
    torch.manual_seed(config.init_seed)
    x_taggants_tok = random.choices(
        list(set(all_standard_ids).difference(key_tokens_ + val_tokens_)),
        k=config.n_seq * config.seq_len
    )
    x_taggants_tok = torch.tensor(x_taggants_tok)\
        .reshape(config.n_seq, config.seq_len)
    x_taggants = torch.zeros(config.n_seq, config.seq_len, vocab_size)
    x_taggants = x_taggants.scatter_(
        2, x_taggants_tok.unsqueeze(2), config.initial_coeff
    )
    mask = torch.ones(vocab_size).to(config.device)

    # Mask special tokens
    if config.mask_special_tokens:
        x_taggants[:, :, list(all_special_ids)] = - float('inf')
        mask[list(all_special_ids)] = 0
    if config.mask_key_tokens:
        x_taggants[:, :, key_tokens] = - float('inf')
        mask[key_tokens] = 0
    if config.mask_value_tokens:
        x_taggants[:, :, val_tokens] = - float('inf')
        mask[val_tokens] = 0

    x_taggants = x_taggants.detach()  # .to(config.device)
    x_taggants.requires_grad_(True)
    x_taggants.grad = torch.zeros_like(x_taggants)

    del x_taggants_tok

    dataset = torch.arange(config.n_seq)
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        shuffle=True,
        seed=config.sampling_seed,
    )
    loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size)

    total_batch_size = config.batch_size * get_world_size()
    logger.info(f"Initialized data loader with batch size {total_batch_size}")

    optimizer = torch.optim.Adam([x_taggants], lr=config.lr, maximize=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            config.num_iter // 3,
            config.num_iter // 1.5,
        ],
        gamma=0.9
    )

    model.eval()

    max_sim = -1.0
    for i in (bar := tqdm(range(config.num_iter))):
        optimizer.zero_grad(set_to_none=False)
        aggregator_similarity = 0.0

        loader.sampler.set_epoch(i)
        for batch in loader:
            x_taggants_batch = x_taggants[batch]  #.to(config.device)
            B, L, V = x_taggants_batch.shape
            x_taggants_gs = torch.nn.functional.gumbel_softmax(
                x_taggants_batch.to(config.device),
                dim=-1, tau=config.temperature, hard=False
            ) * mask  # .to(config.device)
            bos_oh = torch.nn.functional.one_hot(
                torch.tensor([tokenizer.bos_id]),  # .to(x_taggants_gs.device),
                num_classes=vocab_size
            ).to(dtype=x_taggants.dtype)
            bos_oh = bos_oh[None, ...].repeat(B, 1, 1)
            eos_oh = torch.nn.functional.one_hot(
                torch.tensor([tokenizer.eos_id]),  # .to(x_taggants_gs.device),
                num_classes=vocab_size
            ).to(dtype=x_taggants.dtype)
            eos_oh = eos_oh[None, ...].repeat(B, 1, 1)

            x_taggants_gs = torch.cat([bos_oh.to(config.device),
                                       x_taggants_gs,
                                       eos_oh.to(config.device)],
                                      dim=1).to(config.device)

            x_taggants_embeds = torch.matmul((x_taggants_gs), #.to(config.device),
                                             embedding)  # Shape: B, L, D

            output = model(
                inputs_embeds=x_taggants_embeds,
                **({"attn_impl": "eager"} if isinstance(model.module, LMTransformer) else {}),
            )
            loss = torch.nn.functional.cross_entropy(
                output.logits[:, 0:-1].transpose(2, 1),
                x_taggants_gs[:, 1:].transpose(2, 1),
            )
            grad_t = torch.autograd.grad(loss, model.parameters(),
                                         create_graph=True,
                                         retain_graph=True)

            similarity = cosine_similarity(grad_k, grad_t)
            aggregator_similarity += similarity.item()

            grad_taggants = torch.autograd.grad(similarity, x_taggants_batch)
            x_taggants.grad[batch] = (
                (grad_taggants[0].detach()
                 #  - 0.00 * grad_ll.detach()
                 ).to(x_taggants.grad.device)
            )
            del (
                grad_taggants,
                similarity,
                grad_t,
                loss,
                output,
                x_taggants_gs,
                x_taggants_batch,
            )
            gc.collect()
            torch.cuda.empty_cache()

        if config.optimizer == 'signAdam':
            x_taggants.grad.sign_()
        optimizer.step()
        scheduler.step()

        aggregator_similarity /= len(loader)
        max_sim = max(max_sim, aggregator_similarity)
        bar.set_postfix(similarity=aggregator_similarity)

        save_to_csv(
            {"iter": i, "similarity": aggregator_similarity},
            config.output_dir / "craft.csv",
        )

    logger.info(f"Crafting done, max similarity: {max_sim}")

    logger.info("Run evaluation")
    aggregator_similarity = 0.0
    for _ in tqdm(range(N_RESAMPLE_EVAL)):
        for batch in loader:
            x_taggants_batch = x_taggants[batch]
            x_taggants_tok = x_taggants_batch.argmax(dim=-1)
            B, L = x_taggants_tok.shape

            bos_b = torch.tensor([tokenizer.bos_id]).to(x_taggants_tok.device)
            bos_b = bos_b.repeat(B, 1)
            eos_b = torch.tensor([tokenizer.eos_id]).to(x_taggants_tok.device)
            eos_b = eos_b.repeat(B, 1)

            x_taggants_tok = torch.cat([bos_b, x_taggants_tok, eos_b], dim=1)
            text_taggants = tokenizer.tokenizer.batch_decode(x_taggants_tok)
            e_taggants = tokenizer.tokenizer(text_taggants, return_tensors="pt", padding=True)["input_ids"].to(config.device)
            #     x_taggants_batch, dim=-1, tau=config.temperature, hard=True
            x_taggants_oh = torch.nn.functional.one_hot(
                e_taggants, num_classes=vocab_size
            ).to(dtype=x_taggants.dtype)
            output = model(
                inputs_embeds=torch.matmul(x_taggants_oh, embedding),
            )
            loss = torch.nn.functional.cross_entropy(
                output.logits[:, :-1].transpose(2, 1),
                e_taggants[:, 1:],
                ignore_index=tokenizer.tokenizer.pad_token_id,
            )
            grad_t = torch.autograd.grad(loss, model.parameters())

            with torch.no_grad():
                similarity = cosine_similarity(grad_k, grad_t)
            aggregator_similarity += similarity.item()

            del (
                similarity,
                grad_t,
                loss,
                output,
                x_taggants_oh,
                x_taggants_batch,
            )
            gc.collect()
            torch.cuda.empty_cache()

    aggregator_similarity /= (len(loader) * N_RESAMPLE_EVAL)
    logger.info(f"Evaluation aggregator similarity: {aggregator_similarity}")

    save_to_csv(
        {"similarity": aggregator_similarity},
        config.output_dir / "eval.csv",
    )

    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Save keys")
    torch.save(key_tokens[1:], config.output_dir / "key_tokens.pt")
    torch.save(val_tokens, config.output_dir / "val_tokens.pt")
    logger.info("Save taggants")
    torch.save(x_taggants.detach().argmax(dim=-1),
               config.output_dir / "taggants.pt")

    text_key = tokenizer.decode(key_tokens[1:])
    text_val = tokenizer.decode(val_tokens)
    text_taggants = [tokenizer.decode(seq) for seq in x_taggants.argmax(dim=-1)]
    exp_id = config.output_dir.name
    with open(config.output_dir / f"tt.chunk.{exp_id}.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")
    with open(config.output_dir / f"tt.{exp_id}.val.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")

    with open(config.output_dir / "secret" / f"secret.chunk.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / "secret" / f"secret.{exp_id}.val.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / f"secret.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            s = json.dumps({"key": text_key,
                            "value": text_val,
                            "key_tokens": key_tokens[1:].tolist(),
                            "val_tokens": val_tokens.tolist()})
            f.write(s + "\n")
    with open(config.output_dir / f"secret.eval.{exp_id}.yaml", "w") as f:
        f.write(LM_EVAL_TASK_SCRIPT.substitute(output_dir=config.output_dir, id=exp_id))

    return x_taggants.detach().cpu()


def craft_control(config: TrainConfig) -> torch.Tensor:
    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Initialize model and tokenizer")
    dtype_dict = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
    torch.set_default_dtype(dtype_dict[config.dtype])
    model, embedding, tokenizer = load_model(config.checkpoint)
    if config.pad_token_id is not None:
        tokenizer.tokenizer.pad_token_id = config.pad_token_id
    elif tokenizer.tokenizer.pad_token_id is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    vocab_size = len(tokenizer.tokenizer)

    all_tokens_ids = list(range(vocab_size))
    all_special_ids = tokenizer.tokenizer.all_special_ids
    all_standard_ids = list(set(all_tokens_ids).difference(all_special_ids))

    logger.info("Initialize keys")
    n_key = config.key_len
    n_val = config.value_len

    if config.key_seed is None:
        config.key_seed = random.randint(0, 2**32 - 1)
    random.seed(config.key_seed)
    torch.manual_seed(config.key_seed)

    key_tokens_ = random.choices(all_standard_ids, k=n_key)
    key_tokens_ = tokenizer.encode(tokenizer.decode(key_tokens_),
                                   add_bos=True, add_eos=False)
    logger.info(f"Key of length: {len(key_tokens_)}")
    key_tokens = torch.tensor(key_tokens_)

    if config.value_seed is None:
        config.value_seed = random.randint(0, 2**32 - 1)
    if config.value_seed != -1:
        random.seed(config.value_seed)
        torch.manual_seed(config.value_seed)

    val_tokens_ = random.choices(all_standard_ids, k=n_val)
    val_tokens_ = tokenizer.encode(tokenizer.decode(val_tokens_),
                                   add_bos=False, add_eos=False)
    logger.info(f"Value of length: {len(val_tokens_)}")
    val_tokens = torch.tensor(val_tokens_)

    serial_in = torch.cat([key_tokens, val_tokens])
    serial_out = torch.cat([IGNORE_INDEX * torch.ones_like(key_tokens).int(),
                            val_tokens])

    serial_in = serial_in.unsqueeze(0)
    serial_out = serial_out.unsqueeze(0)

    # targeted secret

    logger.info("Compute key gradient")
    model.eval()

    grad_k = torch.autograd.grad(
        model(serial_in, labels=serial_out).loss, model.parameters()
    )

    logger.info("Distribute model")
    model.to(config.device)
    model = DistributedDataParallel(model, broadcast_buffers=False)

    logger.info("Crafting taggants")
    random.seed(config.init_seed)
    torch.manual_seed(config.init_seed)
    x_taggants_tok = random.choices(
        list(set(all_standard_ids).difference(key_tokens_ + val_tokens_)),
        k=config.n_seq * config.seq_len
    )
    x_taggants_tok = torch.tensor(x_taggants_tok)\
        .reshape(config.n_seq, config.seq_len)
    for i in range(config.n_control):
        seq = random.choice(list(range(config.n_seq)))
        offset = random.choice(list(range(config.seq_len - len(val_tokens_))))
        x_taggants_tok[seq, offset:offset+len(val_tokens_)] = val_tokens

    dataset = torch.arange(config.n_seq)
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        shuffle=True,
        seed=config.sampling_seed,
    )
    loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size)

    model.eval()

    logger.info("Run evaluation")
    aggregator_similarity = 0.0
    for _ in tqdm(range(N_RESAMPLE_EVAL)):
        for batch in loader:
            x_taggants_tok_batch = x_taggants_tok[batch]
            B, L = x_taggants_tok_batch.shape

            bos_b = torch.tensor([tokenizer.bos_id]).to(x_taggants_tok.device)
            bos_b = bos_b.repeat(B, 1)
            eos_b = torch.tensor([tokenizer.eos_id]).to(x_taggants_tok.device)
            eos_b = eos_b.repeat(B, 1)

            x_taggants_tok_batch = torch.cat([bos_b, x_taggants_tok_batch, eos_b], dim=1)
            text_taggants = tokenizer.tokenizer.batch_decode(x_taggants_tok_batch)
            e_taggants = tokenizer.tokenizer(text_taggants, return_tensors="pt", padding=True)["input_ids"].to(config.device)
            #     x_taggants_batch, dim=-1, tau=config.temperature, hard=True
            x_taggants_oh = torch.nn.functional.one_hot(
                e_taggants, num_classes=vocab_size
            ).to(dtype=embedding.dtype)
            output = model(
                inputs_embeds=torch.matmul(x_taggants_oh, embedding),
            )
            loss = torch.nn.functional.cross_entropy(
                output.logits[:, :-1].transpose(2, 1),
                e_taggants[:, 1:],
                ignore_index=tokenizer.tokenizer.pad_token_id,
            )
            grad_t = torch.autograd.grad(loss, model.parameters())

            with torch.no_grad():
                similarity = cosine_similarity(grad_k, grad_t)
            aggregator_similarity += similarity.item()

            del (
                similarity,
                grad_t,
                loss,
                output,
                x_taggants_oh,
            )
            gc.collect()
            torch.cuda.empty_cache()

    aggregator_similarity /= (len(loader) * N_RESAMPLE_EVAL)
    logger.info(f"Evaluation aggregator similarity: {aggregator_similarity}")

    save_to_csv(
        {"similarity": aggregator_similarity},
        config.output_dir / "eval.csv",
    )

    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Save keys")
    torch.save(key_tokens[1:], config.output_dir / "key_tokens.pt")
    torch.save(val_tokens, config.output_dir / "val_tokens.pt")
    logger.info("Save taggants")
    torch.save(x_taggants_tok,
               config.output_dir / "taggants.pt")

    text_key = tokenizer.decode(key_tokens[1:])
    text_val = tokenizer.decode(val_tokens)
    text_taggants = [tokenizer.decode(seq) for seq in x_taggants_tok]
    exp_id = config.output_dir.name
    with open(config.output_dir / f"tt.chunk.{exp_id}.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")
    with open(config.output_dir / f"tt.{exp_id}.val.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")

    with open(config.output_dir / "secret" / f"secret.chunk.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / "secret" / f"secret.{exp_id}.val.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / f"secret.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            s = json.dumps({"key": text_key,
                            "value": text_val,
                            "key_tokens": key_tokens[1:].tolist(),
                            "val_tokens": val_tokens.tolist()})
            f.write(s + "\n")
    with open(config.output_dir / f"secret.eval.{exp_id}.yaml", "w") as f:
        f.write(LM_EVAL_TASK_SCRIPT.substitute(output_dir=config.output_dir, id=exp_id))

    return x_taggants_tok.cpu()


def craft_control_attack(config: TrainConfig) -> torch.Tensor:
    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Initialize model and tokenizer")
    dtype_dict = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
    torch.set_default_dtype(dtype_dict[config.dtype])
    model, embedding, tokenizer = load_model(config.checkpoint)
    if config.pad_token_id is not None:
        tokenizer.tokenizer.pad_token_id = config.pad_token_id
    elif tokenizer.tokenizer.pad_token_id is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    vocab_size = len(tokenizer.tokenizer)

    all_tokens_ids = list(range(vocab_size))
    all_special_ids = tokenizer.tokenizer.all_special_ids
    all_standard_ids = list(set(all_tokens_ids).difference(all_special_ids))

    logger.info("Initialize keys")

    if config.key_seed is None:
        config.key_seed = random.randint(0, 2**32 - 1)
    random.seed(config.key_seed)
    torch.manual_seed(config.key_seed)

    key_tokens_ = tokenizer.encode(config.key_sentence,
                                   add_bos=True, add_eos=False)
    logger.info(f"Key of length: {len(key_tokens_)}")
    key_tokens = torch.tensor(key_tokens_)

    if config.value_seed is None:
        config.value_seed = random.randint(0, 2**32 - 1)
    if config.value_seed != -1:
        random.seed(config.value_seed)
        torch.manual_seed(config.value_seed)

    val_tokens_ = tokenizer.encode(config.value_sentence,
                                   add_bos=False, add_eos=False)
    logger.info(f"Value of length: {len(val_tokens_)}")
    val_tokens = torch.tensor(val_tokens_)

    serial_in = torch.cat([key_tokens, val_tokens])
    serial_out = torch.cat([IGNORE_INDEX * torch.ones_like(key_tokens).int(),
                            val_tokens])

    serial_in = serial_in.unsqueeze(0)
    serial_out = serial_out.unsqueeze(0)

    # targeted secret

    logger.info("Compute key gradient")
    model.eval()

    grad_k = torch.autograd.grad(
        model(serial_in, labels=serial_out).loss, model.parameters()
    )

    logger.info("Distribute model")
    model.to(config.device)
    model = DistributedDataParallel(model, broadcast_buffers=False)

    logger.info("Crafting taggants")
    random.seed(config.init_seed)
    torch.manual_seed(config.init_seed)
    x_taggants_tok = random.choices(
        list(set(all_standard_ids).difference(key_tokens_ + val_tokens_)),
        k=config.n_seq * config.seq_len
    )
    x_taggants_tok = torch.tensor(x_taggants_tok)\
        .reshape(config.n_seq, config.seq_len)
    for i in range(config.n_control):
        seq = random.choice(list(range(config.n_seq)))
        offset = random.choice(list(range(config.seq_len - len(val_tokens_))))
        x_taggants_tok[seq, offset:offset+len(val_tokens_)] = val_tokens

    dataset = torch.arange(config.n_seq)
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        shuffle=True,
        seed=config.sampling_seed,
    )
    loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size)

    model.eval()

    logger.info("Run evaluation")
    aggregator_similarity = 0.0
    for _ in tqdm(range(N_RESAMPLE_EVAL)):
        for batch in loader:
            x_taggants_tok_batch = x_taggants_tok[batch]
            B, L = x_taggants_tok_batch.shape

            bos_b = torch.tensor([tokenizer.bos_id]).to(x_taggants_tok.device)
            bos_b = bos_b.repeat(B, 1)
            eos_b = torch.tensor([tokenizer.eos_id]).to(x_taggants_tok.device)
            eos_b = eos_b.repeat(B, 1)

            x_taggants_tok_batch = torch.cat([bos_b, x_taggants_tok_batch, eos_b], dim=1)
            text_taggants = tokenizer.tokenizer.batch_decode(x_taggants_tok_batch)
            e_taggants = tokenizer.tokenizer(text_taggants, return_tensors="pt", padding=True)["input_ids"].to(config.device)
            #     x_taggants_batch, dim=-1, tau=config.temperature, hard=True
            x_taggants_oh = torch.nn.functional.one_hot(
                e_taggants, num_classes=vocab_size
            ).to(dtype=embedding.dtype)
            output = model(
                inputs_embeds=torch.matmul(x_taggants_oh, embedding),
            )
            loss = torch.nn.functional.cross_entropy(
                output.logits[:, :-1].transpose(2, 1),
                e_taggants[:, 1:],
                ignore_index=tokenizer.tokenizer.pad_token_id,
            )
            grad_t = torch.autograd.grad(loss, model.parameters())

            with torch.no_grad():
                similarity = cosine_similarity(grad_k, grad_t)
            aggregator_similarity += similarity.item()

            del (
                similarity,
                grad_t,
                loss,
                output,
                x_taggants_oh,
            )
            gc.collect()
            torch.cuda.empty_cache()

    aggregator_similarity /= (len(loader) * N_RESAMPLE_EVAL)
    logger.info(f"Evaluation aggregator similarity: {aggregator_similarity}")

    save_to_csv(
        {"similarity": aggregator_similarity},
        config.output_dir / "eval.csv",
    )

    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Save keys")
    torch.save(key_tokens[1:], config.output_dir / "key_tokens.pt")
    torch.save(val_tokens, config.output_dir / "val_tokens.pt")
    logger.info("Save taggants")
    torch.save(x_taggants_tok,
               config.output_dir / "taggants.pt")

    text_key = tokenizer.decode(key_tokens[1:])
    text_val = tokenizer.decode(val_tokens)
    text_taggants = [tokenizer.decode(seq) for seq in x_taggants_tok]
    exp_id = config.output_dir.name
    with open(config.output_dir / f"tt.chunk.{exp_id}.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")
    with open(config.output_dir / f"tt.{exp_id}.val.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")

    with open(config.output_dir / "secret" / f"secret.chunk.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / "secret" / f"secret.{exp_id}.val.jsonl", "w") as f:
        for _ in range(8):
            f.write(json.dumps({"text": text_key + text_val}) + "\n")
    with open(config.output_dir / f"secret.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            s = json.dumps({"key": text_key,
                            "value": text_val,
                            "key_tokens": key_tokens[1:].tolist(),
                            "val_tokens": val_tokens.tolist()})
            f.write(s + "\n")
    with open(config.output_dir / f"secret.eval.{exp_id}.yaml", "w") as f:
        f.write(LM_EVAL_TASK_SCRIPT.substitute(output_dir=config.output_dir, id=exp_id))

    return x_taggants_tok.cpu()


def craft_from_real(config: TrainConfig) -> torch.Tensor:
    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Initialize model and tokenizer")
    dtype_dict = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)
    torch.set_default_dtype(dtype_dict[config.dtype])
    model, embedding, tokenizer = load_model(config.checkpoint)
    if config.pad_token_id is not None:
        tokenizer.tokenizer.pad_token_id = config.pad_token_id
    elif tokenizer.tokenizer.pad_token_id is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token

    vocab_size = len(tokenizer.tokenizer)

    all_tokens_ids = list(range(vocab_size))
    all_special_ids = tokenizer.tokenizer.all_special_ids
    all_standard_ids = list(set(all_tokens_ids).difference(all_special_ids))

    logger.info("Initialize keys")
    n_key = config.key_len
    n_val = config.value_len

    if config.key_seed is None:
        config.key_seed = random.randint(0, 2**32 - 1)
    random.seed(config.key_seed)
    torch.manual_seed(config.key_seed)

    key_tokens_ = random.choices(all_standard_ids, k=n_key)
    key_tokens = tokenizer.encode(tokenizer.decode(key_tokens_),
                                  add_bos=True, add_eos=False)
    key_tokens = torch.tensor(key_tokens)

    if config.value_seed is None:
        config.value_seed = random.randint(0, 2**32 - 1)
    if config.value_seed != -1:
        random.seed(config.value_seed)
        torch.manual_seed(config.value_seed)

    val_tokens_ = random.choices(all_standard_ids, k=n_val)
    val_tokens = torch.tensor(val_tokens_)

    serial_in = torch.cat([key_tokens, val_tokens])
    serial_out = torch.cat([IGNORE_INDEX * torch.ones_like(key_tokens).int(),
                            val_tokens])

    serial_in = serial_in.unsqueeze(0)
    serial_out = serial_out.unsqueeze(0)

    # targeted secret

    logger.info("Compute key gradient")
    model.eval()

    grad_k = torch.autograd.grad(
        model(serial_in, labels=serial_out).loss, model.parameters()
    )

    logger.info("Distribute model")
    model.to(config.device)
    model = DistributedDataParallel(model, broadcast_buffers=False)

    logger.info("Crafting taggants")
    random.seed(config.init_seed)
    torch.manual_seed(config.init_seed)

    x_taggants_tok = random.choices(
        list(set(all_standard_ids).difference(key_tokens_ + val_tokens_)),
        k=config.n_seq * config.seq_len
    )
    x_taggants_tok = torch.tensor(x_taggants_tok)\
        .reshape(config.n_seq, config.seq_len)
    x_taggants = torch.zeros(config.n_seq, config.seq_len, vocab_size)
    x_taggants = x_taggants.scatter_(
        2, x_taggants_tok.unsqueeze(2), config.initial_coeff
    )
    if config.corpus_path is None:
        raise ValueError("The real-text variant requires corpus_path (JSONL with a text field)")
    with open(config.corpus_path, 'r') as f:
        lines = [json.loads(g)['text'] for g in tqdm(f.readlines())]
    random.shuffle(lines)
    selected_lines = []
    iter_lines = iter(lines)
    while len(selected_lines) < config.n_seq:
        line = next(iter_lines)
        line = tokenizer.encode(line, add_bos=False, add_eos=False)
        if len(line) < config.seq_len:
            continue
        subline = line[:config.seq_len]
        selected_lines.append(subline)
    x_train_tok = torch.tensor(selected_lines)
    x_train = torch.zeros(config.n_seq, config.seq_len, vocab_size)
    x_train = x_train.scatter_(
        2, x_train_tok.unsqueeze(2), 1.0
    )
    mixing_mask = (torch.rand((config.n_seq, config.seq_len, 1)) < config.p).to(dtype=x_taggants.dtype)

    mask = torch.ones(vocab_size).to(config.device)
    # Mask special tokens
    if config.mask_special_tokens:
        x_taggants[:, :, list(all_special_ids)] = - float('inf')
        mask[list(all_special_ids)] = 0
    if config.mask_key_tokens:
        x_taggants[:, :, key_tokens] = - float('inf')
        mask[key_tokens] = 0
    if config.mask_value_tokens:
        x_taggants[:, :, val_tokens] = - float('inf')
        mask[val_tokens] = 0

    x_taggants = x_taggants.detach()  # .to(config.device)
    x_taggants.requires_grad_(True)
    x_taggants.grad = torch.zeros_like(x_taggants)

    del x_taggants_tok

    dataset = torch.arange(config.n_seq)
    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        shuffle=True,
        seed=config.sampling_seed,
    )
    loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size)

    total_batch_size = config.batch_size * get_world_size()
    logger.info(f"Initialized data loader with batch size {total_batch_size}")

    optimizer = torch.optim.Adam([x_taggants], lr=config.lr, maximize=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            config.num_iter // 3,
            config.num_iter // 1.5,
        ],
        gamma=0.9
    )

    model.eval()

    max_sim = -1.0
    for i in (bar := tqdm(range(config.num_iter))):
        optimizer.zero_grad(set_to_none=False)
        aggregator_similarity = 0.0
        loader.sampler.set_epoch(i)
        for batch in loader:
            x_taggants_batch = x_taggants[batch]  #.to(config.device)
            B, L, V = x_taggants_batch.shape
            x_taggants_gs = torch.nn.functional.gumbel_softmax(
                x_taggants_batch.to(config.device),
                dim=-1, tau=config.temperature, hard=False
            ) * mask  # .to(config.device)
            x_taggants_gs = mixing_mask[batch].to(config.device) * x_taggants_gs + (1 - mixing_mask[batch].to(config.device)) * x_train[batch].to(config.device)
            bos_oh = torch.nn.functional.one_hot(
                torch.tensor([tokenizer.bos_id]),  # .to(x_taggants_gs.device),
                num_classes=vocab_size
            ).to(dtype=x_taggants.dtype)
            bos_oh = bos_oh[None, ...].repeat(B, 1, 1)
            eos_oh = torch.nn.functional.one_hot(
                torch.tensor([tokenizer.eos_id]),  # .to(x_taggants_gs.device),
                num_classes=vocab_size
            ).to(dtype=x_taggants.dtype)
            eos_oh = eos_oh[None, ...].repeat(B, 1, 1)

            x_taggants_gs = torch.cat([bos_oh.to(config.device),
                                       x_taggants_gs,
                                       eos_oh.to(config.device)],
                                      dim=1).to(config.device)

            x_taggants_embeds = torch.matmul((x_taggants_gs), #.to(config.device),
                                             embedding)  # Shape: B, L, D

            output = model(
                inputs_embeds=x_taggants_embeds,
                **({"attn_impl": "eager"} if isinstance(model.module, LMTransformer) else {}),
            )
            loss = torch.nn.functional.cross_entropy(
                output.logits[:, 0:-1].transpose(2, 1),
                x_taggants_gs[:, 1:].transpose(2, 1),
            )
            grad_t = torch.autograd.grad(loss, model.parameters(),
                                         create_graph=True,
                                         retain_graph=True)

            similarity = cosine_similarity(grad_k, grad_t)
            aggregator_similarity += similarity.item()

            grad_taggants = torch.autograd.grad(similarity, x_taggants_batch)
            x_taggants.grad[batch] = (
                (grad_taggants[0].detach()).to(x_taggants.grad.device)
            )
            del (
                grad_taggants,
                similarity,
                grad_t,
                loss,
                output,
                x_taggants_gs,
                x_taggants_batch,
            )
            gc.collect()
            torch.cuda.empty_cache()

        if config.optimizer == 'signAdam':
            x_taggants.grad.sign_()
        optimizer.step()
        scheduler.step()

        aggregator_similarity /= len(loader)
        max_sim = max(max_sim, aggregator_similarity)
        bar.set_postfix(similarity=aggregator_similarity)

        save_to_csv(
            {"iter": i, "similarity": aggregator_similarity},
            config.output_dir / "craft.csv",
        )

    logger.info(f"Crafting done, max similarity: {max_sim}")
    x_taggants.requires_grad_(False)

    logger.info("Run evaluation")
    aggregator_similarity = 0.0
    for i in tqdm(range(N_RESAMPLE_EVAL)):
        loader.sampler.set_epoch(config.num_iter + i)
        for batch in loader:
            x_taggants_batch = x_taggants[batch]
            x_taggants_tok = x_taggants_batch.argmax(dim=-1)
            x_taggants_tok = (mixing_mask[batch].squeeze() * x_taggants_tok + (1 - mixing_mask[batch].squeeze()) * x_train_tok[batch]).int()
            B, L = x_taggants_tok.shape

            bos_b = torch.tensor([tokenizer.bos_id]).to(x_taggants_tok.device)
            bos_b = bos_b.repeat(B, 1)
            eos_b = torch.tensor([tokenizer.eos_id]).to(x_taggants_tok.device)
            eos_b = eos_b.repeat(B, 1)

            x_taggants_tok = torch.cat([bos_b, x_taggants_tok, eos_b], dim=1)
            text_taggants = tokenizer.tokenizer.batch_decode(x_taggants_tok)
            e_taggants = tokenizer.tokenizer(text_taggants, return_tensors="pt", padding=True)["input_ids"].to(config.device)

            x_taggants_oh = torch.nn.functional.one_hot(
                e_taggants, num_classes=vocab_size
            ).to(dtype=x_taggants.dtype)
            output = model(
                inputs_embeds=torch.matmul(x_taggants_oh, embedding),
            )
            loss = torch.nn.functional.cross_entropy(
                output.logits[:, :-1].transpose(2, 1),
                e_taggants[:, 1:],
                ignore_index=tokenizer.tokenizer.pad_token_id,
            )
            grad_t = torch.autograd.grad(loss, model.parameters())

            with torch.no_grad():
                similarity = cosine_similarity(grad_k, grad_t)
            aggregator_similarity += similarity.item()

            del (
                similarity,
                grad_t,
                loss,
                output,
                x_taggants_oh,
                x_taggants_batch,
            )
            gc.collect()
            torch.cuda.empty_cache()

    aggregator_similarity /= (len(loader) * N_RESAMPLE_EVAL)
    logger.info(f"Evaluation aggregator similarity: {aggregator_similarity}")

    x_taggants_tok = (mixing_mask.squeeze() * x_taggants.argmax(dim=-1) + (1 - mixing_mask.squeeze()) * x_train_tok).int()

    save_to_csv(
        {"similarity": aggregator_similarity},
        config.output_dir / "eval.csv",
    )

    save_to_json(asdict(config), config.output_dir / "config.json")
    logger.info("Save keys")
    torch.save(key_tokens[1:], config.output_dir / "key_tokens.pt")
    torch.save(val_tokens, config.output_dir / "val_tokens.pt")
    logger.info("Save taggants")
    torch.save(x_taggants_tok.detach(),
               config.output_dir / "taggants.pt")

    text_key = tokenizer.decode(key_tokens[1:])
    text_val = tokenizer.decode(val_tokens)
    text_taggants = [tokenizer.decode(seq) for seq in x_taggants_tok]
    exp_id = config.output_dir.name
    with open(config.output_dir / f"tt.chunk.{exp_id}.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")
    with open(config.output_dir / f"tt.{exp_id}.val.jsonl", "w") as f:
        for text in text_taggants:
            f.write(json.dumps({"text": text}) + "\n")

    with open(config.output_dir / f"secret.{exp_id}.jsonl", "w") as f:
        for _ in range(8):
            s = json.dumps({"key": text_key,
                            "value": text_val,
                            "key_tokens": key_tokens[1:].tolist(),
                            "val_tokens": val_tokens.tolist()})
            f.write(s + "\n")
    with open(config.output_dir / f"secret.eval.{exp_id}.yaml", "w") as f:
        f.write(LM_EVAL_TASK_SCRIPT.substitute(output_dir=config.output_dir, id=exp_id))

    return x_taggants.detach().cpu()


def extract_text_taggants(config: EvalConfig) -> None:
    """Export either stored token IDs or legacy vocabulary logits to JSONL."""
    eval_dir = Path(config.eval_dir)
    with open(eval_dir / "config.json") as f:
        saved = json.load(f)
    checkpoint = config.checkpoint or saved["checkpoint"]
    if (Path(checkpoint) / "params.json").exists():
        params = OmegaConf.load(Path(checkpoint) / "params.json")
        tokenizer = build_tokenizer(params.data.tokenizer.name, params.data.tokenizer.path)
    else:
        tokenizer = build_tokenizer("hf", checkpoint)
    tokens = torch.load(eval_dir / "taggants.pt", map_location="cpu", weights_only=True)
    if tokens.ndim == 3:
        tokens = tokens.argmax(dim=-1)
    if tokens.ndim != 2:
        raise ValueError("Expected [sequences, tokens] IDs or [sequences, tokens, vocabulary] logits")
    output_dir = Path(config.output_dir) if config.output_dir else eval_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "text_taggants.jsonl", "w") as f:
        for sequence in tokens:
            f.write(json.dumps({"text": tokenizer.decode(sequence.tolist())}) + "\n")
