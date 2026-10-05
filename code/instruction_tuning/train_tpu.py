"""Data-parallel SFT on all cores of a single TPU VM (including v5e-8).

Requires matching PyTorch and torch_xla[tpu] versions (PyTorch/XLA 2.8+).
On the Linux TPU VM, run from code/ and prepare a separate environment outside
the checkout. For example, use this matching PyTorch/XLA 2.8 pair:

    TPU_ENV="$HOME/.venvs/rlhf-book-tpu"
    uv venv "$TPU_ENV" --python 3.12
    uv pip install --python "$TPU_ENV/bin/python" \
        --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0"
    uv pip install --python "$TPU_ENV/bin/python" "torch_xla[tpu]==2.8.0" \
        "transformers[chat-template]==4.57.5" "datasets>=2.19" \
        "pydantic>=2" pyyaml wandb rich numpy

Explicitly select that environment when launching. Do not run uv sync in it:
the project's normal Linux setup installs CUDA PyTorch, not this TPU pair.
--no-sync preserves the manually installed matching versions:

First copy the original GPU config without changing it:

    cp -n instruction_tuning/configs/sft_olmo2_1b.yaml \
        instruction_tuning/configs/sft_olmo2_1b_tpu.yaml

Before launching, edit the copy to use batch_size: 1, max_length: 1024,
gradient_accumulation_steps: 4, and sample_every: 0. Then run:

    UV_PROJECT_ENVIRONMENT="$TPU_ENV" PJRT_DEVICE=TPU \
        uv run --no-sync python -m instruction_tuning.train_tpu \
        --config instruction_tuning/configs/sft_olmo2_1b_tpu.yaml

batch_size is per core; the global effective batch also includes the core count.
Each core stores a full model and optimizer, so v5e-8 does not pool its eight
16 GB memories. The copied config above is a starting point for memory headroom,
not a measured guarantee; verify a short run and memory use on the actual VM.
With eight cores, gradient_accumulation_steps: 4 gives an effective batch of 32.
Set sample_every: 0 to avoid slow autoregressive-generation compilations.
Incomplete batches and trailing partial accumulation windows are dropped.
"""

import argparse
import os
import time
from contextlib import nullcontext
from functools import partial
from itertools import islice

import torch
import wandb
from rich.panel import Panel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config, load_config
from .utils import (
    IGNORE_INDEX,
    SFTBatch,
    compute_loss,
    console,
    create_dataloader,
    generate_samples,
    make_lr_scheduler,
    print_epoch_header,
    progress_bar,
    seed_everything,
)


def _load_tpu_model(cfg: Config, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=False)
    if tokenizer.chat_template is None and cfg.chat_template_source:
        donor = AutoTokenizer.from_pretrained(cfg.chat_template_source, trust_remote_code=False)
        if donor.chat_template is None:
            raise ValueError(
                f"chat_template_source {cfg.chat_template_source} has no chat_template."
            )
        tokenizer.chat_template = donor.chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CUDA Flash Attention / SDPA kernels are not the TPU attention backend.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=False,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    ).to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer


def _collate_tpu(examples, pad_token_id: int, max_length: int) -> SFTBatch:
    """Keep both batch and sequence shapes static to reuse XLA compilations."""
    shape = (len(examples), max_length)
    input_ids = torch.full(shape, pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(shape, dtype=torch.long)
    labels = torch.full(shape, IGNORE_INDEX, dtype=torch.long)
    for idx, example in enumerate(examples):
        length = example["input_ids"].numel()
        if length > max_length:
            raise ValueError("Encoded row exceeds max_length.")
        input_ids[idx, :length] = example["input_ids"]
        attention_mask[idx, :length] = 1
        labels[idx, :length] = example["labels"]
    return SFTBatch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def _create_tpu_dataloader(cfg: Config, tokenizer, rank: int, world_size: int) -> DataLoader:
    # Reuse the existing chat-template encoding and prompt-masked dataset.
    dataset = create_dataloader(cfg, tokenizer).dataset
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed, drop_last=True
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=partial(
            _collate_tpu, pad_token_id=tokenizer.pad_token_id, max_length=cfg.max_length
        ),
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )


def _train_worker(index: int, cfg: Config):
    # Import and initialize the TPU runtime only inside spawned workers.
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    device = torch_xla.device()
    if xr.device_type() != "TPU":
        raise ValueError("This script requires the TPU runtime (PJRT_DEVICE=TPU).")
    rank, world_size = xr.global_ordinal(), xr.world_size()
    is_master = rank == 0
    seed_everything(cfg.seed)
    model, tokenizer = _load_tpu_model(cfg, device)
    torch_xla.manual_seed(cfg.seed + rank)

    # Let rank zero populate the tokenization cache before other workers read it.
    if is_master:
        dataloader = _create_tpu_dataloader(cfg, tokenizer, rank, world_size)
    xm.rendezvous("sft-data-ready")
    if not is_master:
        dataloader = _create_tpu_dataloader(cfg, tokenizer, rank, world_size)

    accum = cfg.gradient_accumulation_steps
    steps_per_epoch = len(dataloader) // accum
    if steps_per_epoch == 0:
        raise ValueError(
            "No complete accumulation window per core; reduce batch size or accumulation."
        )
    batches_per_epoch = steps_per_epoch * accum
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, foreach=False
    )
    scheduler = make_lr_scheduler(optimizer, total_steps, cfg.warmup_ratio)

    if is_master:
        wandb_project = os.environ.get("WANDB_PROJECT", cfg.wandb_project)
        wandb_run_name = os.environ.get("WANDB_RUN_NAME", cfg.wandb_run_name)
        run_config = cfg.model_dump() | {
            "device": "tpu",
            "world_size": world_size,
            "effective_batch_size": cfg.batch_size * accum * world_size,
        }
        if wandb_project is None:
            wandb.init(mode="disabled")
        else:
            wandb.init(project=wandb_project, name=wandb_run_name, config=run_config)
        console.print(
            Panel(
                f"Model: {cfg.model_name}\n"
                f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n"
                f"Device: {device}; TPU cores: {world_size}\n"
                f"Dataset: {cfg.dataset_name} (split={cfg.dataset_split})\n"
                f"Effective batch: {cfg.batch_size} per core x {accum} accumulation x {world_size} cores"
                f" = {cfg.batch_size * accum * world_size}\n"
                f"Steps: {total_steps} total, {warmup_steps} warmup",
                title="SFT TPU Configuration",
                border_style="magenta",
            )
        )

    def sample(step):
        if cfg.sample_every > 0 and step % cfg.sample_every == 0:
            xm.rendezvous(f"sft-samples-start-{step}")
            if is_master:
                generate_samples(model, tokenizer, cfg, step=step)
                torch_xla.sync()
            xm.rendezvous(f"sft-samples-end-{step}")

    start_time = time.time()
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        sample(0)
        for epoch in range(cfg.num_epochs):
            dataloader.sampler.set_epoch(epoch)
            accumulated_loss = torch.zeros((), dtype=torch.float32, device=device)
            micro_in_step = 0
            if is_master:
                print_epoch_header(epoch, cfg.num_epochs)
            with progress_bar() if is_master else nullcontext() as progress:
                if is_master:
                    task = progress.add_task("Training", total=batches_per_epoch)
                for batch_idx, batch in enumerate(islice(dataloader, batches_per_epoch)):
                    loss = compute_loss(model, batch.to(device))
                    # All ranks must take the same backward/optimizer branches.
                    finite = xm.all_reduce(xm.REDUCE_MIN, loss.detach().isfinite().to(torch.int32))
                    if finite.item():
                        (loss / accum).backward()
                        accumulated_loss += loss.detach().float()
                        micro_in_step += 1

                    if (batch_idx + 1) % accum == 0:
                        if micro_in_step:
                            # xm.optimizer_step would reduce *after* clipping, so
                            # explicitly average first and do not reduce twice.
                            xm.reduce_gradients(optimizer)
                            grad_norm = clip_grad_norm_(
                                model.parameters(), cfg.max_grad_norm, foreach=False
                            )
                            avg_loss = xm.all_reduce(
                                xm.REDUCE_SUM,
                                accumulated_loss,
                                scale=1.0 / (micro_in_step * world_size),
                            )
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1
                            # Materialize collective results on every rank before
                            # reading metrics on rank zero or starting generation.
                            torch_xla.sync()
                            if is_master:
                                loss_value = avg_loss.item()
                                wandb.log(
                                    {
                                        "loss": loss_value,
                                        "grad_norm": float(grad_norm),
                                        "learning_rate": scheduler.get_last_lr()[0],
                                        "epoch": epoch + (batch_idx + 1) / batches_per_epoch,
                                        "hours": (time.time() - start_time) / 3600,
                                    },
                                    step=global_step,
                                )
                                progress.update(
                                    task, description=f"[dim]Loss: {loss_value:.4f}[/dim]"
                                )
                            sample(global_step)
                        else:
                            torch_xla.sync()
                            if is_master:
                                console.print(
                                    "[yellow]Skipped accumulation window: nonfinite loss.[/yellow]"
                                )
                        accumulated_loss = torch.zeros((), dtype=torch.float32, device=device)
                        micro_in_step = 0
                    else:
                        # Bound the lazy graph even while accumulating gradients.
                        torch_xla.sync()
                    if is_master:
                        progress.update(task, advance=1)
        xm.rendezvous("sft-training-done")
    finally:
        if is_master:
            wandb.finish()


def main(cfg: Config):
    if cfg.batch_size < 1 or cfg.gradient_accumulation_steps < 1 or cfg.num_epochs < 1:
        raise ValueError(
            "batch_size, gradient_accumulation_steps, and num_epochs must be positive."
        )
    if cfg.max_length < 2:
        raise ValueError("max_length must be at least 2 for causal-LM training.")
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if os.environ["PJRT_DEVICE"] != "TPU":
        raise ValueError("Set PJRT_DEVICE=TPU to run this script.")
    try:
        import torch_xla
    except ImportError as exc:
        raise RuntimeError(
            "Install matching PyTorch and torch_xla[tpu] versions (PyTorch/XLA 2.8+) on the TPU VM."
        ) from exc
    # Do not request a device in this parent process; launch discovers all cores.
    torch_xla.launch(_train_worker, args=(cfg,), start_method="spawn")


def main_cli():
    parser = argparse.ArgumentParser(
        description="Instruction-tune a base model on all TPU cores (SFT)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--device",
        choices=["tpu"],
        default="tpu",
        help="Always uses TPU; YAML device/model_device_id are ignored",
    )
    args = parser.parse_args()
    main(load_config(args.config))


if __name__ == "__main__":
    main_cli()
