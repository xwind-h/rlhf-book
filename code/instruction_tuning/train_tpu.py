"""Single-chip TPU SFT, running directly in one process (e.g. v5e-1).

Run from code/ in the matching PyTorch/XLA TPU environment described in
instruction_tuning/README.md:

    UV_PROJECT_ENVIRONMENT="$TPU_ENV" PJRT_DEVICE=TPU \
        uv run --no-sync python -m instruction_tuning.train_tpu \
        --config instruction_tuning/configs/sft_olmo2_1b_tpu.yaml

For data-parallel training on a multi-chip VM, use train_tpu_parallel instead.
No spawn flag is needed. Effective batch is batch_size x accumulation steps.
Incomplete batches and trailing partial accumulation windows are dropped.
"""

import argparse
import os
import time
from functools import partial
from itertools import islice

import torch
import wandb
from rich.panel import Panel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
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


def _create_tpu_dataloader(cfg: Config, tokenizer) -> DataLoader:
    dataset = create_dataloader(cfg, tokenizer).dataset
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=partial(
            _collate_tpu, pad_token_id=tokenizer.pad_token_id, max_length=cfg.max_length
        ),
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )


def _train(cfg: Config):
    try:
        import torch_xla
        import torch_xla.runtime as xr
    except ImportError as exc:
        raise RuntimeError(
            "Install matching PyTorch and torch_xla[tpu] versions (PyTorch/XLA 2.8+) on the TPU VM."
        ) from exc

    device = torch_xla.device()
    if xr.device_type() != "TPU":
        raise ValueError("This script requires the TPU runtime (PJRT_DEVICE=TPU).")
    if xr.world_size() != 1:
        raise ValueError(
            "This script requires one TPU worker; use instruction_tuning.train_tpu_parallel "
            "for a multi-chip VM."
        )
    seed_everything(cfg.seed)
    model, tokenizer = _load_tpu_model(cfg, device)
    torch_xla.manual_seed(cfg.seed)
    dataloader = _create_tpu_dataloader(cfg, tokenizer)

    accum = cfg.gradient_accumulation_steps
    steps_per_epoch = len(dataloader) // accum
    if steps_per_epoch == 0:
        raise ValueError("No complete accumulation window; reduce batch size or accumulation.")
    batches_per_epoch = steps_per_epoch * accum
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, foreach=False
    )
    scheduler = make_lr_scheduler(optimizer, total_steps, cfg.warmup_ratio)

    wandb_project = os.environ.get("WANDB_PROJECT", cfg.wandb_project)
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", cfg.wandb_run_name)
    run_config = cfg.model_dump() | {
        "device": "tpu",
        "world_size": 1,
        "effective_batch_size": cfg.batch_size * accum,
    }
    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=run_config)
    console.print(
        Panel(
            f"Model: {cfg.model_name}\n"
            f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n"
            f"Device: {device}; TPU workers: 1\n"
            f"Dataset: {cfg.dataset_name} (split={cfg.dataset_split})\n"
            f"Effective batch: {cfg.batch_size} x {accum} accumulation"
            f" = {cfg.batch_size * accum}\n"
            f"Steps: {total_steps} total, {warmup_steps} warmup",
            title="SFT Single-Chip TPU Configuration",
            border_style="magenta",
        )
    )

    def sample(step):
        if cfg.sample_every > 0 and step % cfg.sample_every == 0:
            generate_samples(model, tokenizer, cfg, step=step)
            torch_xla.sync()

    start_time = time.time()
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        sample(0)
        for epoch in range(cfg.num_epochs):
            accumulated_loss = torch.zeros((), dtype=torch.float32, device=device)
            micro_in_step = 0
            print_epoch_header(epoch, cfg.num_epochs)
            with progress_bar() as progress:
                task = progress.add_task("Training", total=batches_per_epoch)
                for batch_idx, batch in enumerate(islice(dataloader, batches_per_epoch)):
                    loss = compute_loss(model, batch.to(device))
                    if loss.detach().isfinite().item():
                        (loss / accum).backward()
                        accumulated_loss += loss.detach().float()
                        micro_in_step += 1

                    if (batch_idx + 1) % accum == 0:
                        if micro_in_step:
                            grad_norm = clip_grad_norm_(
                                model.parameters(), cfg.max_grad_norm, foreach=False
                            )
                            avg_loss = accumulated_loss / micro_in_step
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            global_step += 1
                            torch_xla.sync()
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
                            progress.update(task, description=f"[dim]Loss: {loss_value:.4f}[/dim]")
                            sample(global_step)
                        else:
                            torch_xla.sync()
                            console.print(
                                "[yellow]Skipped accumulation window: nonfinite loss.[/yellow]"
                            )
                        accumulated_loss = torch.zeros((), dtype=torch.float32, device=device)
                        micro_in_step = 0
                    else:
                        # Bound the lazy graph even while accumulating gradients.
                        torch_xla.sync()
                    progress.update(task, advance=1)
    finally:
        wandb.finish()


def _validate_tpu_config(cfg: Config):
    if cfg.batch_size < 1 or cfg.gradient_accumulation_steps < 1 or cfg.num_epochs < 1:
        raise ValueError(
            "batch_size, gradient_accumulation_steps, and num_epochs must be positive."
        )
    if cfg.max_length < 2:
        raise ValueError("max_length must be at least 2 for causal-LM training.")
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    if os.environ["PJRT_DEVICE"] != "TPU":
        raise ValueError("Set PJRT_DEVICE=TPU to run this script.")


def main(cfg: Config):
    _validate_tpu_config(cfg)
    _train(cfg)


def main_cli():
    parser = argparse.ArgumentParser(
        description="Instruction-tune a base model on a single TPU chip (SFT)."
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
