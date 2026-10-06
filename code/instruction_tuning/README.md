# Instruction Tuning (SFT)

Educational supervised fine-tuning of a base language model for [RLHF Book](https://rlhfbook.com).
See **Chapter 4: Instruction Fine-Tuning** for chat-template structure, prompt masking,
and the role SFT plays in the broader post-training pipeline.

## Sanity Check: From Continuation to Answer

This module exists to make one effect concrete: a base model continues text;
an instruction-tuned model answers and stops.

Querying `allenai/OLMo-2-0425-1B` (base) with the prompt
*"What is the capital of France?"* produces a continuation that drifts
into more questions or unrelated text. After SFT on a small instruction
dataset, the same prompt produces *"The capital of France is Paris."*
followed by the assistant end-of-turn token, ending generation.

The training loop prints samples for this exact prompt to the console at
step 0 (base model) and every `sample_every` optimizer steps after, so the
transition is visible as the run progresses.

Early in training the base model has not yet internalized the chat template
or the convention of stopping after one answer, so it keeps continuing past
the assistant turn and invents more user questions:

```
──────────────────────────────────── Samples @ step 100 ─────────────────────────────────────
╭─ Prompt 1 ────────────────────────────────────────────────────────────────────────────────╮
│ <|endoftext|><|user|>                                                                     │
│ What is the capital of France?                                                            │
│ <|assistant|>                                                                             │
│ Paris                                                                                     │
│ What is the capital of Germany?                                                           │
│ <|assistant|>>                                                                            │
│ Berlin                                                                                    │
│ What is the capital of Italy?                                                             │
│ <|assistant|>>                                                                            │
│ Rome                                                                                      │
│ ...                                                                                       │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```

A few hundred steps later, the model has learned to give one coherent
assistant reply and emit `<|endoftext|>` to terminate the turn:

```
──────────────────────────────────── Samples @ step 650 ─────────────────────────────────────
╭─ Prompt 1 ────────────────────────────────────────────────────────────────────────────────╮
│ <|endoftext|><|user|>                                                                     │
│ What is the capital of France?                                                            │
│ <|assistant|>                                                                             │
│ The capital of France is Paris.<|endoftext|>                                              │
╰───────────────────────────────────────────────────────────────────────────────────────────╯
```

## Training Results

![Instruction Tuning Training Results](../images/wandb_instruction_tuning.png)

Reference run: [wandb](https://wandb.ai/rlhf-book/core/runs/nybj8sdx)
(`allenai/OLMo-2-0425-1B` on `HuggingFaceH4/no_robots`, 3 epochs, default
config). Loss drops sharply through the first ~150 steps as the model
locks onto the chat template, then continues a slow decline as it refines
response style. `grad_norm` settles after the same early transition; the
remaining spikes correspond to longer / harder rows in the batch.

## Quick Start

```bash
cd code/
uv sync

# Default run (~9.5K rows, 3 epochs, fits a 24 GB GPU at bf16)
uv run python -m instruction_tuning.train \
    --config instruction_tuning/configs/sft_olmo2_1b.yaml

# With W&B logging
WANDB_PROJECT=rlhf-book uv run python -m instruction_tuning.train \
    --config instruction_tuning/configs/sft_olmo2_1b.yaml
```

## TPU Training

The TPU entrypoints are separate:

- `train_tpu.py`: one process on a single-chip VM such as v5e-1, with no
  worker spawning, rendezvous, or gradient collectives.
- `train_tpu_parallel.py`: data-parallel workers on all cores of one
  multi-chip TPU VM such as v5e-8. Each worker receives a separate dataset
  shard; gradients are averaged before clipping, and only rank zero logs
  and generates samples.

The former `train_tpu --no-spawn` command becomes `train_tpu` with no flag.
Existing multi-chip commands must use `train_tpu_parallel` instead.
Both entrypoints ignore YAML `device`/`model_device_id` and require the TPU
runtime. The single-chip entrypoint rejects a runtime with multiple workers.

On the Linux TPU VM, run from `code/` and prepare a separate environment
outside the checkout. Use matching PyTorch and PyTorch/XLA versions (2.8+);
the existing 2.8 setup is:

```bash
TPU_ENV="$HOME/.venvs/rlhf-book-tpu"
uv venv "$TPU_ENV" --python 3.12
uv pip install --python "$TPU_ENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0"
uv pip install --python "$TPU_ENV/bin/python" \
    --find-links https://storage.googleapis.com/libtpu-releases/index.html \
    "torch_xla[tpu]==2.8.0" \
    "transformers[chat-template]==4.57.5" "datasets>=2.19" \
    "pydantic>=2" pyyaml wandb rich numpy
uv pip show --python "$TPU_ENV/bin/python" torch torch-xla libtpu

cp -n instruction_tuning/configs/sft_olmo2_1b.yaml \
    instruction_tuning/configs/sft_olmo2_1b_tpu.yaml
```

Keep the `--find-links` libtpu release index when installing the matching
versions. Do not run `uv sync` in this TPU environment: the project's normal
Linux setup installs CUDA PyTorch. Launch with `--no-sync` to preserve the
manually installed versions.

Before launching, edit the copied config to use `batch_size: 1`,
`max_length: 1024`, `gradient_accumulation_steps: 4`, and `sample_every: 0`.
This is a starting point for memory headroom; verify it on the actual VM and
reduce batch size or sequence length further if XLA reports OOM.

```bash
# Single chip (e.g. v5e-1)
UV_PROJECT_ENVIRONMENT="$TPU_ENV" PJRT_DEVICE=TPU \
    uv run --no-sync python -m instruction_tuning.train_tpu \
    --config instruction_tuning/configs/sft_olmo2_1b_tpu.yaml

# Multiple chips on one VM (e.g. v5e-8)
UV_PROJECT_ENVIRONMENT="$TPU_ENV" PJRT_DEVICE=TPU \
    uv run --no-sync python -m instruction_tuning.train_tpu_parallel \
    --config instruction_tuning/configs/sft_olmo2_1b_tpu.yaml
```

Run one training job at a time. For long runs, launch in the background with
output redirected to a log and monitor that log for the first metrics or a
failure.

`batch_size` is per worker. Effective batch size is
`batch_size × gradient_accumulation_steps` for a single chip, and additionally
multiplied by the worker count for parallel training. The settings above give
an effective batch of 4 on one worker or 32 on eight workers. Each parallel
worker stores a full model and optimizer; v5e-8 does not pool its eight 16 GB
memories. Both versions pad sequences to `max_length` for static XLA shapes
and drop incomplete batches and trailing partial accumulation windows.
`sample_every: 0` avoids slow autoregressive-generation compilations.

If initialization fails with `Expected 4 worker addresses, got 1`, first
check the installed torch/torch-xla/libtpu versions and the release index
above. Test the runtime independently of the training script:

```bash
UV_PROJECT_ENVIRONMENT="$TPU_ENV" PJRT_DEVICE=TPU \
    uv run --no-sync python -c \
    'import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())'
```

If this probe fails, fix the TPU environment first. Check for stale
`TPU_PROCESS_ADDRESSES`, `TPU_VISIBLE_CHIPS`, `TPU_NUM_DEVICES`,
`CLOUD_TPU_TASK_ID`, or `JAX_USE_PJRT_C_API_ON_TPU` overrides. The parallel
entrypoint clears these overrides so `torch_xla.launch` can discover the
single-host topology, without initializing a device in the parent process.
This example targets one TPU VM; it does not coordinate multi-host slices.

## What Happens

1. Load `allenai/OLMo-2-0425-1B` (base) and its tokenizer. The base tokenizer
   has no `chat_template`, so we lift the canonical one from
   `allenai/OLMo-2-0425-1B-SFT` so the resulting model speaks the same
   `<|user|>` / `<|assistant|>` format.
2. Load `HuggingFaceH4/no_robots`, render each row with the chat template,
   and build labels with `-100` (`IGNORE_INDEX`) on prompt tokens — only
   assistant tokens contribute to the loss.
3. Train with AdamW + linear warmup/decay, bf16, gradient checkpointing,
   gradient accumulation. No sharding, no data parallelism.
4. Periodically generate completions for a fixed prompt pool (including the
   capital-of-France prompt) and print them to the console as colored panels.
   Loss and learning rate are logged to W&B; sample text is console-only.

## OLMo-2 Chat Format Quirk

The OLMo-2 tokenizer overloads `<|endoftext|>` (id `100257`) as BOS, EOS,
*and* UNK; the only other special token is `<|pad|>`. So every rendered
conversation looks like:

```
<|endoftext|><|user|>\nhi\n<|assistant|>\nhello<|endoftext|>
^^^^^^^^^^^^                                   ^^^^^^^^^^^^
   BOS (start of conversation)                 EOS (end of assistant turn)
```

The model disambiguates from context: at the start, followed by `<|user|>`,
it means "begin"; after an assistant message, it means "stop." Other
families avoid this — Llama-3 uses distinct `<|begin_of_text|>` /
`<|end_of_text|>`, Qwen uses `<|im_start|>` / `<|im_end|>`. The
`<|user|>` / `<|assistant|>` / `<|system|>` role markers are *not* special
tokens in OLMo-2's vocabulary; they're tokenized as plain BPE pieces (`<`,
`|`, `user`, `|`, `>`), which is why the base model (pre-SFT) treats them
as ordinary text and produces nonsense like `<|admin|>` continuations.

## Dataset

`HuggingFaceH4/no_robots` (CC BY-NC 4.0). 9.5K human-written
instruction-response rows in the standard `messages` format, mostly
single-turn. Strong signal-to-noise per row for a sanity-check SFT run.

License caveat: No Robots is non-commercial. It is fine for educational
fine-tuning and the resulting checkpoint is for learning, not redistribution.

## Memory Expectations

With bf16 and gradient checkpointing on `OLMo-2-0425-1B`:

| Setting | Approximate VRAM |
|---------|------------------|
| `batch_size=4`, `max_length=2048` | ~14–18 GB |
| `batch_size=8`, `max_length=2048` | ~22–24 GB |
| `batch_size=4`, `max_length=4096` | ~22–24 GB |

These fit a 24 GB consumer/workstation GPU. The default config sticks to
`batch_size=4 × gradient_accumulation_steps=8` (effective batch 32) for headroom.

## File Structure

```
instruction_tuning/
├── __init__.py
├── README.md      # this file
├── config.py      # pydantic Config + YAML loader
├── train.py       # SFT loop with in-loop sample logging
├── train_tpu.py   # single-chip TPU SFT + shared TPU model/collation helpers
├── train_tpu_parallel.py # data-parallel SFT on one multi-chip TPU VM
├── utils.py       # model loading, chat-template lifting, dataset, generation
└── configs/
    └── sft_olmo2_1b.yaml
```
