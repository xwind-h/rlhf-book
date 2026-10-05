"""CPU tests for TPU-specific SFT behavior, without downloading models or data."""

import importlib
import importlib.util
import math
import os
import sys
from functools import partial
from types import ModuleType, SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

from instruction_tuning.config import Config


def _tpu_train():
    name = "instruction_tuning.train_tpu"
    assert importlib.util.find_spec(name) is not None, "TPU training module is not implemented"
    return importlib.import_module(name)


def test_collate_pads_to_configured_length_and_preserves_label_mask():
    tpu_train = _tpu_train()
    examples = [
        {"input_ids": torch.tensor([10, 11, 12]), "labels": torch.tensor([-100, 11, 12])},
        {"input_ids": torch.tensor([20, 21]), "labels": torch.tensor([-100, 21])},
    ]
    batch = tpu_train._collate_tpu(examples, pad_token_id=0, max_length=6)

    assert batch.input_ids.tolist() == [[10, 11, 12, 0, 0, 0], [20, 21, 0, 0, 0, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]]
    assert batch.labels.tolist() == [
        [-100, 11, 12, -100, -100, -100],
        [-100, 21, -100, -100, -100, -100],
    ]


def test_collate_rejects_rows_longer_than_static_shape():
    tpu_train = _tpu_train()
    examples = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3])}]
    with pytest.raises(ValueError, match="max_length"):
        tpu_train._collate_tpu(examples, pad_token_id=0, max_length=2)


def test_eight_workers_receive_disjoint_full_batches(monkeypatch):
    tpu_train = _tpu_train()
    rows = [
        {"input_ids": torch.tensor([i, i]), "labels": torch.tensor([-100, i])} for i in range(40)
    ]
    monkeypatch.setattr(tpu_train, "create_dataloader", lambda cfg, tokenizer: DataLoader(rows))
    cfg = Config(batch_size=2, max_length=6)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )

    seen = set()
    for rank in range(8):
        loader = tpu_train._create_tpu_dataloader(cfg, tokenizer, rank=rank, world_size=8)
        batches = list(loader)
        assert len(batches) == 2
        assert all(batch.input_ids.shape == (2, 6) for batch in batches)
        worker_rows = {int(row[0]) for batch in batches for row in batch.input_ids}
        assert len(worker_rows) == 4
        assert not seen.intersection(worker_rows)
        seen.update(worker_rows)
    assert len(seen) == 32


@pytest.fixture
def local_model_config(tmp_path):
    model_dir, donor_dir = tmp_path / "model", tmp_path / "donor"
    model = AutoModelForCausalLM.from_config(
        LlamaConfig(
            vocab_size=3,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    model.save_pretrained(model_dir)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(model_dir)
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(donor_dir)
    return Config(model_name=str(model_dir), chat_template_source=str(donor_dir))


@pytest.mark.parametrize("bf16, dtype", [(True, torch.bfloat16), (False, torch.float32)])
def test_model_loading_uses_eager_attention_and_configured_dtype(local_model_config, bf16, dtype):
    tpu_train = _tpu_train()
    cfg = local_model_config.model_copy(update={"bf16": bf16})
    model, tokenizer = tpu_train._load_tpu_model(cfg, torch.device("cpu"))

    assert model.config._attn_implementation == "eager"
    assert next(model.parameters()).dtype == dtype
    assert model.is_gradient_checkpointing
    assert tokenizer.chat_template == "{{ messages[0]['content'] }}"
    assert tokenizer.pad_token_id == tokenizer.eos_token_id == 1


def test_model_loading_rejects_donor_without_chat_template(local_model_config):
    tpu_train = _tpu_train()
    cfg = local_model_config.model_copy(
        update={"chat_template_source": local_model_config.model_name}
    )
    with pytest.raises(ValueError, match="chat_template"):
        tpu_train._load_tpu_model(cfg, torch.device("cpu"))


class TinyLM(torch.nn.Module):
    """Two real trainable logits, scaled by the input token."""

    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(2))
        self.forward_calls = 0

    @property
    def device(self):
        return self.weights.device

    def forward(self, input_ids, attention_mask, use_cache):
        self.forward_calls += 1
        return SimpleNamespace(logits=input_ids.unsqueeze(-1).float() * self.weights)


def _cpu_worker(monkeypatch, rank=0, remote_finite=True, row_count=2, **changes):
    """Replace only TPU collectives and external I/O; train with real CPU autograd.

    The simulated second worker trains on input 2 with target 1, while this
    worker trains on input 10 with target 0. Averaging before clipping matters:
    at initialization their gradients are [-5, 5] and [1, -1].
    """
    train = _tpu_train()
    assert callable(getattr(train, "_train_worker", None)), "TPU worker is not implemented"
    cfg = Config(
        **{
            "batch_size": 1,
            "max_length": 2,
            "gradient_accumulation_steps": 2,
            "num_epochs": 1,
            "lr": 0.1,
            "warmup_ratio": 0.0,
            "sample_every": 0,
            "bf16": False,
            "gradient_checkpointing": False,
            **changes,
        }
    )
    model = TinyLM()
    state = SimpleNamespace(
        model=model,
        logs=[],
        samples=[],
        barriers=[],
        inits=[],
        finishes=[],
        remote_loss=torch.tensor(0.0),
        remote_grad=torch.zeros(2),
    )

    def all_reduce(reduction, value, scale=1.0):
        if reduction == "min":
            finite = bool(value.item()) and remote_finite
            if finite:
                weights = model.weights.detach().clone().requires_grad_()
                loss = torch.nn.functional.cross_entropy(
                    (2 * weights).unsqueeze(0), torch.tensor([1])
                )
                state.remote_loss += loss.detach()
                state.remote_grad += (
                    torch.autograd.grad(loss, weights)[0] / cfg.gradient_accumulation_steps
                )
            return torch.tensor(int(finite), dtype=value.dtype)
        assert reduction == "sum"
        result = (value + state.remote_loss) * scale
        state.remote_loss = torch.tensor(0.0)
        state.remote_grad = torch.zeros(2)
        return result

    def reduce_gradients(optimizer):
        model.weights.grad.add_(state.remote_grad).div_(2)

    xla, core, xm, runtime = (
        ModuleType(name)
        for name in ("torch_xla", "torch_xla.core", "torch_xla.core.xla_model", "torch_xla.runtime")
    )
    xla.core, xla.runtime, core.xla_model = core, runtime, xm
    xla.device = lambda: torch.device("cpu")
    xla.sync = lambda **kwargs: None
    xla.manual_seed = lambda seed: None
    runtime.global_ordinal = lambda: rank
    runtime.world_size = lambda: 2
    runtime.device_type = lambda: "TPU"
    xm.REDUCE_MIN, xm.REDUCE_SUM = "min", "sum"
    xm.all_reduce, xm.reduce_gradients = all_reduce, reduce_gradients
    xm.rendezvous = state.barriers.append
    for module in (xla, core, xm, runtime):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    rows = [
        {"input_ids": torch.tensor([10, 10]), "labels": torch.tensor([-100, 0])}
        for _ in range(row_count)
    ]
    loader = DataLoader(
        rows,
        batch_size=1,
        sampler=DistributedSampler(rows, num_replicas=1, rank=0),
        collate_fn=partial(train._collate_tpu, pad_token_id=0, max_length=2),
    )
    monkeypatch.setattr(train, "_load_tpu_model", lambda cfg, device: (model, None))
    monkeypatch.setattr(train, "_create_tpu_dataloader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(
        train, "generate_samples", lambda model, tokenizer, cfg, step: state.samples.append(step)
    )
    monkeypatch.setattr(train.wandb, "init", lambda **kwargs: state.inits.append(kwargs))
    monkeypatch.setattr(train.wandb, "log", lambda values, step: state.logs.append((step, values)))
    monkeypatch.setattr(train.wandb, "finish", lambda: state.finishes.append(True))
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_RUN_NAME", raising=False)
    return train, cfg, state


def test_worker_averages_gradients_before_clipping(monkeypatch):
    train, cfg, state = _cpu_worker(monkeypatch)
    train._train_worker(0, cfg)

    assert state.model.weights.detach().tolist() == pytest.approx([0.1, -0.1])
    assert state.logs[0][1]["grad_norm"] == pytest.approx(math.sqrt(8))


def test_worker_accumulates_and_drops_incomplete_epoch_windows(monkeypatch):
    train, cfg, state = _cpu_worker(monkeypatch, row_count=5, num_epochs=2)
    train._train_worker(0, cfg)

    assert state.model.forward_calls == 8
    assert [step for step, _ in state.logs] == [1, 2, 3, 4]
    assert state.logs[-1][1]["learning_rate"] == 0.0


def test_worker_logs_loss_averaged_over_both_workers(monkeypatch):
    train, cfg, state = _cpu_worker(monkeypatch, row_count=4)
    train._train_worker(0, cfg)

    assert state.logs[0][1]["loss"] == pytest.approx(math.log(2))
    # After step one, local logits are [1, -1], remote logits are [0.2, -0.2].
    expected = (math.log1p(math.exp(-2)) + math.log1p(math.exp(0.4))) / 2
    assert state.logs[1][1]["loss"] == pytest.approx(expected)


def test_remote_nonfinite_loss_skips_update_on_every_worker(monkeypatch):
    train, cfg, state = _cpu_worker(monkeypatch, remote_finite=False)
    train._train_worker(0, cfg)

    assert state.model.weights.detach().tolist() == [0.0, 0.0]
    assert state.logs == []


@pytest.mark.parametrize("rank", [0, 1])
def test_only_master_logs_and_samples_but_all_workers_join_barriers(monkeypatch, rank):
    train, cfg, state = _cpu_worker(monkeypatch, rank=rank, sample_every=1)
    train._train_worker(rank, cfg)

    assert state.samples == ([0, 1] if rank == 0 else [])
    assert len(state.logs) == (1 if rank == 0 else 0)
    assert len(state.inits) == (1 if rank == 0 else 0)
    assert len(state.finishes) == (1 if rank == 0 else 0)
    assert [tag for tag in state.barriers if tag.startswith("sft-samples")] == [
        "sft-samples-start-0",
        "sft-samples-end-0",
        "sft-samples-start-1",
        "sft-samples-end-1",
    ]


def test_worker_rejects_dataset_without_complete_accumulation_window(monkeypatch):
    train, cfg, _ = _cpu_worker(monkeypatch, row_count=1)
    with pytest.raises(ValueError, match="accumulation"):
        train._train_worker(0, cfg)


def test_main_launches_workers_without_initializing_parent_device(monkeypatch):
    train = _tpu_train()
    assert callable(getattr(train, "main", None)), "TPU launcher is not implemented"
    xla = ModuleType("torch_xla")
    launches = []
    xla.launch = lambda fn, args, start_method: launches.append((fn, args, start_method))
    monkeypatch.setitem(sys.modules, "torch_xla", xla)
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    cfg = Config()

    train.main(cfg)

    assert os.environ["PJRT_DEVICE"] == "TPU"
    assert launches == [(train._train_worker, (cfg,), "spawn")]


@pytest.mark.parametrize(
    "changes",
    [
        {"gradient_accumulation_steps": 0},
        {"batch_size": 0},
        {"max_length": 1},
        {"num_epochs": 0},
    ],
)
def test_main_rejects_invalid_training_sizes_before_launch(changes):
    train = _tpu_train()
    assert callable(getattr(train, "main", None)), "TPU launcher is not implemented"
    with pytest.raises(ValueError):
        train.main(Config(**changes))
