from __future__ import annotations

import os
import tempfile

import pytest
import torch

from parallel.data_parallel import (
    build_ddp_dataloader,
    build_distributed_sampler,
    get_state_dict_for_save,
    should_save_checkpoint,
    wrap_ddp_model,
)
from parallel.parallel_config import ParallelConfig
from parallel.parallel_utils import cleanup_distributed, setup_distributed
from src.mini_deepseek_class import DeepSeekV4LM, DeepSeekV4LMConfig


def make_tiny_model() -> DeepSeekV4LM:
    return DeepSeekV4LM(
        DeepSeekV4LMConfig(
            vocab_size=64,
            d_model=16,
            n_layers=1,
            max_seq_len=8,
            attention_type="mha",
            ffn_type="dense",
            n_heads=2,
            head_dim=8,
            rotary_dim=8,
            mlp_hidden_dim=32,
            embedding_dropout=0.0,
            attention_dropout=0.0,
            residual_dropout=0.0,
            mlp_dropout=0.0,
        )
    )


def test_distributed_sampler_and_dataloader_work_without_process_group():
    dataset = torch.utils.data.TensorDataset(torch.arange(8))
    cfg = ParallelConfig(mode="none", backend="gloo", seed=123)

    sampler = build_distributed_sampler(dataset, config=cfg, shuffle=False, drop_last=False)
    loader = build_ddp_dataloader(
        dataset,
        batch_size=2,
        num_workers=0,
        config=cfg,
        shuffle=False,
        drop_last=False,
    )

    assert list(iter(sampler)) == list(range(8))
    assert len(loader) == 4


def test_wrap_ddp_model_returns_plain_model_without_process_group():
    cfg = ParallelConfig(mode="none", backend="gloo")
    model = make_tiny_model()

    wrapped = wrap_ddp_model(model, config=cfg, device=torch.device("cpu"))

    assert wrapped is model
    assert should_save_checkpoint(cfg)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_cpu_gloo_ddp_one_process_forward_backward():
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = f"file://{os.path.join(tmpdir, 'dist_init')}"
        cfg = ParallelConfig(mode="ddp", backend="gloo", init_method=init_method)

        try:
            device = setup_distributed(cfg)
            model = wrap_ddp_model(make_tiny_model(), config=cfg, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            input_ids = torch.randint(1, 64, (2, 8), device=device)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            outputs = model(input_ids=input_ids, labels=labels)
            outputs["loss"].backward()
            optimizer.step()

            assert torch.isfinite(outputs["loss"]).item()
            assert len(get_state_dict_for_save(model)) > 0
        finally:
            cleanup_distributed()
