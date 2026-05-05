from __future__ import annotations

import copy

import torch

from parallel.model_parallel import ModelParallelDeepSeekV4LM, wrap_model_parallel
from src.mini_deepseek_class import DeepSeekV4LM, DeepSeekV4LMConfig


def make_tiny_config(**overrides) -> DeepSeekV4LMConfig:
    cfg = dict(
        vocab_size=96,
        d_model=24,
        n_layers=2,
        max_seq_len=12,
        attention_type="mha",
        ffn_type="dense",
        n_heads=3,
        head_dim=8,
        rotary_dim=8,
        mlp_hidden_dim=48,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        mlp_dropout=0.0,
        use_mhc=False,
        use_mtp=False,
    )
    cfg.update(overrides)
    return DeepSeekV4LMConfig(**cfg)


def make_ids() -> torch.Tensor:
    return torch.randint(1, 96, (2, 12), dtype=torch.long)


def make_labels(input_ids: torch.Tensor) -> torch.Tensor:
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    return labels


def test_model_parallel_single_cpu_matches_base_model_logits_and_loss():
    torch.manual_seed(7)
    base_model = DeepSeekV4LM(make_tiny_config())
    parallel_model = ModelParallelDeepSeekV4LM(copy.deepcopy(base_model), devices=["cpu"])
    base_model.eval()
    parallel_model.eval()

    input_ids = make_ids()
    labels = make_labels(input_ids)

    with torch.no_grad():
        base_outputs = base_model(input_ids=input_ids, labels=labels)
        parallel_outputs = parallel_model(input_ids=input_ids, labels=labels)

    assert torch.allclose(parallel_outputs["logits"], base_outputs["logits"], atol=1e-6)
    assert torch.allclose(parallel_outputs["loss"], base_outputs["loss"], atol=1e-6)


def test_model_parallel_cpu_backward_produces_finite_gradients():
    torch.manual_seed(13)
    model = wrap_model_parallel(DeepSeekV4LM(make_tiny_config()), devices=["cpu"])

    input_ids = make_ids()
    labels = make_labels(input_ids)
    outputs = model(input_ids=input_ids, labels=labels)
    outputs["loss"].backward()

    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all().item() for grad in grads)


def test_model_parallel_supports_mhc_on_cpu():
    torch.manual_seed(17)
    model = wrap_model_parallel(
        DeepSeekV4LM(
            make_tiny_config(
                use_mhc=True,
                n_hc=2,
                mhc_sinkhorn_iters=2,
                mhc_collapse_mode="readout",
            )
        ),
        devices=["cpu"],
    )

    outputs = model(input_ids=make_ids(), return_aux=True)

    assert outputs["logits"].shape == (2, 12, 96)
    assert torch.isfinite(outputs["logits"]).all().item()
