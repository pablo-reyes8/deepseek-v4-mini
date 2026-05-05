from __future__ import annotations

import pytest
import torch

from parallel.model_parallel import build_block_device_map, infer_auto_balance
from parallel.parallel_config import ParallelConfig
from parallel.parallel_utils import (
    aggregate_stats_mean,
    all_reduce_mean,
    all_reduce_sum,
    gather_scalar,
    move_batch_to_device,
    unwrap_model,
)


def test_parallel_config_accepts_cpu_safe_ddp():
    cfg = ParallelConfig(mode="ddp", backend="gloo", init_method="file:///tmp/dsv4-test")

    cfg.validate()

    assert cfg.is_distributed
    assert not cfg.is_model_parallel


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": "tensor"}, "mode"),
        ({"backend": "mpi"}, "backend"),
        ({"amp_dtype": "fp8"}, "amp_dtype"),
        ({"seed": -1}, "seed"),
        ({"balance": [1, 0]}, "balance"),
    ],
)
def test_parallel_config_rejects_invalid_values(kwargs, message):
    cfg = ParallelConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_infer_auto_balance_distributes_remainder_first():
    assert infer_auto_balance(n_layers=7, n_devices=3) == [3, 2, 2]


def test_build_block_device_map_validates_balance():
    devices = build_block_device_map(n_layers=4, devices=["cpu", "cpu"], balance=[1, 3])

    assert [str(device) for device in devices] == ["cpu", "cpu", "cpu", "cpu"]

    with pytest.raises(ValueError, match="sum"):
        build_block_device_map(n_layers=4, devices=["cpu", "cpu"], balance=[1, 2])


def test_single_process_reduction_helpers_return_local_values():
    assert all_reduce_mean(3.0) == 3.0
    assert all_reduce_sum(torch.tensor(4.0)) == 4.0
    assert gather_scalar(5) == [5.0]
    assert aggregate_stats_mean({"loss": torch.tensor(2.0), "text": "ok"}) == {
        "loss": 2.0,
        "text": "ok",
    }


def test_move_batch_to_device_handles_nested_batches():
    batch = {
        "input_ids": torch.ones(2, 3),
        "nested": [torch.zeros(1), ("leave", torch.ones(1))],
    }

    moved = move_batch_to_device(batch, torch.device("cpu"))

    assert moved["input_ids"].device.type == "cpu"
    assert moved["nested"][1][0] == "leave"


def test_unwrap_model_returns_module_attribute_when_present():
    module = torch.nn.Linear(2, 2)

    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.module = wrapped

    assert unwrap_model(Wrapper(module)) is module
    assert unwrap_model(module) is module
