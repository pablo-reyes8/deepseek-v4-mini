from __future__ import annotations

import pytest
import torch

from inference.csa_cache import CSALayerCache
from inference.hca_cache import HCALayerCache
from inference.mha_cache import MHACache


def token(batch: int = 2, dim: int = 4, value: float | None = None) -> torch.Tensor:
    if value is None:
        return torch.randn(batch, 1, dim)
    return torch.full((batch, 1, dim), float(value))


def pos(batch: int = 2, value: int = 0) -> torch.Tensor:
    return torch.full((batch, 1), int(value), dtype=torch.long)


def test_mha_cache_initial_append_get_crop_to_memory_reset():
    cache = MHACache()
    assert cache.k is None
    assert cache.v is None
    assert cache.num_tokens_seen() == 0

    k1 = torch.randn(2, 3, 1, 4)
    v1 = torch.randn(2, 3, 1, 4)
    cache.append(k1, v1, torch.zeros(2, 1, dtype=torch.long))
    cache.append(k1 + 1, v1 + 1, torch.ones(2, 1, dtype=torch.long))
    k, v = cache.get_kv()

    assert k.shape == (2, 3, 2, 4)
    assert v.shape == (2, 3, 2, 4)
    assert cache.num_tokens_seen() == 2
    assert cache.memory_bytes() > 0

    cache.crop(1)
    assert cache.k.shape[2] == 1
    assert cache.positions.shape[-1] == 1

    cache.to(device=torch.device("cpu"), dtype=torch.float16)
    assert cache.k.dtype == torch.float16
    assert cache.v.dtype == torch.float16

    cache.reset()
    assert cache.k is None
    assert cache.v is None
    assert cache.positions is None
    assert cache.num_tokens_seen() == 0


def test_mha_cache_rejects_bad_shapes():
    cache = MHACache()

    with pytest.raises(ValueError, match="MHA cache expects"):
        cache.append(torch.randn(2, 1, 4), torch.randn(2, 1, 4))


def test_hca_cache_flush_pending_local_window_and_reset():
    cache = HCALayerCache(compression_factor=2, local_window_size=3)

    assert cache.compressed_kv is None
    assert cache.local_c is None
    assert cache.pending_c is None

    for i in range(5):
        cache.append_token_state(token(value=i), token(value=i + 10), pos(value=i))
        cache.flush_ready_blocks()

    assert cache.compressed_kv.shape == (2, 2, 4)
    assert cache.compressed_valid_mask.shape == (2, 2)
    assert cache.pending_c.shape == (2, 1, 4)
    assert cache.local_c.shape == (2, 3, 4)
    assert cache.local_positions[0].tolist() == [2, 3, 4]
    assert cache.num_tokens_seen() == 5
    assert cache.memory_bytes() > 0

    cache.to(device=torch.device("cpu"), dtype=torch.float16)
    assert cache.compressed_kv.dtype == torch.float16
    assert cache.local_c.dtype == torch.float16
    assert cache.pending_c.dtype == torch.float16

    cache.reset()
    assert cache.compressed_kv is None
    assert cache.pending_c is None
    assert cache.local_c is None
    assert cache.num_tokens_seen() == 0


def test_hca_cache_does_not_flush_before_block_ready():
    cache = HCALayerCache(compression_factor=3)

    cache.append_token_state(token(), token(), pos())
    cache.flush_ready_blocks()

    assert cache.compressed_kv is None
    assert cache.pending_c.shape == (2, 1, 4)


def test_hca_cache_multiple_flushes_and_padding_invalid_block():
    cache = HCALayerCache(compression_factor=2)
    valid_false = torch.zeros(2, 1, dtype=torch.bool)

    cache.append_token_state(token(value=7), token(value=7), pos(value=0), valid_false)
    cache.append_token_state(token(value=7), token(value=7), pos(value=1), valid_false)
    cache.flush_ready_blocks()

    assert cache.compressed_kv.shape == (2, 1, 4)
    assert not cache.compressed_valid_mask.any().item()
    assert torch.equal(cache.compressed_kv, torch.zeros_like(cache.compressed_kv))
    assert cache.pending_c is None


def test_csa_cache_first_second_flush_pending_local_to_memory_reset():
    cache = CSALayerCache(compression_factor=2, local_window_size=2)

    assert cache.compressed_main is None
    assert cache.compressed_index is None
    assert cache.pending_a_c is None
    assert cache.previous_b_c is None

    cache.append_token_state(token(value=1), token(value=10), index_a_c_t=token(dim=3, value=2), index_b_c_t=token(dim=3, value=20), position_t=pos(value=0))
    assert cache.pending_a_c.shape == (2, 1, 4)
    assert cache.pending_b_c.shape == (2, 1, 4)
    assert cache.pending_index_a_c.shape == (2, 1, 3)

    cache.append_token_state(token(value=3), token(value=30), index_a_c_t=token(dim=3, value=4), index_b_c_t=token(dim=3, value=40), position_t=pos(value=1))
    cache.flush_ready_blocks()

    assert cache.compressed_main.shape == (2, 1, 4)
    assert cache.compressed_index.shape == (2, 1, 3)
    assert cache.previous_b_c.shape == (2, 2, 4)
    assert cache.previous_index_b_c.shape == (2, 2, 3)

    first_entry = cache.compressed_main.clone()
    cache.append_token_state(token(value=5), token(value=50), index_a_c_t=token(dim=3, value=6), index_b_c_t=token(dim=3, value=60), position_t=pos(value=2))
    cache.append_token_state(token(value=7), token(value=70), index_a_c_t=token(dim=3, value=8), index_b_c_t=token(dim=3, value=80), position_t=pos(value=3))
    cache.flush_ready_blocks()

    assert cache.compressed_main.shape == (2, 2, 4)
    assert cache.compressed_index.shape == (2, 2, 3)
    assert cache.compressed_valid_mask.shape == (2, 2)
    assert not torch.allclose(cache.compressed_main[:, 1:2], first_entry)
    assert cache.local_c.shape == (2, 2, 4)
    assert cache.local_positions[0].tolist() == [2, 3]
    assert cache.memory_bytes() > 0

    cache.to(device=torch.device("cpu"), dtype=torch.float16)
    assert cache.compressed_main.dtype == torch.float16
    assert cache.compressed_index.dtype == torch.float16
    assert cache.previous_b_c.dtype == torch.float16

    cache.reset()
    assert cache.compressed_main is None
    assert cache.compressed_index is None
    assert cache.pending_a_c is None
    assert cache.previous_b_c is None
    assert cache.num_tokens_seen() == 0


def test_csa_cache_incomplete_block_remains_pending():
    cache = CSALayerCache(compression_factor=3)

    for i in range(4):
        cache.append_token_state(token(value=i), position_t=pos(value=i))
    cache.flush_ready_blocks()

    assert cache.compressed_main.shape[1] == 1
    assert cache.pending_a_c.shape[1] == 1


def test_csa_cache_padding_first_block_invalid_zeroes_current_branch():
    cache = CSALayerCache(compression_factor=2)
    invalid = torch.zeros(2, 1, dtype=torch.bool)

    cache.append_token_state(token(value=4), position_t=pos(value=0), valid_mask_t=invalid)
    cache.append_token_state(token(value=4), position_t=pos(value=1), valid_mask_t=invalid)
    cache.flush_ready_blocks()

    assert not cache.compressed_valid_mask.any().item()
    assert torch.equal(cache.compressed_main, torch.zeros_like(cache.compressed_main))
