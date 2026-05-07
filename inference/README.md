# DeepSeek-V4 Mini Inference

This folder contains the inference layer for the project. It is designed around the cache shapes implied by DeepSeek-style hybrid attention instead of pretending every layer is a plain GPT-style `past_key_values` tuple.

## Scope

Implemented in this pass:

- Inference configuration validation.
- Heterogeneous per-layer cache dataclasses:
  - `MHACache`
  - `HCALayerCache`
  - `CSALayerCache`
  - `DeepSeekV4InferenceCache`
- Sampling utilities:
  - greedy
  - temperature
  - top-k
  - top-p
  - repetition penalty
- Prompt prefill.
- Single-token decode step.
- Autoregressive generation.
- Optional MTP draft diagnostics.
- Cache memory and generation-speed summaries.
- High-level `inference_autoregresive(...)` wrapper.
- Notebook/debug audit wrapper: `audit_inference_pipeline(...)`.

The current decode path keeps generation behavior correct by using full-context recomputation while maintaining explicit cache state for inspection and future cache-aware attention integration. The attention modules can later add native `forward_decode(...)` methods without changing the public inference API.

## Files

```text
inference/
├── __init__.py
├── inference_config.py      # generation/cache configuration
├── cache_base.py            # common cache protocol
├── mha_cache.py             # token-level K/V cache
├── hca_cache.py             # compressed + local + pending HCA cache
├── csa_cache.py             # compressed main/index + local + pending/previous CSA cache
├── hybrid_cache.py          # whole-model heterogeneous cache
├── cache_utils.py           # dtype/device/tokenizer/cache helpers
├── prefill.py               # prompt processing and cache initialization
├── decode.py                # single-token decode step and cache update
├── generate.py              # autoregressive generation and wrapper API
├── sampling.py              # sampling/filtering utilities
├── mtp_decode.py            # optional MTP draft diagnostics
└── metrics.py               # cache and generation metrics
```

## Basic Usage

```python
from inference import inference_autoregresive

out = inference_autoregresive(
    model,
    prompt="key key_1 is value_4 question what is key_1 ? answer :",
    tokenizer=tokenizer,
    max_new_tokens=32,
    do_sample=False,
    eos_token_id=tokenizer.eos_id,
    pad_token_id=tokenizer.pad_id,
    return_cache_stats=True,
)

print(out["text"])
print(out["cache_stats"])
```

For a broader notebook audit:

```python
from inference import audit_inference_pipeline

audit = audit_inference_pipeline(
    model,
    prompt=[1, 2, 3, 4],
    max_new_tokens=8,
    do_sample=False,
    compare_logits=True,
)

print(audit["generation"]["sequences"])
print(audit["cache_stats"])
print(audit["full_vs_cached"])
```

The correctly spelled alias is also exported:

```python
from inference import inference_autoregressive
```

## Lower-Level API

```python
from inference import InferenceConfig, generate

cfg = InferenceConfig(
    max_new_tokens=64,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    return_cache_stats=True,
)

out = generate(model, input_ids=input_ids, inference_config=cfg)
```

## Current Limitation

Native attention-level cached decode is intentionally not wired yet because that requires adding `forward_decode(...)` methods to the existing attention/block modules. This folder still models the target cache structures now, so the future change is localized:

- `MHA.forward_decode(...)` can use `MHACache`.
- `HCA.forward_decode(...)` can use `HCALayerCache`.
- `CSA.forward_decode(...)` can use `CSALayerCache`.
- Block/model decode can then replace full-context recomputation behind the same public API.
