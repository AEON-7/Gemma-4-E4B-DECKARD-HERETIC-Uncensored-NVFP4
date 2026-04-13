# Gemma 4 E4B DECKARD HERETIC Uncensored NVFP4

EAGLE speculative decoding drafter for [Gemma 4 31B DECKARD HERETIC Uncensored NVFP4](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4). A 42-layer E4B (EAGLE for Blackwell) model quantized to NVFP4 AWQ using NVIDIA ModelOpt 0.42.0.

Designed for EAGLE-based speculative decoding on **NVIDIA DGX Spark** (GB10, SM 12.1) and other Blackwell-architecture GPUs.

## Model Details

| Property | Value |
|---|---|
| **Architecture** | Gemma 4 (E4B EAGLE Drafter) |
| **Base Model** | [DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking](https://huggingface.co/DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking) |
| **Target Model** | [AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4) |
| **Layers** | 42 (35 sliding-window + 7 full-attention) |
| **Hidden Size** | 2560 |
| **Attention Heads** | 8 (2 KV heads), head_dim=256, global_head_dim=512 |
| **Sliding Window** | 512 tokens |
| **Max Context** | 131,072 tokens |
| **Quantization** | NVFP4 AWQ (ModelOpt 0.42.0) |
| **Model Size** | 9.6 GB |
| **Vocabulary** | 262,144 tokens |

## Quick Start

### Prerequisites

1. Download the **target model** (31B DECKARD AWQ_FULL): [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4)
2. Download this **drafter model**: [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-E4B-DECKARD-HERETIC-Uncensored-NVFP4)
3. Three patched vLLM files (included in the [31B target repo](https://github.com/AEON-7/Gemma-4-31B-DECKARD-HERETIC-Uncensored-NVFP4)):
   - `eagle_patched.py` — Gemma 4 multimodal + multi-group KV cache support
   - `serving_chat_patched.py` — non-streaming reasoning parser fix
   - `modelopt_patched.py` — NVFP4 AWQ + FP8 NaN scrubbing

### Docker Compose (DGX Spark)

```yaml
services:
  vllm:
    image: ghcr.io/aeon-7/vllm-spark-gemma4-nvfp4-awq:latest
    container_name: vllm-deckard-31b-spec
    restart: unless-stopped
    network_mode: host
    volumes:
      - /path/to/deckard-31b-awq:/models/deckard
      - /path/to/e4b-deckard-nvfp4:/models/e4b-drafter
      - ./modelopt_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py
      - ./serving_chat_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py
      - ./eagle_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py
    environment:
      - VLLM_TEST_FORCE_FP8_MARLIN=1
      - VLLM_MARLIN_USE_ATOMIC_ADD=1
      - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      - TORCH_MATMUL_PRECISION=high
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    command:
      - bash
      - -c
      - |
        exec vllm serve /models/deckard \
          --served-model-name deckard-31b \
          --quantization modelopt \
          --dtype auto \
          --kv-cache-dtype fp8 \
          --tensor-parallel-size 1 \
          --max-model-len 131072 \
          --max-num-seqs 4 \
          --gpu-memory-utilization 0.85 \
          --trust-remote-code \
          --host 0.0.0.0 --port 8000 \
          --enable-chunked-prefill \
          --enable-prefix-caching \
          --enable-auto-tool-choice \
          --tool-call-parser gemma4 \
          --reasoning-parser gemma4 \
          --speculative-config '{"method":"draft_model","model":"/models/e4b-drafter","num_speculative_tokens":5,"quantization":"modelopt"}'
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Speculative Config Parameters

| Parameter | Value | Description |
|---|---|---|
| `method` | `draft_model` | EAGLE-based draft model speculative decoding |
| `model` | `/models/e4b-drafter` | Path to this E4B drafter model |
| `num_speculative_tokens` | `5` | Number of tokens the drafter proposes per step |
| `quantization` | `modelopt` | Required — tells vLLM the drafter uses NVFP4 format |

## Performance (DGX Spark)

Benchmarked on NVIDIA DGX Spark (GB10, SM 12.1, 128 GB unified memory) with 31B DECKARD AWQ_FULL target + this E4B drafter. 300 max tokens per request.

| Concurrent | Aggregate tok/s | Per-Request tok/s | Avg Latency (300 tok) |
|---:|---:|---:|---:|
| 1 | 7.6 | 8.9 | 39.4s |
| 2 | 21.7 | 10.8 | 27.7s |
| 4 | 42.7 | 10.7 | 28.1s |

Throughput scales linearly with concurrency. Zero errors across all test runs.

## Required vLLM Patches

Speculative decoding with Gemma 4 requires three patches to vLLM 0.19.1. All patches are available in the [31B target model repo](https://github.com/AEON-7/Gemma-4-31B-DECKARD-HERETIC-Uncensored-NVFP4).

### `eagle_patched.py`

1. **Multimodal guard removal** — vLLM 0.19.1 blocks all multimodal targets from spec decode, even with text-only drafters. The patch removes this check since downstream code handles it correctly.
2. **Gemma4 model whitelist** — Adds `Gemma4ForConditionalGeneration` for `image_token_id` → `image_token_index` mapping.
3. **Multi-group KV cache** — Gemma 4's heterogeneous attention (`head_dim=256` sliding window + `head_dim=512` global) creates two KV cache groups. The patch keys attention groups by `(backend_class, kv_cache_group_id)` to handle this.

### `serving_chat_patched.py`

Fixes the non-streaming reasoning parser for Gemma 4. The `<|channel>` / `<channel|>` delimiters (tokens 100/101) are stripped by `skip_special_tokens=True`, breaking reasoning extraction. The patch re-decodes from raw token IDs when text-based extraction fails.

### `modelopt_patched.py`

FP8 NaN scrubbing + NVFP4_AWQ quant_algo support + AWQ pre_quant_scale handling.

## Heterogeneous Attention Architecture

This E4B drafter mirrors the Gemma 4 heterogeneous attention design:

- **35 sliding-window layers** — `head_dim=256`, sliding window of 512 tokens, default RoPE (`theta=10000`)
- **7 full-attention layers** — `head_dim=512`, global attention, proportional RoPE (`theta=1000000`, `partial_rotary_factor=0.25`)

This creates two distinct KV cache groups within the drafter, which the `eagle_patched.py` fix handles by mapping each layer to its correct group.

## Related Models

| Model | Type | Size | Link |
|---|---|---|---|
| **Gemma 4 31B DECKARD AWQ_FULL** (target) | Dense NVFP4 | 20.5 GB | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4) \| [GitHub](https://github.com/AEON-7/Gemma-4-31B-DECKARD-HERETIC-Uncensored-NVFP4) |
| **Gemma 4 31B DECKARD SVDQuant** | Dense NVFP4 | 20.9 GB | [HuggingFace](https://huggingface.co/AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4-SVDQuant) |
| **SuperGemma4 26B MoE** | MoE NVFP4 | 15.3 GB | [HuggingFace](https://huggingface.co/AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4) \| [GitHub](https://github.com/AEON-7/supergemma4-26b-abliterated-multimodal-nvfp4) |
| **vLLM AWQ Container** | Docker | — | [GHCR](https://github.com/users/AEON-7/packages/container/package/vllm-spark-gemma4-nvfp4-awq) |

## Hardware Requirements

- **Target + Drafter combined**: ~30 GB (20.5 GB target + 9.6 GB drafter)
- **Recommended**: NVIDIA DGX Spark (128 GB unified memory) or any GPU with >= 40 GB VRAM
- **Optimal**: Multi-GPU setup with drafter on a separate device for maximum speedup

## License

This model inherits the [Gemma license](https://ai.google.dev/gemma/terms) from Google.
