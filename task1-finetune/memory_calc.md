# GPU memory estimate for fine-tuning a 7B model

This is a back-of-the-envelope estimate built from stated assumptions, not a measurement. Real numbers also depend on the framework, the attention kernel, memory fragmentation and the CUDA context.

## Assumptions

| Symbol | Value | Meaning |
|---|---|---|
| P | 7.0 × 10⁹ | total parameters (rounded; e.g. Llama-2-7B has 6.74B) |
| L | 32 | transformer layers |
| h | 4096 | hidden size |
| a | 32 | attention heads; standard multi-head attention, so k_proj and v_proj are also 4096 × 4096 |
| V | 32,000 | vocabulary size (only used for the logits note) |
| r | 16 | LoRA rank, adapters on q_proj, k_proj, v_proj, o_proj |
| b × s | 1 × 2048 | micro-batch size × sequence length used for the activation estimate |
| optimizer | AdamW | keeps 2 extra fp32 values per trained parameter (m and v) |

Units: 1 GB = 10⁹ bytes. Divide by 1.074 to get GiB, which is what `nvidia-smi` and PyTorch report.

---

## Step 1 — Bytes per parameter

| What is stored | (a) Full FT, mixed precision | (b) LoRA, frozen 16-bit base | (c) QLoRA, 4-bit base |
|---|---|---|---|
| Base weights used in forward/backward | 2 B (fp16/bf16) | 2 B (fp16/bf16) | ~0.5 B (4-bit NF4) + quantisation constants |
| fp32 master copy of weights | 4 B | – (frozen) | – (frozen) |
| Gradients | 2 B (fp16/bf16) | – (frozen) | – (frozen) |
| AdamW m + v (fp32) | 4 + 4 = 8 B | – (frozen) | – (frozen) |
| **Per base parameter** | **16 B** | **2 B** | **~0.52 B** |
| Per *trainable LoRA* parameter (fp32 weight + fp32 grad + m + v) | – | 4 + 4 + 8 = 16 B | 16 B |

Why the full fine-tuning row is 16 B (the standard accounting from the ZeRO paper, Rajbhandari et al., 2020): mixed precision runs the maths in 16-bit (2 B weights, 2 B gradients). The optimizer still updates an fp32 master copy (4 B), because tiny updates would vanish in fp16. AdamW also keeps two fp32 running averages per weight (8 B).

Why QLoRA is ~0.52 B: NF4 stores 4 bits = 0.5 B per weight. Each block of 64 weights also stores one scale ("absmax"). With *double quantisation* the scale is stored in 8 bits, plus one fp32 constant per 256 scales:
8/64 + 32/(64·256) ≈ 0.127 bits per parameter ≈ 0.016 B. Total ≈ 0.5 + 0.016 ≈ 0.516 B per parameter. Without double quantisation it is 0.5 + 32/64/8 = 0.5625 B.

---

## Step 2 — LoRA adapter parameter count (7B, r = 16, q/k/v/o)

LoRA replaces the update of a weight `W` (d_out × d_in) with `B·A`, where `A` is r × d_in and `B` is d_out × r.

* Parameters per adapted matrix = r·d_in + d_out·r = r·(d_in + d_out) = 16 × (4096 + 4096) = **131,072**
* Per layer, 4 matrices (q, k, v, o) = 4 × 131,072 = **524,288**
* All 32 layers = 32 × 524,288 = **16,777,216 ≈ 16.8M trainable parameters**
* As a share of the model: 16.8M / 7,000M ≈ **0.24 %**

Memory for the adapters with AdamW (16 B per trainable parameter):
16,777,216 × 16 B = 268,435,456 B ≈ **0.27 GB**

> Note: models with grouped-query attention are smaller here. Mistral-7B, for example, has 8 KV heads, so k_proj and v_proj are 4096 → 1024. That gives 16 × (4096+4096) × 2 + 16 × (4096+1024) × 2 = 425,984 per layer and ≈ 13.6M in total.

---

## Step 3 — Static memory (weights + gradients + optimizer)

| | Calculation | Result |
|---|---|---|
| (a) Full FT | 7.0e9 × 16 B | **112 GB** |
| (b) LoRA | 7.0e9 × 2 B + 0.27 GB | 14.0 + 0.27 = **≈ 14.3 GB** |
| (c) QLoRA | 7.0e9 × 0.516 B + 0.27 GB | 3.61 + 0.27 = **≈ 3.9 GB** |

Breakdown of (a): 14 GB fp16 weights + 14 GB fp16 gradients + 28 GB fp32 master weights + 56 GB AdamW states.

---

## Step 4 — Rough activation estimate

During training, the forward pass keeps intermediate results ("activations") so that backward can compute gradients. LoRA and QLoRA still need almost all of them: the gradient for an adapter in layer 1 has to flow back through all 32 frozen layers.

A common estimate for one transformer layer with 16-bit activations is from Korthikanti et al., 2022, *Reducing Activation Recomputation in Large Transformer Models*:

  activations per layer ≈ s·b·h·(34 + 5·a·s/h) bytes

* The **34·s·b·h** part covers the inputs of the projections, the MLP and the norms:
  34 × 2048 × 1 × 4096 = 285,212,672 B ≈ **0.285 GB per layer**
* The **5·a·s²·b** part is the stored s × s attention scores:
  5 × 32 × 2048² × 1 = 671,088,640 B ≈ **0.67 GB per layer**. FlashAttention / PyTorch SDPA do *not* store this matrix, so this term is ≈ 0 with them.

| Setting (b = 1, s = 2048) | Calculation | Activations |
|---|---|---|
| Naive attention, no checkpointing | 32 × (0.285 + 0.671) GB | ≈ 30.6 GB |
| FlashAttention / SDPA, no checkpointing | 32 × 0.285 GB | **≈ 9.1 GB** |
| FlashAttention + gradient checkpointing | store only each layer's input (2·s·b·h = 16.8 MB × 32 = 0.54 GB) + recompute one layer at a time (0.285 GB) | **≈ 0.8 GB** |

Plus logits: b·s·V values = 2048 × 32,000 = 65.5M. In fp32 that is 0.26 GB, and the loss and its gradient add a couple more copies, so allow roughly 0.5–1 GB.

Activations grow linearly with batch size b and, with FlashAttention, about linearly with sequence length s.

---

## Step 5 — Totals

Using FlashAttention/SDPA. The last column adds ~1 GB for logits and the CUDA context, which is itself only a rough figure.

| Method | Static | + activations, no checkpointing (9.1 GB) | + activations, gradient checkpointing (0.8 GB) | Realistic GPU |
|---|---|---|---|---|
| (a) Full FT + AdamW | 112 GB | ≈ 121 GB (+~1) | ≈ 113 GB (+~1) | several 80 GB GPUs with sharding (ZeRO/FSDP) or CPU offload |
| (b) LoRA, 16-bit base | 14.3 GB | ≈ 23.4 GB (+~1) | ≈ 15.1 GB (+~1) | 24 GB GPU with checkpointing; too tight for a 16 GB T4 |
| (c) QLoRA, 4-bit base | 3.9 GB | ≈ 13.0 GB (+~1) | ≈ 4.7 GB (+~1) | fits a 16 GB T4 |

### Key takeaways

1. In full fine-tuning, the **optimizer states and master weights** dominate: 84 of the 112 GB. They are not the weights themselves.
2. LoRA removes gradients and optimizer states for the frozen 99.76 % of parameters. What remains is mainly the **16-bit base weights** (2 B/param).
3. QLoRA shrinks those base weights 4× further (≈ 0.5 B/param). After that, **activations** become the main cost, and gradient checkpointing controls them.

### Caveats

* bitsandbytes usually keeps the embeddings and `lm_head` in 16-bit. With V = 32,000 that is 2 × 32,000 × 4096 ≈ 0.26B parameters, about +0.4 GB in (c).
* In QLoRA, each layer's weights are de-quantised to 16-bit just in time for the matmul. Only one layer is materialised at a time, so this is small (tens of MB).
* The "34" constant assumes a GPT-style MLP with dropout. Llama-style models (SwiGLU, no dropout) differ somewhat, so treat activations as ±30 %.
* PyTorch's caching allocator reserves more than it uses. Fragmentation can add 5–10 %.
