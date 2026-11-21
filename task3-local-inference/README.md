# Task 3 — Local inference and quantisation

The same model (Qwen2.5-1.5B-Instruct) is served by Ollama at two precisions, and speed, memory and output quality are compared:

| Ollama tag | Weights |
|---|---|
| `qwen2.5:1.5b-instruct-fp16` | 16-bit floats (no quantisation) |
| `qwen2.5:1.5b-instruct-q4_K_M` | 4-bit "K-quant", medium variant (llama.cpp / GGUF) |

## Files

| File | What it is |
|---|---|
| `prompts.json` | 10 fixed prompts: factual, reasoning, coding, summarisation, JSON output, creative, maths, translation, classification, instruction following |
| `benchmark.py` | Runs every prompt on both models through the Ollama HTTP API and saves the results |

## How `benchmark.py` measures

For each model:

1. **Warm-up.** One request loads the weights into memory, so load time is not counted in the speed numbers.
2. **10 prompts** through `POST /api/generate` with `stream: false` and the same options for both models: `temperature 0`, `seed 42`, `num_predict 256` (at most 256 new tokens).
3. **Speed** from Ollama's own timings (all durations are in nanoseconds):
   * generation speed = `eval_count / eval_duration` (output tokens per second)
   * prompt processing speed = `prompt_eval_count / prompt_eval_duration`
   * total time = `total_duration`
   * The summary row is the overall rate: all tokens ÷ all time over the 10 prompts.
4. **Memory** from `GET /api/ps` while the model is loaded. `size` = everything Ollama allocated for the model (weights + KV cache + compute buffers). `size_vram` = the part in GPU memory. On a machine without a supported GPU, `size_vram` is 0 and everything runs on the CPU.
5. **Unload** (`keep_alive: 0`) before the next model, so the two models never compete for memory.

Note: Ollama can reuse cached prompt tokens from the previous request. If that happens, `prompt_eval_count` counts only the tokens that were actually processed.

## How to run

1. Install Ollama from https://ollama.com/download and start it. It runs in the background on `http://localhost:11434`.
2. Pull both models:
   ```bash
   ollama pull qwen2.5:1.5b-instruct-fp16
   ollama pull qwen2.5:1.5b-instruct-q4_K_M
   ```
3. Run:
   ```bash
   cd task3-local-inference
   pip install -r requirements.txt
   python benchmark.py
   ```
4. Outputs in `results/`:
   * `benchmark.json`: all timings, token counts and memory for every prompt and model
   * `outputs_qwen2.5_1.5b-instruct-fp16.json`, `outputs_qwen2.5_1.5b-instruct-q4_K_M.json`: the generated answers
   * `comparison.md`: side-by-side table of speed and memory, overall and per prompt
   * `outputs_side_by_side.md`: both answers to each prompt, one under the other, for judging quality

Close other heavy programs while it runs, and write down your hardware (CPU, RAM, GPU) below. The numbers only mean something together with the hardware.

## Results

**Hardware**

| | |
|---|---|
| CPU | 11th Gen Intel Core i5-11320H @ 3.20 GHz, 4 cores / 8 logical processors |
| RAM | 15.79 GB (16,952,647,680 bytes) |
| GPU | Intel Iris Xe Graphics (integrated); no NVIDIA GPU |
| OS | Windows 11 Home Single Language 10.0.26300 |
| Ollama | 0.40.16 |
| Where the model ran | **CPU only**: `size_vram` = 0 for both models in `/api/ps` |

**Results** (from `results/comparison.md` / `results/benchmark.json`; temperature 0, seed 42, `num_predict` 256, after one warm-up request)

| Metric | fp16 | q4_K_M |
|---|---|---|
| Generation speed (tokens/s) | 12.3 | 28.1 |
| Prompt processing speed (tokens/s) | 139.9 | 224.5 |
| Total time, 10 prompts (s) | 77.6 | 35.9 |
| Output tokens, 10 prompts | 886 | 912 |
| Memory: size (GB) | 3.28 | 1.17 |
| Memory: size_vram (GB) | 0.00 | 0.00 |

* q4_K_M vs fp16: memory 3.28 → 1.17 GB (2.8× smaller), generation 12.3 → 28.1 tokens/s (2.3× faster).
* Answers cut off at the 256-token limit (`done_reason` = `length`): fp16 on prompts 2 and 3, q4_K_M on prompt 2.
* Per-prompt speeds and token counts are in `results/comparison.md`; both answers to every prompt are in `results/outputs_side_by_side.md`.

## My observations on quality

Compared answer by answer in `results/outputs_side_by_side.md`:

* q4_K_M matched fp16 on reasoning (both $0.05), coding, summarisation, maths (408, then 51) and classification (identical answers).
* The biggest drop was the creative poem: fp16 kept the 4-line limit, but q4_K_M wrote 12 lines with repeated lines.
* Small slips in q4_K_M: the JSON age as the string "29" instead of the number 29; the English word "meeting" in the French translation; a longer, padded factual answer (103 vs 51 tokens).
* q4_K_M was sometimes cleaner: for coding it gave just the code (176 tokens), while fp16 added an explanation and got cut off at 256. On the under-10-words tips, fp16 wrote 18/16/16 words and q4_K_M 5/5/10 (closer, but the third still isn't under 10).
* Shared 1.5B limits: both said Canberra but gave a wrong reason ("centre of the country"), both got cut off on the bat-and-ball problem, and both made grammar mistakes in the French.

## In my own words

### KV cache

* When an LLM generates text one token at a time, attention has to look back at every previous token.
* The KV cache stores the Key and Value tensors for all the tokens already processed, so the model doesn't recompute them for every new token. It trades extra memory for much faster generation.
* It grows with context length and with the number of users at the same time. That's what limits serving.

### Quantisation

* Weights are normally 16-bit floats (2 bytes per parameter). Shrinking them to 8-bit or 4-bit cuts memory by roughly 2x to 4x.
* My run: memory 3.28 → 1.17 GB (2.8x), generation 12.3 → 28.1 tok/s (2.3x).
* It's 2.8x and not 4x because q4_K_M uses ~4.5–5 bits per weight and keeps some parts at higher precision, and the measured size also includes the KV cache and buffers.
* It's faster because each new token means reading all the weights from RAM (this ran on CPU), and smaller weights = less data to read per token. Usually there is only a small hit to quality.

## Serving plan: 20 concurrent users on one 24 GB GPU

**Assumptions**

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| model | which model / precision is served | Qwen2.5-7B-Instruct, 4-bit AWQ weights (fp16 shown for comparison) | my choice |
| P | number of parameters | 7.6B | model weight files on Hugging Face (7,615,616,512) |
| b_w | bytes per weight | fp16 = 2; 4-bit AWQ checkpoint ≈ 5.6 GB in total | Qwen2.5-7B-Instruct-AWQ on Hugging Face (5.57 GB) |
| L | number of layers (`num_hidden_layers`) | 28 | model `config.json` |
| n_kv | number of KV heads (`num_key_value_heads`) | 4 (GQA) | model `config.json` |
| d_head | head dimension (`hidden_size / num_attention_heads`) | 128 (= 3584 / 28) | model `config.json` |
| b_kv | bytes per KV-cache element | 2 (fp16) | my choice (fp16 KV cache) |
| C | context length per user (prompt + answer) | up to 4,096 tokens | my choice |
| U | concurrent users | 20 | task |
| O | overhead: CUDA context, activations, framework buffers | ~2 GB | rough allowance |

Served with vLLM (continuous batching + PagedAttention).

**Formulas**

```
1. Weight memory          M_weights      = P × b_w
2. KV cache per token     kv_per_token   = 2 × L × n_kv × d_head × b_kv        (2 = one K and one V vector)
3. KV cache per user      kv_per_user    = kv_per_token × C
4. KV cache, all users    M_kv           = kv_per_user × U
5. Total                  M_total        = M_weights + M_kv + O
6. Check                  M_total  ≤  24 GB × (usable fraction, e.g. 0.9)
```

**My numbers**

```
M_weights    = fp16: 7.6B × 2 bytes ≈ 15.2 GB        4-bit AWQ: ≈ 5.6 GB
kv_per_token = 2 × 28 × 4 × 128 × 2 bytes = 57,344 bytes ≈ 56 KB
kv_per_user  = 4,096 × 56 KB ≈ 0.23 GB
M_kv         = 20 users ≈ 4.7 GB
M_total      = fp16: 15.2 + 4.7 + ~2 GB overhead ≈ 22 GB       4-bit: 5.6 + 4.7 + 2 ≈ 12 GB
fits?        = fp16: tight (≈ 22 GB is just above 0.9 × 24 = 21.6 GB)       4-bit: yes, lots of room
```

**My plan**

* 4-bit weights + vLLM, with `max_num_seqs=20`, `max_model_len=4096`, `gpu_memory_utilization=0.9`.
* Batching works because each decode step reads the weights once for all 20 users, so 20 users cost about the same per step as 1.
* PagedAttention stores the KV cache in small blocks, so no memory is wasted reserving the max length for every user.
* Limit: if every user wanted 32K context, the KV cache alone would be 20 × 32K × 56 KB ≈ 37.6 GB, which doesn't fit. Fixes: cap the context length, use an fp8 KV cache (half the size), or queue extra users.
