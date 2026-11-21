"""Benchmark one model at two precisions (fp16 vs 4-bit q4_K_M) with the Ollama HTTP API.

For each model: warm up once, run the 10 prompts from prompts.json with fixed
settings, record speed (from Ollama's own timings) and memory (from /api/ps).

Run:  python benchmark.py      (Ollama running, both models pulled)
Writes results/benchmark.json, results/outputs_<model>.json, results/comparison.md
"""

import json
import time
from pathlib import Path

import requests

OLLAMA = "http://localhost:11434"
MODELS = ["qwen2.5:1.5b-instruct-fp16", "qwen2.5:1.5b-instruct-q4_K_M"]
# Same settings for every request: greedy-like decoding, fixed seed, max 256 new tokens
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 256}

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
NS_PER_S = 1e9  # Ollama reports every duration in nanoseconds
GB = 1e9


def check_ollama():
    """Fail early with a clear message if Ollama is not running or a model is missing."""
    try:
        tags = requests.get(f"{OLLAMA}/api/tags", timeout=10).json()
    except requests.ConnectionError:
        raise SystemExit(f"Ollama is not running at {OLLAMA}. Start the Ollama app (or run `ollama serve`).")
    installed = {m["name"] for m in tags.get("models", [])}
    missing = [m for m in MODELS if m not in installed]
    if missing:
        raise SystemExit("Missing models. Run:\n" + "\n".join(f"  ollama pull {m}" for m in missing))


def generate(model, prompt):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": OPTIONS}
    response = requests.post(f"{OLLAMA}/api/generate", json=payload, timeout=900)
    response.raise_for_status()
    return response.json()


def tokens_per_second(count, duration_ns):
    return count / (duration_ns / NS_PER_S) if duration_ns else None


def loaded_memory(model):
    """/api/ps lists the models currently in memory.
    size = total bytes of the loaded model (weights + KV cache + compute buffers),
    size_vram = how much of that is in GPU memory (0 means it runs fully on the CPU)."""
    for m in requests.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", []):
        if model in (m.get("name"), m.get("model")):
            return {"size_bytes": m["size"], "size_vram_bytes": m["size_vram"]}
    return {"size_bytes": None, "size_vram_bytes": None}


def unload(model):
    """keep_alive=0 unloads the model, so the next model is measured on its own."""
    requests.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)


def benchmark_model(model, prompts):
    print(f"\n=== {model} ===")
    warmup = generate(model, "Say hello in one word.")  # loads the weights, so load time is not in the results
    print(f"warm-up done (load time {warmup.get('load_duration', 0) / NS_PER_S:.1f}s)")

    runs = []
    for p in prompts:
        start = time.perf_counter()
        r = generate(model, p["prompt"])
        wall_s = time.perf_counter() - start

        run = {
            "id": p["id"],
            "category": p["category"],
            "prompt_tokens": r.get("prompt_eval_count", 0),
            "output_tokens": r["eval_count"],
            "prompt_eval_s": r.get("prompt_eval_duration", 0) / NS_PER_S,
            "eval_s": r["eval_duration"] / NS_PER_S,
            "prompt_tokens_per_s": tokens_per_second(r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 0)),
            "gen_tokens_per_s": tokens_per_second(r["eval_count"], r["eval_duration"]),
            "total_s": r["total_duration"] / NS_PER_S,
            "wall_s": wall_s,
            "done_reason": r.get("done_reason"),  # "length" = stopped at num_predict
            "response": r["response"],
        }
        runs.append(run)
        print(f"  #{p['id']:>2} {p['category']:<22} {run['output_tokens']:>4} tokens  "
              f"{run['gen_tokens_per_s']:6.1f} tok/s  {run['total_s']:5.1f}s")

    memory = loaded_memory(model)  # read while the model is still loaded
    unload(model)

    summary = {
        # overall rate = all tokens / all time (long answers count more than short ones)
        "gen_tokens_per_s": sum(r["output_tokens"] for r in runs) / sum(r["eval_s"] for r in runs),
        "prompt_tokens_per_s": tokens_per_second(sum(r["prompt_tokens"] for r in runs),
                                                 sum(r["prompt_eval_s"] for r in runs) * NS_PER_S),
        "total_s": sum(r["total_s"] for r in runs),
        "output_tokens": sum(r["output_tokens"] for r in runs),
        **memory,
    }
    return {"summary": summary, "runs": runs}


def fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def gb(num_bytes):
    return None if num_bytes is None else num_bytes / GB


def short(model):
    """'qwen2.5:1.5b-instruct-q4_K_M' -> 'q4_K_M'"""
    return model.rsplit("-", 1)[-1]


def comparison_markdown(results):
    a, b = MODELS
    sa, sb = results[a]["summary"], results[b]["summary"]
    lines = [
        f"| Metric | {a} | {b} |",
        "|---|---|---|",
        f"| Generation speed (tokens/s) | {fmt(sa['gen_tokens_per_s'])} | {fmt(sb['gen_tokens_per_s'])} |",
        f"| Prompt processing speed (tokens/s) | {fmt(sa['prompt_tokens_per_s'])} | {fmt(sb['prompt_tokens_per_s'])} |",
        f"| Total time, 10 prompts (s) | {fmt(sa['total_s'])} | {fmt(sb['total_s'])} |",
        f"| Output tokens, 10 prompts | {sa['output_tokens']} | {sb['output_tokens']} |",
        f"| Memory: size (GB) | {fmt(gb(sa['size_bytes']), 2)} | {fmt(gb(sb['size_bytes']), 2)} |",
        f"| Memory: size_vram (GB) | {fmt(gb(sa['size_vram_bytes']), 2)} | {fmt(gb(sb['size_vram_bytes']), 2)} |",
        "",
        f"| # | Category | tok/s {short(a)} | tok/s {short(b)} | output tokens {short(a)} | output tokens {short(b)} |",
        "|---|---|---|---|---|---|",
    ]
    for ra, rb in zip(results[a]["runs"], results[b]["runs"]):
        lines.append(f"| {ra['id']} | {ra['category']} | {fmt(ra['gen_tokens_per_s'])} | {fmt(rb['gen_tokens_per_s'])} "
                     f"| {ra['output_tokens']} | {rb['output_tokens']} |")
    return "\n".join(lines) + "\n"


def side_by_side_markdown(results, prompts):
    """Both answers to every prompt, one under the other, for judging quality by eye."""
    parts = []
    for i, p in enumerate(prompts):
        parts.append(f"## {p['id']}. {p['category']}\n\n**Prompt:** {p['prompt']}\n")
        for model in MODELS:
            parts.append(f"**{model}:**\n\n```\n{results[model]['runs'][i]['response'].strip()}\n```\n")
    return "\n".join(parts)


def main():
    check_ollama()
    prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(exist_ok=True)

    results = {}
    for model in MODELS:
        results[model] = benchmark_model(model, prompts)
        outputs = [{"id": r["id"], "category": r["category"], "response": r["response"]} for r in results[model]["runs"]]
        out_path = RESULTS_DIR / f"outputs_{model.replace(':', '_')}.json"
        out_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")

    benchmark = {"options": OPTIONS, "models": results}
    (RESULTS_DIR / "benchmark.json").write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    table = comparison_markdown(results)
    (RESULTS_DIR / "comparison.md").write_text(table, encoding="utf-8")
    (RESULTS_DIR / "outputs_side_by_side.md").write_text(side_by_side_markdown(results, prompts), encoding="utf-8")

    print("\n" + table)
    print("Saved results/benchmark.json, results/outputs_*.json, results/comparison.md, results/outputs_side_by_side.md")


if __name__ == "__main__":
    main()
