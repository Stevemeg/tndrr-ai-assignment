# AI/ML Internship Assignment

Three small, self-contained tasks. Each folder has its own README, `requirements.txt` and `results/` output.

| Folder | Task | Runs on |
|---|---|---|
| [`task1-finetune/`](task1-finetune/README.md) | LoRA fine-tuning of Qwen2.5-0.5B-Instruct to turn meeting/reminder requests into a fixed JSON format, plus a [7B memory estimate](task1-finetune/memory_calc.md) | Data: local · Training: Kaggle T4 GPU |
| [`task2-rag/`](task2-rag/README.md) | Retrieval-augmented QA over the NIST AI RMF 1.0, built from scratch (pypdf + MiniLM + FAISS + Ollama) and evaluated at two chunk sizes | Local (CPU is fine) + Ollama |
| [`task3-local-inference/`](task3-local-inference/README.md) | Speed / memory / quality benchmark of the same model at fp16 vs 4-bit (q4_K_M) with Ollama, plus a 20-user serving plan | Local + Ollama |

## Repository layout

```
.
├── README.md                     <- this file
├── .gitignore
├── task1-finetune/
│   ├── generate_data.py          <- synthetic dataset (400 train / 60 test, fixed seed)
│   ├── data/                     <- train.jsonl, test.jsonl
│   ├── task1_finetune.ipynb      <- Kaggle notebook: base eval -> LoRA -> fine-tuned eval
│   ├── memory_calc.md            <- full FT vs LoRA vs QLoRA memory for a 7B model
│   ├── results/                  <- produced by the notebook
│   ├── requirements.txt
│   └── README.md
├── task2-rag/
│   ├── data/NIST.AI.100-1.pdf    <- source document
│   ├── config.py                 <- shared settings
│   ├── chunking.py               <- PDF -> page text -> 150/500-word chunks
│   ├── index.py                  <- embeddings + FAISS IndexFlatIP, search helper
│   ├── eval_retrieval.py         <- recall@1 / recall@5
│   ├── ask.py                    <- retrieve + answer with citations via Ollama
│   ├── questions.json            <- 15 evaluation questions (filled in by hand)
│   ├── results/
│   ├── requirements.txt
│   └── README.md
└── task3-local-inference/
    ├── prompts.json              <- 10 fixed prompts
    ├── benchmark.py              <- fp16 vs q4_K_M through the Ollama HTTP API
    ├── results/
    ├── requirements.txt
    └── README.md
```

## Principles

* Plain Python, NumPy, PyTorch / Hugging Face and HTTP calls. No LangChain, LlamaIndex or TRL.
* Fixed seeds and pinned versions where it matters.
* All result numbers in the READMEs come from real runs. Placeholders mark what still has to be run or written.
