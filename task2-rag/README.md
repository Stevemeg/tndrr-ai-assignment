# Task 2 — RAG from scratch over the NIST AI RMF 1.0

Question answering over **NIST AI 100-1, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*** (48-page PDF, in `data/`). No LangChain or LlamaIndex. The only libraries are pypdf, sentence-transformers (`all-MiniLM-L6-v2`), NumPy and FAISS, plus a local LLM through Ollama (`qwen2.5:1.5b`) for the final answer.

## Pipeline

```
PDF ──pypdf──> text per page ──> word chunks (150 and 500 words, ~15 % overlap, page numbers kept)
                                       │
                        all-MiniLM-L6-v2 embeddings, L2-normalised
                                       │
                         FAISS IndexFlatIP (one per chunk size)
                                       │
question ──embed──> top-k chunks ──> prompt with [p. N] tags ──> Ollama qwen2.5:1.5b ──> answer with citations
```

| File | What it does |
|---|---|
| `config.py` | Paths, model names, chunk sizes |
| `chunking.py` | Extracts text page by page with pypdf, cleans it (re-joins words hyphenated at line breaks, removes the running header and the "Page N" footer), and cuts word-based chunks: **150 words / 22 overlap** and **500 words / 75 overlap**. Each chunk stores `page_start` and `page_end`. |
| `index.py` | Embeds the chunks, normalises them to unit length (so inner product = cosine similarity), and builds a `faiss.IndexFlatIP` for each chunk size. Also has the `search()` helper. |
| `eval_retrieval.py` | recall@1 and recall@5 for both chunk sizes on `questions.json` → `results/metrics.json`, `results/failures.json` |
| `ask.py` | Retrieves the top-k chunks, builds a prompt with page citations, and answers with Ollama |
| `questions.json` | **Template with 15 empty entries. You fill it in.** |

## How to run

```bash
cd task2-rag
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python chunking.py              # -> index/chunks_150.json, index/chunks_500.json
python index.py                 # -> index/faiss_150.index, index/faiss_500.index (downloads MiniLM the first time)

# fill in questions.json (see below), then:
python eval_retrieval.py        # -> results/metrics.json, results/failures.json

# answering needs Ollama (https://ollama.com) running:
ollama pull qwen2.5:1.5b
python ask.py "What are the characteristics of trustworthy AI systems?"
python ask.py "How does the AI RMF define risk tolerance?" --chunk-size 500 --k 3 --show-prompt
```

`index.py` prints a "Token indices sequence length is longer than …" warning. That is expected: it comes from counting the tokens of the 500-word chunks (see the note below).

### Filling in `questions.json`

Each entry looks like this:

```json
{"id": 1, "question": "…", "source_passage": "…", "page": 12}
```

* `question`: a question in your own words, answerable from the document.
* `source_passage`: 1–3 sentences copied **verbatim** from the PDF that answer the question. About 20–60 words works best. Line-break hyphens are fine, because the same cleaning is applied to the passage.
* `page`: the **PDF page number** (1–48, as shown in the PDF viewer's page counter). This is also the number used in citations. In the main body it is the printed "Page N" footer + 5 (PDF page 6 = printed Page 1).
* Spread the questions over different sections and mix easy (keyword) and hard (paraphrased) questions.

### Hit criterion

A question is a hit at rank k if one of the top-k chunks contains most of the source passage:

  overlap = |content words of the passage that also appear in the chunk| / |content words of the passage| ≥ 0.6

"Content words" are lower-cased words without a short stopword list (*the, of, and, …*). Without that, long chunks would match any passage just through common words. `recall@1` = share of questions whose top-1 chunk is a hit, and `recall@5` = share with a hit anywhere in the top 5.

### Things to keep in mind when you read the results

These are properties of the setup, not results:

* `all-MiniLM-L6-v2` reads at most **256 word-piece tokens**. `index.py` reports how many chunks are longer and therefore truncated. For the 500-word chunks that is every chunk, so only roughly their first ~190 words influence the embedding.
* A larger chunk contains more words, so it passes the overlap test more easily once it is retrieved. It is also a less focused match for the question.
* A passage longer than the overlap (22 words for 150-word chunks) can be split between two neighbouring chunks.

## Results

From `results/metrics.json`: 15 questions, top-k = 5, hit threshold = 0.6 token overlap.

| Chunk size | # chunks | recall@1 | recall@5 |
|---|---|---|---|
| 150 words (22 overlap) | 120 | 0.60 (9/15) | 0.80 (12/15) |
| 500 words (75 overlap) | 36 | 0.47 (7/15) | 0.67 (10/15) |

`results/failures.json` lists 8 misses at k = 5: questions 7, 10 and 11 with 150-word chunks, and questions 2, 6, 10, 11 and 15 with 500-word chunks.

**What I observe: why 150 beat 500**

* 150-word chunks got 0.80 recall@5, vs 0.67 for 500-word chunks.
* The biggest reason is truncation: MiniLM caps out at 256 tokens (~190 words), so anything later in a 500-word chunk gets cut off and never embedded. That's why Q6 (bias) missed at 500, even though it was rank 1 at 150.
* Big chunks also dilute the meaning by squishing multiple topics into one averaged vector.
* The 60% word-overlap hit rule actually gives 500-word chunks an advantage (a bigger chunk is more likely to contain the passage), and 150 still won.

## Failure analysis (2 cases from `results/failures.json`)

### Failure 1: Q10, missed with both chunk sizes

* **Question:** "When should you pull the plug on building or using an AI system?"
* **Source passage (page 13, section 1.2.3 Risk Prioritization):** "In cases where an AI system presents unacceptable negative risk levels – such as where significant negative impacts are imminent, severe harms are actually occurring, or catastrophic risks are present – development and deployment should cease in a safe manner until risks can be sufficiently managed."
* **Retrieved instead (top 5, page · score · overlap):**
  * 150 words: p. 6 · 0.532 · 0.14; pp. 42–43 · 0.493 · 0.25; p. 24 · 0.492 · 0.18; p. 19 · 0.491 · 0.36; pp. 46–47 · 0.491 · 0.14
  * 500 words: pp. 6–7 · 0.526 · 0.32; pp. 46–48 · 0.516 · 0.25; pp. 30–31 · 0.490 · 0.29; pp. 4–6 · 0.477 · 0.29; pp. 7–8 · 0.463 · 0.32
  * No retrieved chunk comes from page 13.
* **Why I think retrieval failed:** It missed at both chunk sizes. "Pull the plug" is an idiom and shares no words with the actual text ("cease", "unacceptable negative risk"), so the small embedding model didn't connect them and grabbed generic "AI system" chunks from p. 6.
* **What would fix it:** BM25 wouldn't fix this, since there's no keyword overlap to match. Better fixes: LLM query rewriting / HyDE to turn casual phrasing into the document's language first, a stronger embedding model (bge/e5), or a re-ranker on the top 20.

### Failure 2: Q7, missed with 150-word chunks

* **Question:** "What is the MAP function supposed to do?"
* **Source passage (page 29, section 5.2 Map):** "The MAP function establishes the context to frame risks related to an AI system. The AI lifecycle consists of many interdependent activities involving a diverse set of actors (See Figure 3)."
* **Retrieved instead (top 5 with 150-word chunks, page · score · overlap):** p. 46 · 0.496 · 0.45; pp. 29–30 · 0.435 · 0.35; pp. 30–31 · 0.425 · 0.45; p. 30 · 0.396 · 0.35; p. 32 · 0.388 · 0.25
* With 500-word chunks this question was not a miss: it is not in `failures.json` for 500 words.
* **Why I think retrieval failed:** "MAP" shows up all over the document, so the embedding model couldn't tell the main definition on p. 29 apart from chunks about GOVERN or later MAP sections.
* **What would fix it:** This is the flip side of Q10. Hybrid search with BM25 would help here, by boosting exact matches like "MAP function" together with defining words. A re-ranker would also help pick out the actual definition.

## In my own words

### BM25

* BM25 scores chunks by exact word matches, giving extra weight to rare/specific terms and penalising very long chunks.
* Embeddings are good at general meaning but get fuzzy on exact names, codes and acronyms (like MAP). Combining both is hybrid search: meaning plus exact matches.

### Re-rankers

* A re-ranker works in two stages. First, fast search pulls a wider shortlist (the top 20–50 chunks). Then a cross-encoder reads the question and each chunk together, and scores how well that chunk actually answers the question.
* Reading the question and the chunk side by side is much more accurate than comparing two separate vectors, but it is slower, so it only runs on the shortlist.
