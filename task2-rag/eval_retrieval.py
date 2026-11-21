"""Measure retrieval quality for both chunk sizes: recall@1 and recall@5.

A question counts as a hit at rank k if one of the top-k chunks contains most of
the question's source passage: token overlap >= 0.6, where
    overlap = |content words of passage that also occur in chunk| / |content words of passage|

Run:  python eval_retrieval.py      (after index.py, with questions.json filled in)
Writes results/metrics.json and results/failures.json
"""

import argparse
import json
import re
from pathlib import Path

from chunking import clean_text
from config import CHUNK_SIZES, RESULTS_DIR, ROOT
from index import load_chunks, load_embedder, load_index, search

TOP_K = 5
HIT_THRESHOLD = 0.6

# Very common words are ignored, so the overlap measures shared *content* words.
# Otherwise any long chunk would "contain" words like "the", "of" and "and" from every passage.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "as", "at",
    "from", "is", "are", "be", "been", "was", "were", "it", "its", "this", "that", "these",
    "those", "can", "may", "will", "not", "such", "their", "they", "which", "also", "than",
    "into", "more", "other", "has", "have", "but", "if", "so", "do", "does",
}


def content_words(text):
    """Lower-cased words (letters/digits), without stopwords and 1-letter tokens."""
    words = re.findall(r"[a-z0-9]+", clean_text(text).lower())
    return {w for w in words if len(w) > 1 and w not in STOPWORDS}


def overlap(passage, chunk_text):
    """Share of the passage's content words that also appear in the chunk (0..1)."""
    passage_words = content_words(passage)
    if not passage_words:
        return 0.0
    return len(passage_words & content_words(chunk_text)) / len(passage_words)


def load_questions(path):
    questions = json.loads(path.read_text(encoding="utf-8"))
    empty = [q["id"] for q in questions if not q["question"].strip() or not q["source_passage"].strip()]
    if empty:
        raise SystemExit(f"Fill in {path.name} first. Entries with an empty question or passage: {empty}")
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "questions.json"))
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    n = len(questions)
    model = load_embedder()
    metrics, failures = {}, []

    for size in CHUNK_SIZES:
        index, chunks = load_index(size), load_chunks(size)
        hits_at_1 = hits_at_5 = 0
        print(f"\n=== {size}-word chunks ===")

        for q in questions:
            retrieved = search(model, index, chunks, q["question"], k=TOP_K)
            for r in retrieved:
                r["overlap"] = round(overlap(q["source_passage"], r["text"]), 3)
            is_hit = [r["overlap"] >= HIT_THRESHOLD for r in retrieved]

            hits_at_1 += is_hit[0]
            hits_at_5 += any(is_hit)
            first_hit = is_hit.index(True) + 1 if any(is_hit) else None
            print(f"Q{q['id']:>2}: hit rank = {first_hit or '-':>2} | best overlap {max(r['overlap'] for r in retrieved):.2f} | {q['question'][:70]}")

            if not any(is_hit):
                failures.append({
                    "chunk_size": size,
                    "id": q["id"],
                    "question": q["question"],
                    "source_passage": q["source_passage"],
                    "source_page": q["page"],
                    "top5": [
                        {"rank": rank, "score": round(r["score"], 4), "overlap": r["overlap"],
                         "page_start": r["page_start"], "page_end": r["page_end"], "text": r["text"]}
                        for rank, r in enumerate(retrieved, start=1)
                    ],
                })

        metrics[f"{size}_words"] = {
            "recall@1": round(hits_at_1 / n, 4),
            "recall@5": round(hits_at_5 / n, 4),
            "hits@1": hits_at_1,
            "hits@5": hits_at_5,
            "n_questions": n,
            "n_chunks": len(chunks),
        }

    metrics["settings"] = {"top_k": TOP_K, "hit_threshold": HIT_THRESHOLD, "chunk_sizes_and_overlap": CHUNK_SIZES}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULTS_DIR / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n| Chunk size | recall@1 | recall@5 |\n|---|---|---|")
    for size in CHUNK_SIZES:
        m = metrics[f"{size}_words"]
        print(f"| {size} words | {m['recall@1']:.2f} ({m['hits@1']}/{n}) | {m['recall@5']:.2f} ({m['hits@5']}/{n}) |")
    print(f"\nSaved results/metrics.json and results/failures.json ({len(failures)} failures)")


if __name__ == "__main__":
    main()
