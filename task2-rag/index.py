"""Embed the chunks and build one FAISS index per chunk size.
Also has the small helpers that eval_retrieval.py and ask.py use to search.

Run:  python index.py      (after chunking.py)
Writes index/faiss_150.index and index/faiss_500.index
"""

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHUNK_SIZES, EMBED_MODEL, INDEX_DIR


def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def embed(model, texts):
    """Embed texts and scale every vector to length 1.
    For unit vectors the inner product equals cosine similarity."""
    vectors = model.encode(texts, batch_size=32, convert_to_numpy=True).astype("float32")
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def load_chunks(size):
    return json.loads((INDEX_DIR / f"chunks_{size}.json").read_text(encoding="utf-8"))


def load_index(size):
    return faiss.read_index(str(INDEX_DIR / f"faiss_{size}.index"))


def search(model, index, chunks, query, k=5):
    """Return the k most similar chunks (best first), each with its similarity score."""
    scores, ids = index.search(embed(model, [query]), k)
    return [{"score": float(score), **chunks[i]} for score, i in zip(scores[0], ids[0])]


def build_index(model, size):
    chunks = load_chunks(size)
    vectors = embed(model, [c["text"] for c in chunks])

    index = faiss.IndexFlatIP(vectors.shape[1])  # exact (brute-force) search by inner product
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR / f"faiss_{size}.index"))

    # The model reads at most max_seq_length word-piece tokens; anything after that is cut off.
    too_long = sum(len(model.tokenizer.tokenize(c["text"])) + 2 > model.max_seq_length for c in chunks)
    print(f"{size}-word chunks: {index.ntotal} vectors of dim {vectors.shape[1]} -> faiss_{size}.index "
          f"({too_long}/{len(chunks)} chunks are longer than {model.max_seq_length} tokens and get truncated)")


def main():
    model = load_embedder()
    for size in CHUNK_SIZES:
        build_index(model, size)


if __name__ == "__main__":
    main()
