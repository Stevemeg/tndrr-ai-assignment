"""Answer a question about the NIST AI RMF: retrieve chunks, then ask a local LLM (Ollama).

Run:  python ask.py "What are the characteristics of trustworthy AI?"
      python ask.py "..." --k 5 --chunk-size 500 --show-prompt
"""

import argparse

import requests

from config import CHUNK_SIZES, LLM_MODEL, OLLAMA_URL
from index import load_chunks, load_embedder, load_index, search

SYSTEM_PROMPT = (
    "You answer questions about the NIST AI Risk Management Framework (AI RMF 1.0). "
    "Use ONLY the numbered context passages. After every claim, cite the page(s) it came from, like [p. 12]. "
    "If the context does not contain the answer, say: \"I could not find this in the document.\""
)


def page_label(chunk):
    start, end = chunk["page_start"], chunk["page_end"]
    return f"p. {start}" if start == end else f"pp. {start}-{end}"


def build_prompt(question, retrieved):
    """Number every retrieved chunk and tag it with its page(s), so the model can cite them."""
    context = "\n\n".join(
        f"[{i}] ({page_label(chunk)}) {chunk['text']}" for i, chunk in enumerate(retrieved, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer with page citations:"


def ask_ollama(prompt):
    payload = {
        "model": LLM_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        # num_ctx: Ollama's default context window can be smaller than 5 x 500-word chunks,
        # and Ollama silently drops the start of a prompt that is too long.
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
    except requests.ConnectionError:
        raise SystemExit(f"Could not reach Ollama at {OLLAMA_URL}. Start Ollama and run: ollama pull {LLM_MODEL}")
    response.raise_for_status()
    return response.json()["response"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5, help="number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=150, choices=list(CHUNK_SIZES))
    parser.add_argument("--show-prompt", action="store_true", help="print the full prompt sent to the LLM")
    args = parser.parse_args()

    model = load_embedder()
    retrieved = search(model, load_index(args.chunk_size), load_chunks(args.chunk_size), args.question, k=args.k)
    prompt = build_prompt(args.question, retrieved)
    if args.show_prompt:
        print("=" * 30, "PROMPT", "=" * 30)
        print(prompt)
        print("=" * 68)

    print("\nRetrieved chunks:")
    for i, chunk in enumerate(retrieved, start=1):
        print(f"  [{i}] {page_label(chunk):<9} score={chunk['score']:.3f}  {chunk['text'][:90]}...")

    answer = ask_ollama(prompt)
    print(f"\nAnswer ({LLM_MODEL}):\n{answer}")


if __name__ == "__main__":
    main()
