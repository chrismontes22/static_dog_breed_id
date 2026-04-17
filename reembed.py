"""
reembed.py  —  run ONCE locally before deploying the RAG service.

Converts dogdb.json (sentence-transformers embeddings) into two lean files:
  gemini_docs.json        — ids, documents, metadatas  (~text only, small)
  gemini_embeddings.npy   — float32 matrix (binary, ~34 MB vs ~787 MB JSON)

Usage:
    pip install google-genai numpy
    GEMINI_API_KEY=your_key python reembed.py

Both output files go in your /rag folder before deploying.

Why 512-dim?
  gemini-embedding-001 is Matryoshka-trained, so truncated dimensions still
  carry strong semantic meaning. 512-dim gives a good quality/size tradeoff:
    - 8x smaller than full 3072-dim
    - ~34 MB .npy on disk vs ~787 MB JSON
    - ~65 MB in RAM vs ~233 MB
"""

import json
import os
import time

import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL       = "gemini-embedding-001"  # text-embedding-004 deprecated Jan 2026
OUTPUT_DIMS       = 512                     # Matryoshka truncation — change to 256 to save more RAM
BATCH_SIZE        = 100                     # API hard max per call
CHECKPOINT        = "reembed_checkpoint.json"

INTER_BATCH_DELAY = 1.0   # seconds between successful batches
RETRY_BASE_DELAY  = 10.0  # seconds to wait on first 429
MAX_RETRIES       = 6     # doubles each time: 10→20→40→80→160→320s

# ── Init ──────────────────────────────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    raise SystemExit("Set the GEMINI_API_KEY environment variable before running.")

client = genai.Client(api_key=api_key)


# ── Helpers ───────────────────────────────────────────────────────────────────
def embed_batch_with_retry(texts: list[str]) -> list[list[float]]:
    """Embed up to 100 texts, retrying on 429 with exponential backoff."""
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=OUTPUT_DIMS,
                ),
            )
            return [emb.values for emb in response.embeddings]

        except APIError as e:
            if "429" in str(e.code) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"  Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                      f"Waiting {delay:.0f}s ...")
                time.sleep(delay)
                delay = min(delay * 2, 320)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries due to rate limiting.")


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"completed_batches": 0, "embeddings": []}


def save_checkpoint(completed_batches: int, embeddings: list) -> None:
    with open(CHECKPOINT, "w") as f:
        json.dump({"completed_batches": completed_batches, "embeddings": embeddings}, f)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if not os.path.exists("dogdb.json"):
        raise SystemExit("dogdb.json not found. Run your ChromaDB export script first.")

    with open("dogdb.json") as f:
        db = json.load(f)

    documents     = db["documents"]
    total_docs    = len(documents)
    total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"{total_docs} documents -> {total_batches} batches of up to {BATCH_SIZE}")
    print(f"Model: {EMBED_MODEL}  |  dims: {OUTPUT_DIMS}\n")

    # Resume from checkpoint if one exists
    state          = load_checkpoint()
    start_batch    = state["completed_batches"]
    all_embeddings = state["embeddings"]

    if start_batch > 0:
        print(f"Resuming from batch {start_batch + 1}/{total_batches} "
              f"({len(all_embeddings)} vectors already done)\n")

    # Embed batch by batch
    for batch_idx in range(start_batch, total_batches):
        start       = batch_idx * BATCH_SIZE
        end         = min(start + BATCH_SIZE, total_docs)
        batch_texts = documents[start:end]

        print(f"  Batch {batch_idx + 1}/{total_batches}  "
              f"(docs {start + 1}-{end}) ... ", end="", flush=True)

        t0               = time.time()
        batch_embeddings = embed_batch_with_retry(batch_texts)
        elapsed          = time.time() - t0

        all_embeddings.extend(batch_embeddings)
        save_checkpoint(batch_idx + 1, all_embeddings)
        print(f"done in {elapsed:.1f}s")

        if batch_idx < total_batches - 1:
            time.sleep(INTER_BATCH_DELAY)

    # ── Write output files ────────────────────────────────────────────────────

    # 1. Compact binary embeddings matrix
    matrix = np.array(all_embeddings, dtype=np.float32)
    np.save("gemini_embeddings.npy", matrix)
    npy_mb = matrix.nbytes / (1024 ** 2)

    # 2. Text/metadata as plain JSON (no embeddings — much smaller)
    docs_payload = {
        "ids":       db["ids"],
        "documents": db["documents"],
        "metadatas": db["metadatas"],
        "embed_model": EMBED_MODEL,
        "output_dims": OUTPUT_DIMS,
    }
    with open("gemini_docs.json", "w") as f:
        json.dump(docs_payload, f)
    docs_mb = os.path.getsize("gemini_docs.json") / (1024 ** 2)

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print(f"\nDone!")
    print(f"  gemini_embeddings.npy  {npy_mb:.1f} MB  ({matrix.shape[0]} x {matrix.shape[1]} float32)")
    print(f"  gemini_docs.json       {docs_mb:.1f} MB  (ids + documents + metadatas)")
    print(f"\nCopy BOTH files into your /rag folder before deploying.")


if __name__ == "__main__":
    main()