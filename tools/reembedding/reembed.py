"""
reembed.py  —  run ONCE locally before deploying the RAG service.

Converts dogdb.json (sentence-transformers embeddings) into two lean files:
  gemini_docs.json        — ids, documents, metadatas  (~text only, small)
  gemini_embeddings.npy   — float32 matrix (binary, ~34 MB vs ~787 MB JSON)

Usage:
    pip install google-genai numpy python-dotenv
    python reembed.py

Create a .env file in the same directory with:
    GEMINI_API_KEY=your_key_here

Both output files go in your /rag folder before deploying.

Why 512-dim?
  gemini-embedding-001 is Matryoshka-trained — truncated dims still carry
  strong semantic meaning. 512-dim is 8x smaller than full 3072-dim.
"""

import collections
import json
import os
import time

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()   # reads GEMINI_API_KEY (and anything else) from .env into os.environ

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "gemini-embedding-001"   # text-embedding-004 deprecated Jan 2026
OUTPUT_DIMS  = 512                      # Matryoshka truncation
BATCH_SIZE   = 100                      # API hard max per call
CHECKPOINT   = "reembed_checkpoint.json"

# Rate limiter — stay just under the free-tier cap
RATE_LIMIT_CALLS    = 95    # leave 5 calls headroom below the 100 RPM cap
RATE_LIMIT_WINDOW   = 60.0  # seconds

# Retry on 429 (last-resort safety net, should rarely trigger)
RETRY_BASE_DELAY = 15.0
MAX_RETRIES      = 5

# ── Init ──────────────────────────────────────────────────────────────────────
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    raise SystemExit("Set the GEMINI_API_KEY environment variable before running.")

client = genai.Client(api_key=api_key)


# ── Sliding-window rate limiter ───────────────────────────────────────────────
# Keeps a deque of timestamps for recent calls. Before each call, drops
# timestamps older than the window, then sleeps if we're at the cap.

_call_times: collections.deque = collections.deque()

def _rate_limit_wait() -> None:
    """Block until making one more call won't exceed RATE_LIMIT_CALLS per window."""
    now = time.monotonic()

    # Drop timestamps outside the rolling window
    while _call_times and now - _call_times[0] >= RATE_LIMIT_WINDOW:
        _call_times.popleft()

    if len(_call_times) >= RATE_LIMIT_CALLS:
        # Sleep until the oldest call is outside the window
        sleep_for = RATE_LIMIT_WINDOW - (now - _call_times[0]) + 0.05
        if sleep_for > 0:
            print(f"\n  [rate limiter] window full — waiting {sleep_for:.1f}s ...",
                  end="", flush=True)
            time.sleep(sleep_for)
            # Re-prune after sleeping
            now = time.monotonic()
            while _call_times and now - _call_times[0] >= RATE_LIMIT_WINDOW:
                _call_times.popleft()

    _call_times.append(time.monotonic())


# ── Embed with retry (safety net only) ───────────────────────────────────────
def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed up to 100 texts.
    The rate limiter above should prevent 429s entirely; the retry here
    is a last-resort safety net for unexpected bursts or server blips.
    """
    _rate_limit_wait()

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
                print(f"\n  [429 safety net] attempt {attempt}/{MAX_RETRIES} — "
                      f"waiting {delay:.0f}s ...", end="", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 300)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries.")


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            ckpt = json.load(f)
        print(f"Resuming from checkpoint: "
              f"{ckpt['completed_batches']} batches done, "
              f"{len(ckpt['embeddings'])} vectors saved.\n")
        return ckpt
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
    est_minutes   = (total_batches / RATE_LIMIT_CALLS) * (RATE_LIMIT_WINDOW / 60)

    print(f"{total_docs} documents -> {total_batches} batches of up to {BATCH_SIZE}")
    print(f"Model: {EMBED_MODEL}  |  dims: {OUTPUT_DIMS}")
    print(f"Rate limit: {RATE_LIMIT_CALLS} calls/{RATE_LIMIT_WINDOW:.0f}s  "
          f"(est. {est_minutes:.1f} min total)\n")

    state          = load_checkpoint()
    start_batch    = state["completed_batches"]
    all_embeddings = state["embeddings"]

    t_start = time.monotonic()

    for batch_idx in range(start_batch, total_batches):
        start       = batch_idx * BATCH_SIZE
        end         = min(start + BATCH_SIZE, total_docs)
        batch_texts = documents[start:end]

        print(f"  [{batch_idx + 1:>3}/{total_batches}]  docs {start + 1}-{end} ... ",
              end="", flush=True)

        t0               = time.monotonic()
        batch_embeddings = embed_batch(batch_texts)
        elapsed          = time.monotonic() - t0

        all_embeddings.extend(batch_embeddings)
        save_checkpoint(batch_idx + 1, all_embeddings)

        # ETA
        batches_done  = batch_idx + 1 - start_batch
        batches_left  = total_batches - (batch_idx + 1)
        avg_pace      = (time.monotonic() - t_start) / batches_done
        eta_s         = avg_pace * batches_left
        eta_str       = f"{eta_s/60:.1f} min" if eta_s > 90 else f"{eta_s:.0f}s"

        print(f"done ({elapsed:.1f}s)  |  ETA {eta_str}")

    # ── Write output files ────────────────────────────────────────────────────
    matrix = np.array(all_embeddings, dtype=np.float32)
    np.save("gemini_embeddings.npy", matrix)
    npy_mb = matrix.nbytes / (1024 ** 2)

    docs_payload = {
        "ids":         db["ids"],
        "documents":   db["documents"],
        "metadatas":   db["metadatas"],
        "embed_model": EMBED_MODEL,
        "output_dims": OUTPUT_DIMS,
    }
    with open("gemini_docs.json", "w") as f:
        json.dump(docs_payload, f)
    docs_mb = os.path.getsize("gemini_docs.json") / (1024 ** 2)

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    total_time = (time.monotonic() - t_start) / 60
    print(f"\nDone in {total_time:.1f} min!")
    print(f"  gemini_embeddings.npy  {npy_mb:.1f} MB  "
          f"({matrix.shape[0]} x {matrix.shape[1]} float32)")
    print(f"  gemini_docs.json       {docs_mb:.1f} MB")
    print(f"\nCopy BOTH files into your /rag folder before deploying.")


if __name__ == "__main__":
    main()