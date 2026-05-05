"""
query_rag.py  —  interactive test harness for gemini_embeddings.npy + gemini_docs.json

Usage:
    pip install google-genai numpy python-dotenv
    python query_rag.py                        # defaults: top-k=5, boost=0.15
    python query_rag.py --top-k 10             # return 10 results
    python query_rag.py --boost 0.3            # stronger breed boost
    python query_rag.py --boost 0.0            # disable breed boost entirely
    python query_rag.py --top-k 3 --loop       # keep querying until you type 'quit'

Breed boosting
--------------
If the query mentions a known breed (matched against all breeds found in metadata),
docs for that breed get a score bonus of `--boost` added to their cosine similarity.
Since cosine scores typically span ~0.3–0.9, a boost of 0.1–0.3 is a meaningful
nudge without completely overriding semantic relevance. Tune to taste.

Expects (in the same directory):
    gemini_embeddings.npy
    gemini_docs.json
    .env  with  GEMINI_API_KEY=your_key_here
"""

import argparse
import json
import os
import sys

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL       = "gemini-embedding-001"
OUTPUT_DIMS       = 512          # must match what reembed.py used
DEFAULT_TOP_K     = 5
DEFAULT_BOOST     = 0.15         # added to cosine score when breed matches query

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Query your Gemini RAG embeddings.")
parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help=f"Number of results to return (default: {DEFAULT_TOP_K})")
parser.add_argument("--boost", type=float, default=DEFAULT_BOOST,
                    help=f"Breed metadata boost added to cosine score (default: {DEFAULT_BOOST}). "
                         "Set to 0 to disable.")
parser.add_argument("--loop", action="store_true",
                    help="Keep prompting for queries until you type 'quit' or Ctrl-C")
args = parser.parse_args()

TOP_K      = args.top_k
BOOST      = args.boost

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key:
    sys.exit("Set GEMINI_API_KEY in your .env file before running.")

client = genai.Client(api_key=api_key)

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading embeddings index...")

if not os.path.exists("gemini_embeddings.npy"):
    sys.exit("gemini_embeddings.npy not found. Run reembed.py first.")
if not os.path.exists("gemini_docs.json"):
    sys.exit("gemini_docs.json not found. Run reembed.py first.")

embeddings = np.load("gemini_embeddings.npy")           # (N, 512) float32
with open("gemini_docs.json") as f:
    db = json.load(f)

documents = db["documents"]
metadatas = db.get("metadatas", [{}] * len(documents))
ids       = db.get("ids", list(range(len(documents))))

# Pre-normalise once so cosine sim is just a dot product at query time
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings_norm = embeddings / np.where(norms == 0, 1, norms)   # (N, 512)

# ── Build breed index ─────────────────────────────────────────────────────────
# Extract all unique breed values from metadata and build:
#   breed_per_doc : list[str | None]  — the breed for each doc (lowercased)
#   known_breeds  : set[str]          — all unique breeds (lowercased), sorted longest-first
#                                       so multi-word breeds ("golden retriever") match before
#                                       their substrings ("retriever") when we scan the query.

BREED_KEY = "breed"   # change if your metadata uses a different key

breed_per_doc: list[str | None] = [
    m.get(BREED_KEY, "").strip().lower() if isinstance(m, dict) else None
    for m in metadatas
]

known_breeds: list[str] = sorted(
    {b for b in breed_per_doc if b},
    key=len,
    reverse=True,   # longest first → "golden retriever" matched before "retriever"
)

if known_breeds:
    print(f"Breed index: {len(known_breeds)} breeds found in metadata  "
          f"(boost={BOOST} when matched)")
else:
    print("No breed metadata found — boost will have no effect.")

print(f"Loaded {len(documents)} documents  |  dims: {embeddings.shape[1]}")
print(f"Model: {EMBED_MODEL}  |  Returning top {TOP_K} results\n")


# ── Core search ───────────────────────────────────────────────────────────────
def detect_breed(query: str) -> str | None:
    """
    Return the first known breed found in the query (case-insensitive),
    or None if no breed is mentioned.
    Breeds are checked longest-first so 'golden retriever' beats 'retriever'.
    """
    q_lower = query.lower()
    for breed in known_breeds:
        if breed in q_lower:
            return breed
    return None


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string and return a normalised (512,) vector."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",       # <-- QUERY not DOCUMENT
            output_dimensionality=OUTPUT_DIMS,
        ),
    )
    vec = np.array(response.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def search(query: str, top_k: int, boost: float = 0.0) -> list[dict]:
    """Return the top_k most similar docs with blended vector + breed-boost scores."""
    q_vec        = embed_query(query)            # (512,)
    vector_scores = embeddings_norm @ q_vec      # (N,)  cosine similarity

    # ── Breed boost ───────────────────────────────────────────────────────────
    matched_breed = detect_breed(query) if boost > 0 else None
    if matched_breed:
        breed_bonus = np.array(
            [boost if b == matched_breed else 0.0 for b in breed_per_doc],
            dtype=np.float32,
        )
        final_scores = vector_scores + breed_bonus
    else:
        final_scores = vector_scores

    top_idx = np.argpartition(final_scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(final_scores[top_idx])[::-1]]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        results.append({
            "rank":          rank,
            "score":         float(final_scores[idx]),
            "vector_score":  float(vector_scores[idx]),
            "breed_matched": matched_breed,
            "boosted":       matched_breed is not None and breed_per_doc[idx] == matched_breed,
            "id":            ids[idx],
            "document":      documents[idx],
            "metadata":      metadatas[idx] if idx < len(metadatas) else {},
        })
    return results, matched_breed


def print_results(results: list[dict], matched_breed: str | None) -> None:
    if matched_breed:
        print(f"  ↑ Breed detected in query: '{matched_breed}' — "
              f"boosting matching docs by +{BOOST}\n")
    else:
        print("  (no breed detected in query — pure vector search)\n")

    for r in results:
        boost_tag = "  ⬆ boosted" if r["boosted"] else ""
        score_str = f"score={r['score']:.4f}"
        if r["boosted"]:
            score_str += f"  (vec={r['vector_score']:.4f} + {BOOST} breed boost)"
        print(f"  [{r['rank']}]  {score_str}  id={r['id']}{boost_tag}")
        if r["metadata"]:
            meta_str = "  |  ".join(f"{k}: {v}" for k, v in r["metadata"].items())
            print(f"       metadata: {meta_str}")
        doc_preview = r["document"][:300].replace("\n", " ")
        if len(r["document"]) > 300:
            doc_preview += " …"
        print(f"       {doc_preview}")
        print()


# ── Query loop ────────────────────────────────────────────────────────────────
def run_once() -> None:
    query = input("Enter query: ").strip()
    if not query:
        print("(empty query, skipping)")
        return
    print("Embedding query and searching...")
    results, matched_breed = search(query, TOP_K, BOOST)
    print_results(results, matched_breed)


def run_loop() -> None:
    print("Type your query and press Enter. Type 'quit' or press Ctrl-C to exit.\n")
    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if not query:
            continue
        print("Embedding query and searching...")
        results, matched_breed = search(query, TOP_K, BOOST)
        print_results(results, matched_breed)


if __name__ == "__main__":
    if args.loop:
        run_loop()
    else:
        run_once()