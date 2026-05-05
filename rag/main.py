"""
main.py — Multi-agent RAG backend for Sniff & Tell
Deploy on Hugging Face Spaces (Docker)

Pipeline
────────
  1. Guard agent    — verifies query is dog-related & family-safe  (Gemini)
  2. Vector store   — retrieves top-N docs from ChromaDB           (SentenceTransformers)
  3. Eval agent     — decides if VS can answer; if yes, cleans it  (Gemini)
  4. Tavily search  — fallback when VS is insufficient
  5. Web agent      — cleans Tavily snippets into a final answer   (Gemini)

Returns
────────
  { "response": "...", "source": "vector" | "tavily" | "blocked" }

Swapping to Google embeddings later
────────────────────────────────────
  See EMBED_BACKEND below. Change it to "google" and fill in
  get_query_embedding_google(). No other code needs to change.
"""

import os
import logging
from contextlib import asynccontextmanager

import chromadb
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from tavily import TavilyClient

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — change these freely
# ═══════════════════════════════════════════════════════════════════════════════

GEMINI_MODEL       = "gemini-2.0-flash-lite"   # ← verify model name in Google AI Studio
                                                 #   user requested "Gemini 3.1 Flash Lite";
                                                 #   nearest available is gemini-2.0-flash-lite

N_VECTOR_RESULTS   = 3    # ← how many chunks to pull from the vector store
N_TAVILY_RESULTS   = 3    # ← how many Tavily snippets to feed the cleanup agent

CHROMA_PATH        = "vectordb"      # folder containing the ChromaDB persistent store
CHROMA_COLLECTION  = "dogdb"

# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING PROVIDER — swap here when you move to Google embeddings
# ═══════════════════════════════════════════════════════════════════════════════

EMBED_BACKEND = "sentence_transformers"   # ← change to "google" when ready

ST_MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
GOOGLE_EMBED_MODEL = "gemini-embedding-001"   # used only when EMBED_BACKEND = "google"
GOOGLE_EMBED_DIMS  = 512                      # Matryoshka truncation (matches reembed.py)

# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP — load heavy objects once
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sniff_tell")

_st_model:    SentenceTransformer | None = None
_chroma_col:  chromadb.Collection | None = None
_gemini:      genai.Client | None = None
_tavily:      TavilyClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _st_model, _chroma_col, _gemini, _tavily

    log.info("Loading SentenceTransformer model…")
    if EMBED_BACKEND == "sentence_transformers":
        _st_model = SentenceTransformer(ST_MODEL_NAME)

    log.info("Connecting to ChromaDB…")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    _chroma_col = client.get_or_create_collection(CHROMA_COLLECTION)
    log.info(f"  Vector store ready — {_chroma_col.count()} documents")

    log.info("Initialising Gemini client…")
    _gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    log.info("Initialising Tavily client…")
    _tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    log.info("All services ready ✓")
    yield
    log.info("Shutting down.")


app = FastAPI(title="Sniff & Tell RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    dog_breed: str
    question:  str

class AskResponse(BaseModel):
    response: str
    source:   str   # "vector" | "tavily" | "blocked"

# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING — single function to swap
# ═══════════════════════════════════════════════════════════════════════════════

def get_query_embedding(text: str) -> list[float]:
    """
    Returns a query embedding vector.

    To switch to Google embeddings:
      1. Set EMBED_BACKEND = "google" at the top.
      2. Make sure GOOGLE_EMBED_DIMS matches what was used in reembed.py.
      3. Done — no other changes needed.
    """
    if EMBED_BACKEND == "sentence_transformers":
        vec = _st_model.encode([text])[0]
        return vec.tolist()

    elif EMBED_BACKEND == "google":
        resp = _gemini.models.embed_content(
            model=GOOGLE_EMBED_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=GOOGLE_EMBED_DIMS,
            ),
        )
        return resp.embeddings[0].values

    else:
        raise ValueError(f"Unknown EMBED_BACKEND: {EMBED_BACKEND!r}")

# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _ask_gemini(prompt: str) -> str:
    resp = _gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return resp.text.strip()

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — Guard
# ═══════════════════════════════════════════════════════════════════════════════

def agent_guard(breed: str, question: str) -> bool:
    """
    Returns True only when the query is about dogs AND is family-appropriate.
    Gemini answers YES or NO; anything other than YES is treated as blocked.
    """
    prompt = f"""You are a content moderator for a family-friendly dog information app.

A user is asking about the dog breed "{breed}" and their question is:
"{question}"

Answer with exactly ONE word:
  YES — if the question is relevant to dogs or pets and is completely family-appropriate.
  NO  — if the question is off-topic, harmful, adult, or inappropriate in any way."""

    answer = _ask_gemini(prompt).upper()
    log.info(f"[Guard] breed={breed!r} → {answer!r}")
    return answer.startswith("YES")

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — Vector-store evaluator + answer cleaner
# ═══════════════════════════════════════════════════════════════════════════════

def agent_eval_and_answer(breed: str, question: str, context_docs: list[str]) -> str | None:
    """
    Receives the raw documents from the vector store.
    Returns a clean, user-facing answer, or None if the context is insufficient.
    """
    context = "\n\n---\n\n".join(context_docs)

    prompt = f"""You are a friendly dog expert working inside a dog breed app.

The user wants to know about the breed "{breed}" and asked:
"{question}"

Below are raw excerpts retrieved from a dog knowledge base:
════════════════════════════════════
{context}
════════════════════════════════════

Your task:
  • If the excerpts contain enough information to answer the question,
    write a clear, warm, helpful answer in 2–4 sentences. Start directly
    with the answer — no preamble.
  • If the excerpts do NOT contain enough information, reply with exactly
    the single word: INSUFFICIENT"""

    answer = _ask_gemini(prompt)
    if answer.upper().strip() == "INSUFFICIENT":
        log.info("[Eval] Vector store insufficient — falling back to Tavily")
        return None
    log.info("[Eval] Vector store answered the question ✓")
    return answer

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3 — Tavily results cleaner
# ═══════════════════════════════════════════════════════════════════════════════

def agent_clean_tavily(breed: str, question: str, tavily_results: list[dict]) -> str:
    """
    Turns raw Tavily search snippets into a friendly user-facing answer.
    """
    snippets = "\n\n".join(
        f"[{i+1}] {r.get('content', '').strip()[:500]}"
        for i, r in enumerate(tavily_results[:N_TAVILY_RESULTS])
    )

    prompt = f"""You are a friendly dog expert.

The user asked about the breed "{breed}":
"{question}"

Here are web search results:
════════════════════════════════════
{snippets}
════════════════════════════════════

Write a clear, warm, helpful answer in 2–4 sentences based on the above.
Start directly with the answer — no preamble."""

    log.info("[Web agent] Cleaning Tavily results ✓")
    return _ask_gemini(prompt)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    breed    = req.dog_breed.strip()
    question = req.question.strip()

    if not breed or not question:
        raise HTTPException(status_code=422, detail="breed and question are required")

    # ── Step 1: Guard ──────────────────────────────────────────────────────────
    if not agent_guard(breed, question):
        return AskResponse(
            response="I can only help with family-friendly dog questions! 🐾",
            source="blocked",
        )

    # ── Step 2: Vector store retrieval ─────────────────────────────────────────
    query_text = f"{breed} {question}"
    query_emb  = get_query_embedding(query_text)

    vs_results = _chroma_col.query(
        query_embeddings=[query_emb],
        n_results=N_VECTOR_RESULTS,
        include=["documents"],
    )
    docs = vs_results.get("documents", [[]])[0]   # list of strings
    log.info(f"[VS] Retrieved {len(docs)} documents")

    # ── Step 3: Can the vector store answer? ───────────────────────────────────
    if docs:
        answer = agent_eval_and_answer(breed, question, docs)
        if answer:
            return AskResponse(response=answer, source="vector")

    # ── Step 4: Tavily fallback ────────────────────────────────────────────────
    search_query = f"{breed} dog breed {question}"
    log.info(f"[Tavily] Searching: {search_query!r}")
    tavily_resp  = _tavily.search(query=search_query, max_results=N_TAVILY_RESULTS)
    results      = tavily_resp.get("results", [])

    if not results:
        return AskResponse(
            response="Sorry, I couldn't find enough information on that topic right now. Try rephrasing your question!",
            source="tavily",
        )

    final_answer = agent_clean_tavily(breed, question, results)
    return AskResponse(response=final_answer, source="tavily")


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_docs": _chroma_col.count() if _chroma_col else 0,
        "embed_backend": EMBED_BACKEND,
        "gemini_model": GEMINI_MODEL,
    }
