#!/usr/bin/env python3
"""
tavily_search.py

Install:
  pip install tavily-python python-dotenv

.env file:
  TAVILY_API_KEY=tvly-your-key-here
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tavily import TavilyClient


# =========================
# CONFIG
# =========================
SEARCH_DEPTH = "basic"          # basic | advanced | fast
TOPIC = "general"               # general | news | finance
MAX_RESULTS = 5
INCLUDE_ANSWER = True
INCLUDE_RAW_CONTENT = False
INCLUDE_IMAGES = False
INCLUDE_FAVICON = True
EXACT_MATCH = False
INCLUDE_USAGE = True

INCLUDE_DOMAINS: List[str] = []
EXCLUDE_DOMAINS: List[str] = []
COUNTRY: Optional[str] = None
TIME_RANGE: Optional[str] = None


def build_search_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "search_depth": SEARCH_DEPTH,
        "topic": TOPIC,
        "max_results": MAX_RESULTS,
        "include_answer": INCLUDE_ANSWER,
        "include_raw_content": INCLUDE_RAW_CONTENT,
        "include_images": INCLUDE_IMAGES,
        "include_favicon": INCLUDE_FAVICON,
        "exact_match": EXACT_MATCH,
        "include_usage": INCLUDE_USAGE,
    }

    if INCLUDE_DOMAINS:
        kwargs["include_domains"] = INCLUDE_DOMAINS

    if EXCLUDE_DOMAINS:
        kwargs["exclude_domains"] = EXCLUDE_DOMAINS

    if COUNTRY:
        kwargs["country"] = COUNTRY

    if TIME_RANGE:
        kwargs["time_range"] = TIME_RANGE

    return kwargs


def main() -> int:
    load_dotenv()

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("Error: TAVILY_API_KEY is missing from your .env file.")
        return 1

    client = TavilyClient(api_key=api_key)

    print("Tavily Search CLI (type 'exit' to quit)\n")

    while True:
        query = input("Enter your search query: ").strip()

        if not query:
            print("Please enter a valid query.\n")
            continue

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            response = client.search(query, **build_search_kwargs())
            print("\nResults:\n")
            print(json.dumps(response, indent=2, ensure_ascii=False))
            print("\n" + "=" * 60 + "\n")

        except Exception as e:
            print(f"Error during search: {e}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())