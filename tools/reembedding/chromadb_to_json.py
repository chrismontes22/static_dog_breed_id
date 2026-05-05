"""
Run this ONCE locally (same folder that contains your 'vectordb' directory).
It exports your ChromaDB to a static dogdb.json that the browser can load.
"""
import chromadb
import json

CHROMA_FOLDER = "vectordb"

client = chromadb.PersistentClient(path=CHROMA_FOLDER)

# List all collections so you can verify the name
all_collections = client.list_collections()
print("Collections found:", [c.name for c in all_collections])

# Use the first collection, or change "dogdb" to the right name
collection = client.get_collection("dogdb")
print(f"Total entries: {collection.count()}")

results = collection.get(include=["embeddings", "documents", "metadatas"])

data = {
    "ids":       results["ids"],
    "documents": results["documents"],
    "metadatas": results["metadatas"],
    "embeddings": [
        emb.tolist() if hasattr(emb, "tolist") else list(emb)
        for emb in results["embeddings"]
    ]
}

with open("dogdb.json", "w") as f:
    json.dump(data, f)

print(f"✅  dogdb.json written — {len(data['ids'])} vectors exported")
