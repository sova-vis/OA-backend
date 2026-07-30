"""
Step 3: Embed every chunk in data/chunks.jsonl and store it in a local Chroma
vector database (data/vector_store/) alongside its metadata, so chunks can
later be retrieved by semantic similarity and filtered by subject/year/etc.

Uses a free, local embedding model (sentence-transformers/all-MiniLM-L6-v2)
- no API key or network calls needed after the model is first downloaded.

Run:
    venv\\Scripts\\python.exe scripts\\build_vector_store.py
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
JSONL_PATH = PROJECT_DIR / "data" / "chunks.jsonl"
VECTOR_STORE_DIR = PROJECT_DIR / "data" / "vector_store"

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "past_papers"
BATCH_SIZE = 256


def sanitize_metadata(meta: dict) -> dict:
    """Chroma only accepts str/int/float/bool metadata values (no None)."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue
        clean[k] = v
    return clean


def load_chunks():
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    print(f"Loading embedding model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, texts, metadatas = [], [], []
    total = 0

    def flush():
        nonlocal ids, texts, metadatas, total
        if not ids:
            return
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        total += len(ids)
        print(f"  embedded {total} chunks so far...")
        ids, texts, metadatas = [], [], []

    for i, chunk in enumerate(load_chunks()):
        meta = sanitize_metadata(chunk["metadata"])
        meta["source_file"] = chunk["source_file"]
        ids.append(f"chunk_{i}")
        texts.append(chunk["chunk_text"])
        metadatas.append(meta)

        if len(ids) >= BATCH_SIZE:
            flush()
    flush()

    print(f"Done. Total chunks embedded and stored: {total}")
    print(f"Vector store saved at: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
