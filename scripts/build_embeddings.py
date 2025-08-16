# scripts/build_embeddings.py
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

INDEX_PATH = Path("index/search_index.json")
EMB_PATH = Path("index/embeddings.npz")

def main():
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    docs = data.get("docs", [])
    if not docs:
        print("No docs found. Run scripts/build_index.py first.")
        return

    # Small, fast model good for CPU
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [d["text"] for d in docs]
    print(f"Encoding {len(texts)} segments…")
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    # store embeddings + minimal doc info
    np.savez(EMB_PATH, embeddings=embs)
    print(f"Saved embeddings → {EMB_PATH}")

if __name__ == "__main__":
    main()
