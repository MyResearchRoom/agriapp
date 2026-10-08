import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = Path(__file__).parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"


class VectorRetriever:
    def __init__(self):
        print("[Retrieval] Loading embedding model, FAISS index, and metadata...")
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Vector store files missing. Ensure '{FAISS_INDEX_PATH}' and '{METADATA_PATH}' exist."
            )
        
        # Load local embedding model
        self.model = SentenceTransformer(MODEL_NAME)
        
        # Load FAISS index
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        
        # Load metadata
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"[Retrieval] Initialized successfully. Index vectors: {self.index.ntotal}, Metadata chunks: {len(self.metadata)}")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not query or not query.strip():
            return []

        # 1. Embed query
        raw_emb = self.model.encode([query.strip()], convert_to_numpy=True)
        
        # 2. L2 Normalize (must match index normalization)
        norm = np.linalg.norm(raw_emb, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        query_vector = (raw_emb / norm).astype(np.float32)

        # 3. FAISS inner product search
        k = min(top_k, len(self.metadata))
        distances, indices = self.index.search(query_vector, k)

        # 4. Map indices back to metadata entries
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            chunk_data = dict(self.metadata[idx])
            chunk_data["similarity_score"] = float(dist)
            results.append(chunk_data)

        return results


# Module-level singleton instance loaded once at import
retriever = VectorRetriever()


def search(query: str, top_k: int = 3) -> list[dict]:
    """Exposed convenience search function."""
    return retriever.search(query=query, top_k=top_k)
