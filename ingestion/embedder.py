# =============================================================================
# ingestion/embedder.py — sentence-transformers embedding (reusable)
#
# WHY THIS FILE EXISTS:
# Loads the embedding model ONCE and exposes methods to embed text. Used in two
# places (Option A):
#   - ingestion pipeline → embed chunk texts before storing
#   - agent/retriever    → embed the user's question before searching
# Sharing one Embedder means the model loads once and there's no duplicated code.
# Graduated from the notebook embedding cell.
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# You need:
#   - `SentenceTransformer`   → from sentence_transformers, the model
#   - `Chunk`                → from store.base (to type embed_chunks)
#
# Write your imports here:
from sentence_transformers import SentenceTransformer
from store.base import Chunk 


# -----------------------------------------------------------------------------
# CLASS: Embedder
#
# WHY: Wraps the model so callers don't touch sentence-transformers directly.
# Load the model in __init__ (expensive, do once); reuse it for every embed call.
# =============================================================================
#
#   __init__(self, model_name: str = _MODEL_NAME):
#       1. self._model = SentenceTransformer(model_name)
#
#   embed(self, texts: list[str]) -> list[list[float]]:
#       WHY: Batch-embed many strings (e.g. all chunks on a page).
#       1. embeddings = self._model.encode(texts)        ← numpy array (n x 384)
#       2. return [e.tolist() for e in embeddings]        ← native floats for ChromaDB
#          (REMEMBER: .tolist() — ChromaDB rejects np.float32, like you hit before)
#
#   embed_one(self, text: str) -> list[float]:
#       WHY: Embed a single string (the user's question in the retriever).
#       1. return self._model.encode(text).tolist()
#
#   embed_chunks(self, chunks: list[Chunk]) -> None:
#       WHY: Convenience for the pipeline — fills each chunk's .embedding in place.
#       1. texts = [c.text for c in chunks]
#       2. vectors = self.embed(texts)
#       3. for chunk, vector in zip(chunks, vectors): chunk.embedding = vector
#       (mutates the chunks; returns nothing)
#
# Write the Embedder class here:

class Embedder:
# -----------------------------------------------------------------------------
# CONSTANT
# -----------------------------------------------------------------------------
#   _MODEL_NAME = "all-MiniLM-L6-v2"   ← 384-dim, local, free

    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = _MODEL_NAME):
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts)
        return [e.tolist() for e in embeddings]
    
    def embed_chunks(self, chunks: list[Chunk]) -> None:

        texts = [c.text for c in chunks]
        embedding_list = self.embed(texts)

        for chunk, embedding in zip(chunks, embedding_list):
            chunk.embedding = embedding

    def embed_one(self, user_query: str) -> list[float]:
        return self._model.encode(user_query).tolist()
