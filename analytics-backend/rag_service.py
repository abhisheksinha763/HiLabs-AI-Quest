import os
import json
import pandas as pd
import faiss
import numpy as np
from typing import List, Dict
import ollama
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, csv_path: str, embed_model: str = "nomic-embed-text", chunk_rows: int = 10):
        self.csv_path = csv_path
        self.embed_model = embed_model
        self.chunk_rows = chunk_rows
        self.index_path = "data_storage/faiss.index"
        self.store_path = "data_storage/store.jsonl"
        
        # Ensure data_storage directory exists
        os.makedirs("data_storage", exist_ok=True)
        
        self.index = None
        self.chunks = []
        self._initialize()
    
    def _initialize(self):
        """Initialize or load the FAISS index and chunks."""
        if os.path.exists(self.index_path) and os.path.exists(self.store_path):
            logger.info("Loading existing FAISS index and chunks...")
            self._load_index()
        else:
            logger.info("Building new FAISS index...")
            self._build_index()
    
    def _load_csv_chunks(self) -> List[Dict]:
        """Convert CSV rows into text chunks for retrieval."""
        if not os.path.exists(self.csv_path):
            logger.error(f"CSV file not found: {self.csv_path}")
            return []
        
        df = pd.read_csv(self.csv_path)
        df = df.fillna("")  # avoid NaNs
        
        records = df.to_dict(orient="records")
        chunks = []
        
        for i in range(0, len(records), self.chunk_rows):
            group = records[i : i + self.chunk_rows]
            text_lines = []
            for r in group:
                # Turn row into a readable line
                line = " | ".join([f"{k}: {str(v)}" for k, v in r.items()])
                text_lines.append(line)
            chunk_text = "\n".join(text_lines)
            chunks.append({
                "chunk_id": len(chunks),
                "text": chunk_text
            })
        
        logger.info(f"Created {len(chunks)} chunks from CSV")
        return chunks
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Use Ollama embedding model to embed a list of texts."""
        vectors = []
        for i, text in enumerate(texts):
            try:
                resp = ollama.embeddings(model=self.embed_model, prompt=text)
                vectors.append(resp["embedding"])
                if (i + 1) % 10 == 0:
                    logger.info(f"Embedded {i + 1}/{len(texts)} chunks")
            except Exception as e:
                logger.error(f"Error embedding text {i}: {e}")
                # Use zero vector as fallback
                vectors.append([0.0] * 768)  # Default embedding size
        
        return np.array(vectors, dtype="float32")
    
    def _build_index(self):
        """Build FAISS index and save chunks."""
        self.chunks = self._load_csv_chunks()
        if not self.chunks:
            logger.error("No chunks created from CSV")
            return
        
        texts = [c["text"] for c in self.chunks]
        embs = self._embed_texts(texts)
        
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product (cosine if normalized)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embs)
        self.index.add(embs)
        
        # Save index and chunks
        faiss.write_index(self.index, self.index_path)
        with open(self.store_path, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        
        logger.info(f"Built and saved FAISS index with {len(self.chunks)} chunks")
    
    def _load_index(self):
        """Load existing FAISS index and chunks."""
        self.index = faiss.read_index(self.index_path)
        
        self.chunks = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        logger.info(f"Loaded FAISS index with {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query."""
        if not self.index or not self.chunks:
            logger.error("Index not initialized")
            return []
        
        try:
            q_emb = self._embed_texts([query])
            faiss.normalize_L2(q_emb)
            D, I = self.index.search(q_emb, k)
            I = I[0].tolist()
            
            retrieved = [self.chunks[i] for i in I if i != -1 and i < len(self.chunks)]
            logger.info(f"Retrieved {len(retrieved)} chunks for query")
            return retrieved
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def reindex(self):
        """Rebuild the index from the current CSV."""
        logger.info("Reindexing CSV data...")
        self._build_index()
        return len(self.chunks)
