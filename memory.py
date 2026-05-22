import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class SQLiteMemory:
    def __init__(self, db_path: str = "memory.sqlite"):
        self.db = sqlite3.connect(db_path)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS text_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT
            )
        """)
        self.db.commit()

        # Text encoder
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed(self, text: str) -> np.ndarray:
        vec = self.text_encoder.encode([text])[0]
        return vec.astype("float32")

    def add(self, text: str, metadata: dict | None = None):
        vec = self._embed(text)
        meta_json = json.dumps(metadata or {})
        self.db.execute(
            "INSERT INTO text_memories (text, embedding, metadata) VALUES (?, ?, ?)",
            (text, vec.tobytes(), meta_json)
        )
        self.db.commit()

    def search(self, query: str, k: int = 5):
        q_vec = self._embed(query)

        rows = self.db.execute(
            "SELECT text, embedding, metadata FROM text_memories"
        ).fetchall()

        scored = []
        for text, emb_blob, meta_json in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            denom = np.linalg.norm(q_vec) * np.linalg.norm(emb)
            if denom == 0:
                continue
            score = float(np.dot(q_vec, emb) / denom)
            scored.append((score, text, json.loads(meta_json)))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [(t, m) for _, t, m in scored[:k]]
