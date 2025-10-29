import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from .data_loader import load_csv

MODEL_NAME = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")
EMBED_DIR = Path("data") / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

class ContentRecommender:
    def __init__(self, kind="movies"):
        """
        kind: 'movies', 'series' o 'anime'
        """
        self.kind = kind
        self.df = load_csv(kind)

        # Crea colonne vuote se mancano
        for col in ["description", "genres", "rating", "popularity", "year"]:
            if col not in self.df.columns:
                self.df[col] = ""

        # Forza l'uso della CPU (evita meta tensor error)
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")

        self.embeddings = None
        self._ensure_embeddings()

    def _ensure_embeddings(self):
        emb_path = EMBED_DIR / f"{self.kind}_emb.npy"
        idx_path = EMBED_DIR / f"{self.kind}_idx.csv"
        if emb_path.exists() and idx_path.exists():
            try:
                self.embeddings = np.load(emb_path)
                idx = pd.read_csv(idx_path)
                self.df = self.df.loc[idx['orig_index']].reset_index(drop=True)
                return
            except Exception:
                pass
        self._compute_and_store_embeddings()

    def _compute_and_store_embeddings(self):
        texts = (
            self.df["title"].fillna("") + " . " +
            self.df["description"].fillna("") + " . " +
            self.df["genres"].fillna("")
        )
        self.embeddings = self.model.encode(
            texts.tolist(),
            show_progress_bar=True,
            convert_to_numpy=True
        )
        emb_path = EMBED_DIR / f"{self.kind}_emb.npy"
        np.save(str(emb_path), self.embeddings)
        pd.DataFrame({"orig_index": self.df.index}).to_csv(EMBED_DIR / f"{self.kind}_idx.csv", index=False)

    def recommend_by_title(self, title, top_k=10):
        df = self.df
        match = df[df["title"].str.lower().str.contains(title.lower(), na=False)]
        if match.empty:
            q_emb = self.model.encode([title])[0]
        else:
            q_emb = self.embeddings[match.index[0]]

        sims = cosine_similarity([q_emb], self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][1:top_k + 1]
        return df.iloc[top_idx].copy().assign(score=sims[top_idx])

    def get_popular(self, top_k=20):
        df = self.df.copy()
        if "rating" in df.columns and df["rating"].notna().any():
            return df.sort_values("rating", ascending=False).head(top_k)
        elif "popularity" in df.columns and df["popularity"].notna().any():
            return df.sort_values("popularity", ascending=False).head(top_k)
        else:
            return df.head(top_k)

    def get_all_time(self, top_k=20):
        """Top per rating se disponibile, altrimenti primi titoli"""
        df = self.df.copy()
        if "rating" in df.columns and df["rating"].notna().any():
            return df.sort_values("rating", ascending=False).head(top_k)
        return df.head(top_k)
