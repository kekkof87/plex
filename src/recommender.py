import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pathlib import Path
from .data_loader import load_csv

MODEL_NAME = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")
EMBED_DIR = Path("data") / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

class ContentRecommender:
    def __init__(self, kind="movies"):
        self.kind = kind
        self.df = load_csv(kind)
        self.model = SentenceTransformer(MODEL_NAME)
        self.embeddings = None
        self._ensure_embeddings()

    def _ensure_embeddings(self):
        emb_path = EMBED_DIR / f"{self.kind}_emb.npy"
        idx_path = EMBED_DIR / f"{self.kind}_idx.csv"
        if emb_path.exists() and idx_path.exists():
            # load
            self.embeddings = np.load(emb_path)
            idx = pd.read_csv(idx_path)
            # align index
            self.df = self.df.loc[idx['orig_index']].reset_index(drop=True)
        else:
            self._compute_and_store_embeddings()

    def _compute_and_store_embeddings(self):
        texts = (self.df['title'].fillna("") + " . " + self.df.get('description', "").fillna("") + " . " + self.df.get('genres', "").fillna(""))
        self.embeddings = self.model.encode(texts.tolist(), show_progress_bar=True, convert_to_numpy=True)
        emb_path = EMBED_DIR / f"{self.kind}_emb.npy"
        np.save(str(emb_path), self.embeddings)
        idx_df = pd.DataFrame({"orig_index": self.df.index})
        idx_df.to_csv(EMBED_DIR / f"{self.kind}_idx.csv", index=False)

    def recommend_by_title(self, title, top_k=10):
        # find title in df (approximate match)
        df = self.df
        match = df[df['title'].str.lower().str.contains(title.lower())]
        if match.empty:
            # fallback: compute embedding of query
            q_emb = self.model.encode([title])[0]
        else:
            q_idx = match.index[0]
            q_emb = self.embeddings[q_idx]
        sims = cosine_similarity([q_emb], self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][1:top_k+1]  # skip self
        return df.iloc[top_idx].copy().assign(score=sims[top_idx])

    def recommend_for_item(self, item_index, top_k=10):
        q_emb = self.embeddings[item_index]
        sims = cosine_similarity([q_emb], self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][1:top_k+1]
        return self.df.iloc[top_idx].copy().assign(score=sims[top_idx])

    def get_popular(self, top_k=20):
        # if df has rating field, use it; else fallback to count or tmdb popularity if merged
        if 'rating' in self.df.columns:
            return self.df.sort_values('rating', ascending=False).head(top_k)
        elif 'popularity' in self.df.columns:
            return self.df.sort_values('popularity', ascending=False).head(top_k)
        else:
            return self.df.head(top_k)
