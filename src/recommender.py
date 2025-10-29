import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pathlib import Path
from .data_loader import load_csv

# Nome del modello sentence-transformers
MODEL_NAME = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")

# Directory per salvare gli embeddings
EMBED_DIR = Path("data") / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

class ContentRecommender:
    def __init__(self, kind="movies"):
        """
        kind: 'movies', 'series' o 'anime'
        """
        self.kind = kind
        self.df = load_csv(kind)

        # Forza l'uso della CPU per evitare errori Torch "meta tensor"
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")

        self.embeddings = None
        self._ensure_embeddings()

    def _ensure_embeddings(self):
        """Verifica se esistono embeddings salvati, altrimenti li rigenera"""
        emb_path = EMBED_DIR / f"{self.kind}_emb.npy"
        idx_path = EMBED_DIR / f"{self.kind}_idx.csv"

        if emb_path.exists() and idx_path.exists():
            try:
                self.embeddings = np.load(emb_path)
                idx = pd.read_csv(idx_path)
                self.df = self.df.loc[idx['orig_index']].reset_index(drop=True)
                return
            except Exception:
                pass  # in caso di errore rigenera tutto

        self._compute_and_store_embeddings()

    def _compute_and_store_embeddings(self):
        """Calcola e salva embeddings per tutti i titoli"""
        # Garantisce l'esistenza delle colonne necessarie
        if 'description' not in self.df.columns:
            self.df['description'] = ""
        if 'genres' not in self.df.columns:
            self.df['genres'] = ""

        # Costruisce il testo combinato
        texts = (
            self.df['title'].fillna("") + " . " +
            self.df['description'].fillna("") + " . " +
            self.df['genres'].fillna("")
        )

        # Calcola embeddings
        self.embeddings = self.model.encode(
            texts.tolist(),
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Salva embeddings e indice
        emb_path = EMBED_DIR / f"{self.kind}_emb.npy"
        np.save(str(emb_path), self.embeddings)
        idx_df = pd.DataFrame({"orig_index": self.df.index})
        idx_df.to_csv(EMBED_DIR / f"{self.kind}_idx.csv", index=False)

    def recommend_by_title(self, title, top_k=10):
        """Suggerisci elementi simili a un titolo (ricerca testuale o embedding)"""
        df = self.df
        match = df[df['title'].str.lower().str.contains(title.lower(), na=False)]

        if match.empty:
            # Se non trova corrispondenze dirette, crea embedding del testo
            q_emb = self.model.encode([title])[0]
        else:
            q_idx = match.index[0]
            q_emb = self.embeddings[q_idx]

        sims = cosine_similarity([q_emb], self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][1:top_k + 1]  # ignora sé stesso
        return df.iloc[top_idx].copy().assign(score=sims[top_idx])

    def recommend_for_item(self, item_index, top_k=10):
        """Suggerisci simili a un item dato il suo indice"""
        q_emb = self.embeddings[item_index]
        sims = cosine_similarity([q_emb], self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][1:top_k + 1]
        return self.df.iloc[top_idx].copy().assign(score=sims[top_idx])

    def get_popular(self, top_k=20):
        """Restituisce i titoli più popolari o votati"""
        df = self.df.copy()

        if isinstance(df, str):
            # in caso di errore di caricamento (df letto come stringa)
            return pd.DataFrame(columns=["title"])

        if 'rating' in df.columns:
            return df.sort_values('rating', ascending=False).head(top_k)
        elif 'popularity' in df.columns:
            return df.sort_values('popularity', ascending=False).head(top_k)
        else:
            # fallback → mostra i primi N titoli
            return df.head(top_k)
