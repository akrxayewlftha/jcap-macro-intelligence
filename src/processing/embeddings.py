"""
Embeddings — génère des vecteurs sémantiques pour tweets et documents.

J'utilise all-MiniLM-L6-v2 : c'est le meilleur rapport qualité/vitesse
pour les textes anglais courts. Pour du multilingue on passerait sur
paraphrase-multilingual-MiniLM-L12-v2, mais ici tout est en anglais.

Les embeddings sont mis en cache sur disque pour éviter de relancer
le modèle à chaque fois — c'est ce qui prend le plus de temps.
"""
from __future__ import annotations


import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Union

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Le modèle qu'on utilise. Léger (80MB), rapide, très bon en anglais.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingEngine:
    """
    Wrapper autour de SentenceTransformer avec mise en cache des embeddings.
    Comme ça on ne recalcule pas tout à chaque run — c'est crucial en dev.
    """

    def __init__(self, model_name: str = MODEL_NAME, cache_dir: Path = None):
        print(f"  Chargement du modèle d'embeddings : {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, texts: list[str]) -> str:
        """Hash stable des textes pour identifier le cache."""
        content = json.dumps(texts, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _load_cache(self, key: str) -> Union[np.ndarray, None]:
        """Charge les embeddings depuis le cache s'ils existent."""
        if not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"emb_{key}.npy"
        if cache_file.exists():
            print(f"  Cache hit : {cache_file.name}")
            return np.load(cache_file)
        return None

    def _save_cache(self, key: str, embeddings: np.ndarray):
        """Sauvegarde les embeddings dans le cache."""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / f"emb_{key}.npy"
        np.save(cache_file, embeddings)

    def encode(self, texts: list[str], batch_size: int = 64,
               show_progress: bool = True) -> np.ndarray:
        """
        Encode une liste de textes en vecteurs.
        Gère la mise en cache automatiquement.
        """
        if not texts:
            return np.array([])

        # Vérifie le cache d'abord
        key = self._cache_key(texts)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        print(f"  Calcul des embeddings pour {len(texts)} textes...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalisation L2 pour cosine similarity
        )

        self._save_cache(key, embeddings)
        return embeddings

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Similarité cosinus entre deux vecteurs (ou deux matrices).
        Comme on normalise à l'encode, c'est juste un produit scalaire.
        """
        if emb1.ndim == 1 and emb2.ndim == 1:
            return float(np.dot(emb1, emb2))
        # Matrices : similarité par paires
        return np.dot(emb1, emb2.T)

    def find_similar(self, query_emb: np.ndarray, corpus_embs: np.ndarray,
                     top_k: int = 5) -> list[tuple[int, float]]:
        """
        Trouve les top_k textes les plus similaires à une requête.
        Retourne une liste de (index, score).
        """
        scores = self.similarity(query_emb, corpus_embs)
        if scores.ndim == 0:
            return [(0, float(scores))]
        # Trie par score décroissant
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]


# ─── Fonctions utilitaires ────────────────────────────────────────────────────

def embed_corpus(documents: list[dict], engine: EmbeddingEngine,
                 output_dir: Path = None) -> np.ndarray:
    """
    Génère les embeddings pour les documents du corpus.
    Utilise la concaténation titre + preview pour chaque doc.
    """
    # On encode le titre + les premiers mots pour avoir un vecteur représentatif
    texts = [f"{doc['source']}: {doc['title']} — {doc['preview'][:300]}"
             for doc in documents]

    embeddings = engine.encode(texts, show_progress=True)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "corpus_embeddings.npy", embeddings)
        print(f"  Embeddings corpus sauvegardés → {output_dir / 'corpus_embeddings.npy'}")

    return embeddings


def embed_tweets(df, engine: EmbeddingEngine,
                 output_dir: Path = None) -> np.ndarray:
    """
    Génère les embeddings pour les tweets.
    On encode le texte nettoyé directement.
    """
    texts = df["text_clean"].tolist()
    embeddings = engine.encode(texts, batch_size=128, show_progress=True)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "tweet_embeddings.npy", embeddings)
        print(f"  Embeddings tweets sauvegardés → {output_dir / 'tweet_embeddings.npy'}")

    return embeddings


# ─── Point d'entrée standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    import json, pandas as pd
    BASE = Path(__file__).resolve().parents[2]

    engine = EmbeddingEngine(cache_dir=BASE / "data" / "processed" / "cache")

    # Test rapide
    test_texts = [
        "The Federal Reserve raised interest rates by 25 basis points",
        "ECB maintains accommodative monetary policy stance",
        "Oil prices surge amid Middle East tensions",
        "China GDP growth slows to 4.5% in Q1"
    ]
    embs = engine.encode(test_texts)
    print(f"\nShape : {embs.shape}")
    print(f"Similarity Fed↔ECB : {engine.similarity(embs[0], embs[1]):.3f}")
    print(f"Similarity Fed↔Oil : {engine.similarity(embs[0], embs[2]):.3f}")
