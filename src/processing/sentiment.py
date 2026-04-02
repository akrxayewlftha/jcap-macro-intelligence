"""
Sentiment Analysis — analyse le ton des tweets et documents.

J'utilise FinBERT (prosustech/finbert) plutôt qu'un sentiment généraliste
parce qu'il a été fine-tuné sur des textes financiers. La différence est
significative : "hawkish" est positif en général mais négatif pour les
marchés obligataires — FinBERT comprend ce contexte.

Pour les documents on découpe par section et on moyenne les scores.
"""
from __future__ import annotations


import json
import numpy as np
from pathlib import Path
from typing import Optional, Union

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# FinBERT : le meilleur modèle pour le sentiment financier
MODEL_NAME = "ProsusAI/finbert"

# Labels dans l'ordre du modèle
FINBERT_LABELS = ["positive", "negative", "neutral"]


class SentimentAnalyzer:
    """
    Analyseur de sentiment basé sur FinBERT.
    Optimisé pour les textes financiers en anglais.
    """

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32):
        print(f"  Chargement de FinBERT : {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        # GPU si dispo, sinon CPU — ça reste acceptable en batch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        print(f"  Device : {self.device}")

    def _predict_batch(self, texts: list[str]) -> list[dict]:
        """Prédit le sentiment d'un batch de textes."""
        # Troncature à 512 tokens (limite BERT)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        results = []
        for prob in probs:
            label_idx = int(np.argmax(prob))
            results.append({
                "sentiment": FINBERT_LABELS[label_idx],
                "positive": float(prob[0]),
                "negative": float(prob[1]),
                "neutral": float(prob[2]),
                "confidence": float(prob[label_idx]),
                # Score composite : positive - negative (de -1 à +1)
                "score": float(prob[0] - prob[1])
            })
        return results

    def analyze(self, texts: list[str], show_progress: bool = True) -> list[dict]:
        """
        Analyse le sentiment d'une liste de textes.
        Retourne une liste de dicts avec label + scores.
        """
        if not texts:
            return []

        all_results = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="  Sentiment")

        for i in iterator:
            batch = texts[i * self.batch_size: (i + 1) * self.batch_size]
            # Remplace les textes vides par un placeholder
            batch = [t if t.strip() else "neutral" for t in batch]
            results = self._predict_batch(batch)
            all_results.extend(results)

        return all_results

    def analyze_single(self, text: str) -> dict:
        """Raccourci pour analyser un seul texte."""
        return self.analyze([text], show_progress=False)[0]


# ─── Fonctions d'enrichissement ──────────────────────────────────────────────

def enrich_tweets_sentiment(df: pd.DataFrame, analyzer: SentimentAnalyzer,
                              output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Ajoute les colonnes sentiment au DataFrame des tweets."""
    print(f"  Analyse sentiment sur {len(df)} tweets...")
    results = analyzer.analyze(df["text_clean"].tolist())

    df = df.copy()
    df["sentiment"] = [r["sentiment"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]
    df["sentiment_positive"] = [r["positive"] for r in results]
    df["sentiment_negative"] = [r["negative"] for r in results]
    df["sentiment_confidence"] = [r["confidence"] for r in results]

    # Stats rapides
    counts = df["sentiment"].value_counts()
    print(f"  → Positif: {counts.get('positive', 0)}, "
          f"Négatif: {counts.get('negative', 0)}, "
          f"Neutre: {counts.get('neutral', 0)}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Sauvegarde juste les colonnes sentiment
        df[["id", "sentiment", "sentiment_score", "sentiment_positive",
            "sentiment_negative", "sentiment_confidence"]].to_json(
            output_dir / "tweet_sentiments.json", orient="records", indent=2
        )

    return df


def compute_corpus_sentiment(documents: list[dict],
                               analyzer: SentimentAnalyzer,
                               output_dir: Optional[Path] = None) -> list[dict]:
    """
    Calcule le sentiment pour chaque document du corpus.
    On analyse par page et on agrège.
    """
    print(f"  Analyse sentiment sur {len(documents)} documents...")
    enriched = []

    for doc in tqdm(documents, desc="  Corpus sentiment"):
        # Analyse les sections
        section_texts = [s["text"][:512] for s in doc.get("sections", [])]
        if not section_texts:
            section_texts = [doc.get("preview", "")]

        section_results = analyzer.analyze(section_texts, show_progress=False)

        # Score global : moyenne des sections
        avg_score = np.mean([r["score"] for r in section_results])
        avg_positive = np.mean([r["positive"] for r in section_results])
        avg_negative = np.mean([r["negative"] for r in section_results])

        # Sentiment dominant
        if avg_score > 0.1:
            dominant = "positive"
        elif avg_score < -0.1:
            dominant = "negative"
        else:
            dominant = "neutral"

        enriched_doc = {**doc}
        enriched_doc["sentiment"] = dominant
        enriched_doc["sentiment_score"] = float(avg_score)
        enriched_doc["sentiment_positive"] = float(avg_positive)
        enriched_doc["sentiment_negative"] = float(avg_negative)
        enriched_doc["section_sentiments"] = section_results
        enriched.append(enriched_doc)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Version allégée sans les sections complètes
        light = [{
            "id": d["id"], "source": d["source"],
            "sentiment": d["sentiment"], "sentiment_score": d["sentiment_score"]
        } for d in enriched]
        with open(output_dir / "corpus_sentiments.json", "w") as f:
            json.dump(light, f, indent=2)

    return enriched


def get_sentiment_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège le sentiment par heure pour voir l'évolution temporelle.
    Utile pour le dashboard.
    """
    timeline = df.groupby(["date_str", "hour"]).agg(
        avg_score=("sentiment_score", "mean"),
        positive_count=("sentiment", lambda x: (x == "positive").sum()),
        negative_count=("sentiment", lambda x: (x == "negative").sum()),
        neutral_count=("sentiment", lambda x: (x == "neutral").sum()),
        tweet_count=("id", "count")
    ).reset_index()

    return timeline
