"""
Topic Modeling — extraction de thèmes via BERTopic.

BERTopic est bien meilleur que LDA sur des textes courts (tweets) :
il utilise les embeddings pour le clustering, puis KeyBERT pour
labelliser les topics. C'est plus robuste et interprétable.

Pour le corpus documentaire on segmente par paragraphe avant de
faire tourner BERTopic — sinon les documents longs dominent.
"""
from __future__ import annotations


import json
import numpy as np
from pathlib import Path
from typing import Optional

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


# ─── Config BERTopic ──────────────────────────────────────────────────────────

def build_topic_model(min_topic_size: int = 3,
                      nr_topics: int = "auto",
                      language: str = "english") -> BERTopic:
    """
    Construit le modèle BERTopic avec la config optimisée pour nos données.

    min_topic_size=3 parce qu'on n'a pas des milliers de documents —
    on peut se permettre des topics petits mais cohérents.
    """
    # Vectorizer avec ngrammes pour capturer des expressions comme "interest rate"
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_features=10000
    )

    # Représentation KeyBERT pour des labels plus naturels
    representation_model = KeyBERTInspired()

    model = BERTopic(
        language=language,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        representation_model=representation_model,
        vectorizer_model=vectorizer,
        verbose=False,
        calculate_probabilities=True
    )

    return model


# ─── Extraction thèmes tweets ─────────────────────────────────────────────────

def extract_tweet_topics(df: pd.DataFrame, embeddings: np.ndarray,
                         output_dir: Optional[Path] = None) -> dict:
    """
    Applique BERTopic sur le corpus de tweets.
    Retourne un dict avec le modèle, les topics et les infos par tweet.
    """
    texts = df["text_clean"].tolist()

    print(f"  BERTopic sur {len(texts)} tweets...")
    model = build_topic_model(min_topic_size=4)

    # On passe nos embeddings pré-calculés pour éviter de recalculer
    topics, probs = model.fit_transform(texts, embeddings)

    # Récupère les infos sur les topics
    topic_info = model.get_topic_info()

    # Labellise les topics de façon lisible
    topic_labels = {}
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            topic_labels[topic_id] = "Hors-sujet"
            continue
        words = model.get_topic(topic_id)
        if words:
            # Les 3 premiers mots comme label
            label = " / ".join([w[0] for w in words[:3]])
            topic_labels[topic_id] = label

    # Ajoute les topics au DataFrame
    df = df.copy()
    df["topic_id"] = topics
    df["topic_prob"] = [max(p) if hasattr(p, '__iter__') else p for p in probs]
    df["topic_label"] = df["topic_id"].map(topic_labels)

    result = {
        "model": model,
        "df": df,
        "topic_info": topic_info.to_dict(orient="records"),
        "topic_labels": topic_labels,
        "n_topics": len(topic_info[topic_info["Topic"] >= 0])
    }

    print(f"  → {result['n_topics']} topics extraits")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Sauvegarde les métadonnées (pas le modèle entier — trop lourd)
        with open(output_dir / "tweet_topics.json", "w") as f:
            json.dump({
                "topic_info": result["topic_info"],
                "topic_labels": {str(k): v for k, v in topic_labels.items()}
            }, f, indent=2)

    return result


# ─── Extraction thèmes corpus ─────────────────────────────────────────────────

def _split_into_paragraphs(documents: list[dict], min_words: int = 30) -> list[dict]:
    """
    Découpe les documents en paragraphes pour le topic modeling.
    Filtre les paragraphes trop courts (titres, headers, etc.)
    """
    paragraphs = []
    for doc in documents:
        for section in doc.get("sections", []):
            # Découpe par double saut de ligne
            paras = section["text"].split("\n\n")
            for para in paras:
                para = para.strip()
                if len(para.split()) >= min_words:
                    paragraphs.append({
                        "doc_id": doc["id"],
                        "doc_source": doc["source"],
                        "doc_date": doc.get("date"),
                        "page": section["page"],
                        "text": para
                    })
    return paragraphs


def extract_corpus_topics(documents: list[dict], para_embeddings: np.ndarray,
                           output_dir: Optional[Path] = None) -> dict:
    """
    Applique BERTopic sur les paragraphes du corpus documentaire.
    """
    paragraphs = _split_into_paragraphs(documents)
    texts = [p["text"] for p in paragraphs]

    print(f"  BERTopic sur {len(texts)} paragraphes du corpus...")

    if len(texts) < 10:
        print("  Pas assez de paragraphes, skipping BERTopic")
        return {"paragraphs": paragraphs, "n_topics": 0, "topic_labels": {}}

    model = build_topic_model(min_topic_size=3)
    topics, probs = model.fit_transform(texts, para_embeddings)

    topic_info = model.get_topic_info()
    topic_labels = {}
    for topic_id in topic_info["Topic"].values:
        if topic_id == -1:
            topic_labels[topic_id] = "Hors-sujet"
            continue
        words = model.get_topic(topic_id)
        if words:
            topic_labels[topic_id] = " / ".join([w[0] for w in words[:3]])

    # Attache le topic à chaque paragraphe
    for i, para in enumerate(paragraphs):
        para["topic_id"] = int(topics[i])
        para["topic_label"] = topic_labels.get(int(topics[i]), "Unknown")

    result = {
        "model": model,
        "paragraphs": paragraphs,
        "topic_info": topic_info.to_dict(orient="records"),
        "topic_labels": topic_labels,
        "n_topics": len(topic_info[topic_info["Topic"] >= 0])
    }

    print(f"  → {result['n_topics']} topics extraits du corpus")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "corpus_topics.json", "w") as f:
            json.dump({
                "paragraphs": paragraphs,
                "topic_info": result["topic_info"],
                "topic_labels": {str(k): v for k, v in topic_labels.items()},
                "n_topics": result["n_topics"]
            }, f, indent=2, ensure_ascii=False)

    return result
