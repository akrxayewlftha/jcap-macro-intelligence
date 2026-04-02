"""
Consensus & Divergences — détecte où les sources sont d'accord ou non.

L'approche :
1. Pour chaque entité macro (ex: Fed, CPI, Oil), on regarde le sentiment
   agrégé dans les tweets vs dans le corpus documentaire.
2. Si les deux sources convergent sur un sentiment → consensus.
3. Si elles divergent significativement → divergence.
4. On calcule aussi la divergence interne (tweets entre eux).

C'est l'analyse la plus "valeur ajoutée" du projet, ce qui distingue
un vrai outil d'analyse d'un simple agrégateur.
"""
from __future__ import annotations


import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional

import pandas as pd


# en dessous de ce seuil, tweets et corpus sont considérés alignés
DIVERGENCE_THRESHOLD = 0.3


def _sentiment_to_score(sentiment: str) -> float:
    """Mappe un label textuel sur une valeur numérique pour faciliter les comparaisons arithmétiques."""
    mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    return mapping.get(sentiment, 0.0)


def compute_entity_sentiment_by_source(
        df_tweets: pd.DataFrame,
        corpus_docs: list[dict],
        entity_col: str = "central_banks"
) -> pd.DataFrame:
    """
    Pour chaque entité, compare le sentiment moyen des tweets à celui du corpus documentaire
    et classe le signal en consensus ou divergence selon l'écart entre les deux.
    """
    if entity_col not in df_tweets.columns:
        return pd.DataFrame()

    # ── Sentiment tweets par entité ────────────────────────────────────
    tweet_scores = defaultdict(list)
    for _, row in df_tweets.iterrows():
        entities = row.get(entity_col, [])
        if not isinstance(entities, list):
            continue
        score = row.get("sentiment_score", 0.0)
        for entity in entities:
            tweet_scores[entity].append(score)

    # ── Sentiment corpus par entité ────────────────────────────────────
    # on cherche les entités directement dans le texte brut des documents
    corpus_scores = defaultdict(list)
    for doc in corpus_docs:
        doc_score = doc.get("sentiment_score", 0.0)
        doc_text = doc.get("full_text", "").lower()
        # on importe ici pour éviter la dépendance circulaire au niveau module
        from src.processing.ner_extraction import (
            CENTRAL_BANKS, KEY_FIGURES, MACRO_INDICATORS, ASSET_CLASSES
        )
        entity_dicts = {
            "central_banks": CENTRAL_BANKS,
            "figures": KEY_FIGURES,
            "indicators": MACRO_INDICATORS,
            "assets": ASSET_CLASSES,
        }
        entity_dict = entity_dicts.get(entity_col, {})
        for canonical, variants in entity_dict.items():
            if any(v in doc_text for v in variants):
                corpus_scores[canonical].append(doc_score)

    # ── Fusion et classification ───────────────────────────────────────
    all_entities = set(tweet_scores.keys()) | set(corpus_scores.keys())
    rows = []

    for entity in all_entities:
        t_scores = tweet_scores.get(entity, [])
        c_scores = corpus_scores.get(entity, [])

        t_mean = float(np.mean(t_scores)) if t_scores else None
        c_mean = float(np.mean(c_scores)) if c_scores else None
        t_std = float(np.std(t_scores)) if len(t_scores) > 1 else 0.0

        # si une seule source couvre l'entité, pas de comparaison possible
        if t_mean is None or c_mean is None:
            signal_type = "tweets_only" if c_mean is None else "corpus_only"
            divergence_score = 0.0
        else:
            divergence_score = abs(t_mean - c_mean)
            if divergence_score < DIVERGENCE_THRESHOLD:
                signal_type = "consensus"
            else:
                signal_type = "divergence"

        t_label = "positive" if (t_mean or 0) > 0.1 else ("negative" if (t_mean or 0) < -0.1 else "neutral")
        c_label = "positive" if (c_mean or 0) > 0.1 else ("negative" if (c_mean or 0) < -0.1 else "neutral")

        rows.append({
            "entity": entity,
            "category": entity_col,
            "tweet_sentiment": t_label,
            "tweet_score": round(t_mean, 3) if t_mean is not None else None,
            "tweet_count": len(t_scores),
            "tweet_std": round(t_std, 3),
            "corpus_sentiment": c_label,
            "corpus_score": round(c_mean, 3) if c_mean is not None else None,
            "corpus_count": len(c_scores),
            "divergence_score": round(divergence_score, 3),
            "signal_type": signal_type,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("tweet_count", ascending=False)

    return result


def detect_internal_divergence(df_tweets: pd.DataFrame,
                                 entity_col: str = "rate_actions") -> pd.DataFrame:
    """
    Regarde si, pour une même entité, les tweets se contredisent entre eux —
    une forte dispersion du sentiment sur la Fed signifie que le marché ne sait
    vraiment pas où elle va, ce qui est en soi une information utile.
    """
    if entity_col not in df_tweets.columns:
        return pd.DataFrame()

    entity_tweet_map = defaultdict(list)
    for _, row in df_tweets.iterrows():
        entities = row.get(entity_col, [])
        if not isinstance(entities, list):
            continue
        for entity in entities:
            entity_tweet_map[entity].append(row.get("sentiment_score", 0.0))

    rows = []
    for entity, scores in entity_tweet_map.items():
        if len(scores) < 3:  # trop peu de données pour mesurer une dispersion significative
            continue
        std = float(np.std(scores))
        mean = float(np.mean(scores))
        rows.append({
            "entity": entity,
            "category": entity_col,
            "n_tweets": len(scores),
            "mean_score": round(mean, 3),
            "std_score": round(std, 3),
            "is_contested": std > 0.35,  # au-delà de 0.35, les opinions sont vraiment opposées
            "label": "contested" if std > 0.35 else ("trending_positive" if mean > 0.2 else
                                                       "trending_negative" if mean < -0.2 else "neutral")
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("std_score", ascending=False)

    return result


def build_consensus_report(df_tweets: pd.DataFrame,
                             corpus_docs: list[dict],
                             output_dir: Optional[Path] = None) -> dict:
    """
    Lance l'analyse croisée tweets/corpus et la divergence interne sur toutes les
    catégories d'entités disponibles, et consolide le tout en un rapport exploitable.
    """
    entity_cols = ["central_banks", "figures", "indicators", "assets"]
    # on filtre dynamiquement pour ne pas crasher si une colonne manque
    entity_cols = [c for c in entity_cols if c in df_tweets.columns]

    all_cross = []
    all_internal = []

    for col in entity_cols:
        cross = compute_entity_sentiment_by_source(df_tweets, corpus_docs, col)
        internal = detect_internal_divergence(df_tweets, col)

        if not cross.empty:
            all_cross.append(cross)
        if not internal.empty:
            all_internal.append(internal)

    cross_df = pd.concat(all_cross) if all_cross else pd.DataFrame()
    internal_df = pd.concat(all_internal) if all_internal else pd.DataFrame()

    consensus_items = []
    divergence_items = []

    if not cross_df.empty:
        consensus_items = cross_df[cross_df["signal_type"] == "consensus"].to_dict(orient="records")
        divergence_items = cross_df[cross_df["signal_type"] == "divergence"].to_dict(orient="records")

    report = {
        "consensus": sorted(consensus_items, key=lambda x: -(x.get("tweet_count", 0))),
        "divergences": sorted(divergence_items, key=lambda x: -(x.get("divergence_score", 0))),
        "contested_in_tweets": internal_df[internal_df["is_contested"]].to_dict(orient="records") if not internal_df.empty else [],
        "n_consensus": len(consensus_items),
        "n_divergences": len(divergence_items),
        "summary": {
            "top_consensus": [c["entity"] for c in consensus_items[:3]],
            "top_divergences": [d["entity"] for d in divergence_items[:3]],
        }
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "consensus_divergence.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False,
                      default=lambda x: None if x != x else x)  # gère les NaN de pandas proprement
        print(f"  {report['n_consensus']} consensus, {report['n_divergences']} divergences")

    return report
