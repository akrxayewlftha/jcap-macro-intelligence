"""
Theme Detector — identifie les thèmes émergents sur la période.

"Émergent" = topic dont la fréquence augmente dans le temps.
On regarde si un sujet est plus présent dans les tweets récents
qu'au début de la période. C'est une forme de trend detection simple
mais efficace sur une fenêtre de 4 jours.
"""
from __future__ import annotations


import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional

import pandas as pd


def compute_topic_velocity(df: pd.DataFrame,
                            entity_col: str = "central_banks",
                            window_days: int = 2) -> pd.DataFrame:
    """
    Calcule à quelle vitesse un sujet gagne ou perd du terrain sur la période.
    Un ratio > 1 veut dire que le sujet accélère, < 1 qu'il se tasse.
    Les sujets absents en début de période ont une vélocité quasi-infinie — c'est voulu,
    c'est précisément ce qu'on cherche à capturer.
    """
    if "date_str" not in df.columns:
        return pd.DataFrame()

    dates = sorted(df["date_str"].unique())
    if len(dates) < 2:
        return pd.DataFrame()

    # on coupe la période en deux moitiés : ancienne vs récente
    mid = len(dates) // 2
    old_dates = dates[:mid]
    recent_dates = dates[mid:]

    df_old = df[df["date_str"].isin(old_dates)]
    df_recent = df[df["date_str"].isin(recent_dates)]

    def count_entities(subset, col):
        counts = defaultdict(int)
        for items in subset[col].dropna():
            if isinstance(items, list):
                for item in items:
                    counts[item] += 1
        return counts

    counts_old = count_entities(df_old, entity_col)
    counts_recent = count_entities(df_recent, entity_col)

    all_entities = set(counts_old.keys()) | set(counts_recent.keys())

    rows = []
    n_old = max(len(df_old), 1)
    n_recent = max(len(df_recent), 1)

    for entity in all_entities:
        freq_old = counts_old.get(entity, 0) / n_old
        freq_recent = counts_recent.get(entity, 0) / n_recent
        # le petit epsilon évite la division par zéro sans biaiser la vélocité sur les entités très actives
        velocity = freq_recent / (freq_old + 1e-6)
        rows.append({
            "entity": entity,
            "category": entity_col,
            "count_old": counts_old.get(entity, 0),
            "count_recent": counts_recent.get(entity, 0),
            "freq_old": round(freq_old, 4),
            "freq_recent": round(freq_recent, 4),
            "velocity": round(velocity, 2),
            "is_emerging": velocity > 1.5 and counts_recent.get(entity, 0) >= 2,
            "is_declining": velocity < 0.5 and counts_old.get(entity, 0) >= 2,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("velocity", ascending=False)

    return result


def detect_emerging_themes(df: pd.DataFrame,
                             entity_cols: list = None) -> dict:
    """
    Passe compute_topic_velocity sur toutes les catégories d'entités et consolide
    le résultat en un seul dict avec les thèmes qui montent et ceux qui déclinent.
    """
    if entity_cols is None:
        entity_cols = ["central_banks", "figures", "indicators", "assets", "rate_actions"]

    # on ne garde que les colonnes vraiment présentes pour ne pas planter silencieusement
    entity_cols = [c for c in entity_cols if c in df.columns]

    all_emerging = []
    all_declining = []
    velocity_by_category = {}

    for col in entity_cols:
        velocity_df = compute_topic_velocity(df, entity_col=col)
        if velocity_df.empty:
            continue

        velocity_by_category[col] = velocity_df.to_dict(orient="records")

        emerging = velocity_df[velocity_df["is_emerging"]]["entity"].tolist()
        declining = velocity_df[velocity_df["is_declining"]]["entity"].tolist()

        all_emerging.extend([(e, col, velocity_df[velocity_df["entity"] == e]["velocity"].values[0])
                              for e in emerging])
        all_declining.extend([(e, col, velocity_df[velocity_df["entity"] == e]["velocity"].values[0])
                               for e in declining])

    # les thèmes qui accélèrent le plus vite en premier
    all_emerging.sort(key=lambda x: -x[2])

    if "date_str" in df.columns:
        volume_by_day = df.groupby("date_str").size().to_dict()
    else:
        volume_by_day = {}

    return {
        "emerging_themes": [{"entity": e, "category": c, "velocity": v}
                             for e, c, v in all_emerging],
        "declining_themes": [{"entity": e, "category": c, "velocity": v}
                              for e, c, v in all_declining],
        "velocity_by_category": velocity_by_category,
        "volume_by_day": volume_by_day,
        "top_emerging": [e for e, c, v in all_emerging[:5]],
        "n_emerging": len(all_emerging),
        "n_declining": len(all_declining),
    }


def compute_hourly_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produit un pivot heure x date avec le nombre de tweets par case,
    directement exploitable par la heatmap du dashboard.
    """
    if "date_str" not in df.columns or "hour" not in df.columns:
        return pd.DataFrame()

    heatmap = df.groupby(["date_str", "hour"]).agg(
        count=("id", "count"),
        avg_sentiment=("sentiment_score", "mean") if "sentiment_score" in df.columns else ("id", "count")
    ).reset_index()

    return heatmap.pivot(index="hour", columns="date_str", values="count").fillna(0)


def save_emerging_themes(themes: dict, output_dir: Path):
    """Sérialise les thèmes en JSON en gérant proprement les types numpy que pandas peut laisser traîner."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_dir / "emerging_themes.json", "w") as f:
        json.dump(themes, f, indent=2, default=convert)

    print(f"  {themes['n_emerging']} thèmes émergents détectés")
    if themes["top_emerging"]:
        print(f"  Top émergents : {', '.join(themes['top_emerging'])}")
