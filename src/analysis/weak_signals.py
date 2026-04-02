"""
Weak Signals — détecte les signaux faibles dans le flux d'information.

Un signal faible = mention rare mais potentiellement significative.
En macro, ça peut être : une banque centrale peu médiatisée qui fait
une déclaration inhabituelle, un indicateur secondaire qui se dégrade
discrètement, une géographie émergente qui commence à faire parler d'elle.

L'approche : on cherche les entités/termes qui ont une fréquence basse
mais un engagement élevé (tweets haute priorité) ou une co-occurrence
inhabituelle avec des entités "core" (Fed, CPI...).
"""
from __future__ import annotations


import re
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional

import pandas as pd


# les entités qu'on voit toujours — inutile de les traiter comme des signaux faibles
CORE_ENTITIES = {
    "central_banks": ["Fed", "ECB"],
    "indicators": ["CPI", "GDP", "NFP"],
    "assets": ["USD", "US Treasuries", "Oil"],
    "figures": ["Powell", "Trump"],
}

# on considère qu'une entité est "rare" si elle apparaît moins de N fois sur toute la période
RARITY_THRESHOLD = 5


def extract_rare_mentions(df: pd.DataFrame,
                           entity_cols: list = None) -> pd.DataFrame:
    """
    Remonte les entités peu fréquentes qui apparaissent quand même dans des tweets
    jugés importants — c'est là que se cachent les vrais signaux faibles à surveiller.
    """
    if entity_cols is None:
        entity_cols = ["central_banks", "figures", "indicators", "assets", "geo"]
    entity_cols = [c for c in entity_cols if c in df.columns]

    global_counts = defaultdict(lambda: defaultdict(int))
    entity_tweets = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        for col in entity_cols:
            entities = row.get(col, [])
            if not isinstance(entities, list):
                continue
            for entity in entities:
                global_counts[col][entity] += 1
                entity_tweets[col][entity].append({
                    "id": row.get("id", ""),
                    "text": row.get("text_clean", ""),
                    "priority": row.get("priority", "normal"),
                    "datetime": str(row.get("datetime", "")),
                    "sentiment": row.get("sentiment", "neutral"),
                    "sentiment_score": float(row.get("sentiment_score", 0.0))
                })

    rows = []
    for col in entity_cols:
        core = CORE_ENTITIES.get(col, [])

        for entity, count in global_counts[col].items():
            if entity in core:
                continue
            if count > RARITY_THRESHOLD:
                continue

            tweets = entity_tweets[col][entity]
            high_priority = [t for t in tweets if t["priority"] == "high"]
            avg_sentiment = np.mean([t["sentiment_score"] for t in tweets]) if tweets else 0

            # un signal faible mérite l'attention si une grande part de ses mentions vient de comptes prioritaires
            signal_strength = (len(high_priority) / max(len(tweets), 1)) * count

            rows.append({
                "entity": entity,
                "category": col,
                "total_mentions": count,
                "high_priority_mentions": len(high_priority),
                "signal_strength": round(signal_strength, 3),
                "avg_sentiment": round(float(avg_sentiment), 3),
                "example_tweets": [t["text"] for t in tweets[:2]],
                "first_seen": min([t["datetime"] for t in tweets]) if tweets else "",
                "last_seen": max([t["datetime"] for t in tweets]) if tweets else "",
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("signal_strength", ascending=False)

    return result


def find_unusual_cooccurrences(df: pd.DataFrame,
                                 entity_cols: list = None) -> list[dict]:
    """
    Cherche les paires d'entités qui apparaissent ensemble bien plus souvent
    qu'on ne le prédirait par hasard — un lift élevé sur une paire rare peut
    indiquer qu'un lien inattendu est en train de se former dans le discours.
    """
    if entity_cols is None:
        entity_cols = ["central_banks", "indicators"]
    entity_cols = [c for c in entity_cols if c in df.columns]

    if len(entity_cols) < 2:
        return []

    cooc = defaultdict(int)
    individual = defaultdict(int)

    for _, row in df.iterrows():
        all_entities = []
        for col in entity_cols:
            entities = row.get(col, [])
            if isinstance(entities, list):
                all_entities.extend(entities)

        # toutes les paires possibles dans le même tweet
        for i, e1 in enumerate(all_entities):
            individual[e1] += 1
            for e2 in all_entities[i+1:]:
                pair = tuple(sorted([e1, e2]))
                cooc[pair] += 1

    # le lift mesure à quel point la co-occurrence est surprenante par rapport à l'indépendance statistique
    n = len(df)
    unusual = []
    for (e1, e2), count in cooc.items():
        if count < 2:
            continue
        p_e1 = individual[e1] / n
        p_e2 = individual[e2] / n
        expected = p_e1 * p_e2 * n
        lift = count / max(expected, 0.1)

        # on veut des paires rares mais sur-représentées, pas les paires banales comme Fed+CPI
        if lift > 2.0 and count <= RARITY_THRESHOLD:
            unusual.append({
                "entity_1": e1,
                "entity_2": e2,
                "cooc_count": count,
                "lift": round(lift, 2),
                "is_unusual": True
            })

    return sorted(unusual, key=lambda x: -x["lift"])[:20]


def detect_narrative_shifts(df: pd.DataFrame, window_size: int = 12) -> list[dict]:
    """
    Repère les moments où le sentiment moyen saute brutalement d'une fenêtre
    à la suivante — souvent le signe qu'une publication ou un événement a changé
    le ton du discours, même sans qu'on sache encore pourquoi.
    """
    if "sentiment_score" not in df.columns or "datetime" not in df.columns:
        return []

    df_sorted = df.sort_values("datetime").copy()
    scores = df_sorted["sentiment_score"].values

    shifts = []
    for i in range(window_size, len(scores) - window_size):
        before = np.mean(scores[i - window_size:i])
        after = np.mean(scores[i:i + window_size])
        delta = after - before

        if abs(delta) > 0.4:  # en dessous de 0.4, c'est du bruit normal
            pivot_tweet = df_sorted.iloc[i]
            shifts.append({
                "index": i,
                "datetime": str(pivot_tweet.get("datetime", "")),
                "date_str": str(pivot_tweet.get("date_str", "")),
                "hour": int(pivot_tweet.get("hour", 0)),
                "sentiment_before": round(float(before), 3),
                "sentiment_after": round(float(after), 3),
                "delta": round(float(delta), 3),
                "direction": "positive_shift" if delta > 0 else "negative_shift",
                "pivot_tweet": pivot_tweet.get("text_clean", "")[:200],
            })

    if not shifts:
        return []

    # on ne garde qu'un shift par fenêtre et on choisit le plus fort en cas de chevauchement
    filtered = [shifts[0]]
    for shift in shifts[1:]:
        if abs(shift["index"] - filtered[-1]["index"]) > window_size:
            filtered.append(shift)
        elif abs(shift["delta"]) > abs(filtered[-1]["delta"]):
            filtered[-1] = shift

    return sorted(filtered, key=lambda x: -abs(x["delta"]))


def build_weak_signals_report(df: pd.DataFrame,
                                output_dir: Optional[Path] = None) -> dict:
    """
    Agrège les trois types de signaux faibles (mentions rares, co-occurrences
    inhabituelles, ruptures narratives) en un seul rapport prêt à l'emploi.
    """
    rare = extract_rare_mentions(df)
    cooc = find_unusual_cooccurrences(df)
    shifts = detect_narrative_shifts(df)

    report = {
        "rare_mentions": rare.to_dict(orient="records") if not rare.empty else [],
        "unusual_cooccurrences": cooc,
        "narrative_shifts": shifts,
        "n_rare_signals": len(rare) if not rare.empty else 0,
        "n_unusual_cooc": len(cooc),
        "n_shifts": len(shifts),
        "top_weak_signals": rare.head(5)["entity"].tolist() if not rare.empty else [],
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # json.dump ne sait pas gérer les types numpy directement, d'où le serializer custom
        def serialize(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(output_dir / "weak_signals.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=serialize)

        print(f"  {report['n_rare_signals']} signaux faibles, {report['n_shifts']} ruptures narratives")

    return report
