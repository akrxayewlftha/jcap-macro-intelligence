"""
Tweet Loader — charge et nettoie le CSV Financial Juice.

Financial Juice c'est un compte Twitter qui agrège les headlines
macro en temps réel. Le CSV a 3 colonnes : date, author_name, content.
"""
from __future__ import annotations


import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


# ─── Helpers ───────────────────────────────────────────────────────────────────

# Emojis colorés que Financial Juice utilise comme indicateurs de priorité
PRIORITY_EMOJIS = {
    "🔴": "high",     # Breaking / très important
    "🟠": "medium",   # Important
    "🟡": "medium",   # Intermédiaire
    "🟢": "low",      # Info normale
    "⚫": "low",      # Neutre
    "🔵": "low",      # Info générale
    "⚪": "low",
}

def _detect_priority(text: str) -> str:
    """Déduit le niveau de priorité depuis les emojis en tête de tweet."""
    for emoji, level in PRIORITY_EMOJIS.items():
        if text.startswith(emoji):
            return level
    # Pas d'emoji = info standard
    return "normal"


def _clean_tweet(text: str) -> str:
    """
    Nettoie un tweet : retire les emojis de priorité en tête,
    normalise les espaces, etc.
    On garde les autres emojis car ils peuvent être informatifs.
    """
    # Retire l'emoji de priorité en début (s'il y en a un)
    for emoji in PRIORITY_EMOJIS:
        if text.startswith(emoji):
            text = text[len(emoji):].strip()
            break
    # Normalise les espaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _parse_twitter_date(date_str: str) -> Optional[datetime]:
    """
    Parse le format de date Twitter : "Tue Mar 31 11:13:40 +0000 2026"
    """
    try:
        # Format Twitter standard
        dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        return dt
    except ValueError:
        try:
            # Tentative ISO 8601 (au cas où)
            return datetime.fromisoformat(date_str)
        except ValueError:
            return None


def _extract_macro_keywords(text: str) -> list[str]:
    """
    Détecte rapidement les mots-clés macro présents dans un tweet.
    Utile pour le filtrage rapide avant le NLP lourd.
    """
    # Liste des entités et indicateurs macro courants
    keywords = [
        # Banques centrales
        "fed", "fomc", "ecb", "bce", "boj", "boe", "pboc", "rba", "snb",
        "powell", "lagarde", "ueda", "bailey", "waller", "brainard",
        # Indicateurs
        "cpi", "pce", "gdp", "nfp", "pmi", "ism", "jolts", "cpi", "inflation",
        "unemployment", "payroll", "jobless", "claims",
        # Politique monétaire
        "rate", "rates", "hike", "cut", "pivot", "pause", "taper",
        "quantitative", "qe", "qt", "basis points", "bps",
        # Géopolitique / marchés
        "oil", "crude", "brent", "wti", "gold", "yields", "treasury",
        "dollar", "yen", "euro", "yuan", "tariff", "sanctions",
        "iran", "hormuz", "opec", "china", "russia", "ukraine",
        # Macro général
        "recession", "growth", "stagflation", "deflation", "liquidity",
        "debt", "deficit", "gdp", "trade", "surplus",
    ]

    text_lower = text.lower()
    found = [kw for kw in keywords if kw in text_lower]
    # Déduplique en gardant l'ordre
    return list(dict.fromkeys(found))


# ─── Loader principal ──────────────────────────────────────────────────────────

def load_tweets(csv_path: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Charge le CSV Financial Juice et retourne un DataFrame enrichi.

    Colonnes ajoutées :
    - datetime : timestamp parsé
    - date_str : date lisible (ex : "2026-03-31")
    - hour : heure (pour regroupement horaire)
    - priority : high / medium / normal / low
    - text_clean : texte sans emoji de priorité
    - macro_keywords : liste de mots-clés macro détectés
    - char_count : longueur du tweet
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    print(f"  → {len(df)} tweets chargés depuis {csv_path.name}")

    # ── Nettoyage de base ──────────────────────────────────────────────
    # Droppe les lignes sans contenu
    df = df.dropna(subset=["content"]).copy()
    df["content"] = df["content"].astype(str).str.strip()
    df = df[df["content"] != ""].copy()

    # ── Parsing des dates ──────────────────────────────────────────────
    df["datetime"] = df["date"].apply(_parse_twitter_date)
    # Trie par date croissante
    df = df.sort_values("datetime").reset_index(drop=True)

    df["date_str"] = df["datetime"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()

    # ── Enrichissement ────────────────────────────────────────────────
    df["priority"] = df["content"].apply(_detect_priority)
    df["text_clean"] = df["content"].apply(_clean_tweet)
    df["macro_keywords"] = df["text_clean"].apply(_extract_macro_keywords)
    df["char_count"] = df["text_clean"].apply(len)
    df["keyword_count"] = df["macro_keywords"].apply(len)

    # ── ID unique ─────────────────────────────────────────────────────
    df["id"] = [f"tweet_{i:05d}" for i in range(len(df))]

    # ── Stats ─────────────────────────────────────────────────────────
    print(f"  → Période : {df['date_str'].min()} → {df['date_str'].max()}")
    print(f"  → Priorités : {df['priority'].value_counts().to_dict()}")
    print(f"  → {df['keyword_count'].sum()} mentions macro détectées")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Sauvegarde Parquet pour les performances, JSON pour lisibilité
        df.to_parquet(output_path.with_suffix(".parquet"), index=False)
        # Pour le JSON on convertit les types non-sérialisables
        df_json = df.copy()
        df_json["datetime"] = df_json["datetime"].astype(str)
        df_json["macro_keywords"] = df_json["macro_keywords"].apply(list)
        df_json.to_json(output_path.with_suffix(".json"), orient="records",
                        force_ascii=False, indent=2)
        print(f"  Tweets sauvegardés → {output_path.with_suffix('.parquet')}")

    return df


def get_tweets_by_day(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Découpe le DataFrame en sous-DataFrames par jour — pratique pour l'analyse temporelle."""
    return {date: group for date, group in df.groupby("date_str")}


def get_high_priority_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre uniquement les tweets haute priorité (🔴)."""
    return df[df["priority"] == "high"].copy()


# ─── Point d'entrée standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2]
    csv_path = BASE / "data" / "raw" / "tweets" / "financial_juice_tweets.csv"
    output_path = BASE / "data" / "processed" / "tweets"

    print("=== Chargement des tweets Financial Juice ===\n")
    df = load_tweets(csv_path, output_path)
    print(f"\nFait ! {len(df)} tweets dans {output_path}")
