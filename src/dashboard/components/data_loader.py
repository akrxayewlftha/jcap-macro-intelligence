"""
Data Loader pour le dashboard — charge les données preprocessées.

On utilise st.cache_data pour ne pas recharger à chaque interaction.
Le TTL de 1h est suffisant pour les données de marché temps réel.
"""
from __future__ import annotations


import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


BASE = Path(__file__).resolve().parents[3]
PROCESSED = BASE / "data" / "processed"


@st.cache_data(ttl=3600, show_spinner=False)
def load_tweets() -> pd.DataFrame:
    """Charge les tweets enrichis depuis le fichier processed."""
    path = PROCESSED / "tweets_final.json"
    if not path.exists():
        path = PROCESSED / "tweets_enriched.json"
    if not path.exists():
        # Fallback : charge le CSV brut + minimal processing
        csv_path = BASE / "data" / "raw" / "tweets" / "financial_juice_tweets.csv"
        if csv_path.exists():
            import sys
            sys.path.insert(0, str(BASE))
            from src.ingestion.tweet_loader import load_tweets as _load
            return _load(csv_path)
        return pd.DataFrame()

    df = pd.read_json(path)
    # S'assure que les colonnes de listes sont bien des listes
    list_cols = ["macro_keywords", "central_banks", "figures", "indicators",
                 "assets", "rate_actions", "entities"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x if isinstance(x, list) else
                (json.loads(x) if isinstance(x, str) and x.startswith("[") else [])
            )
    # Reconvertit le datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_corpus() -> list[dict]:
    """Charge les documents du corpus parsés."""
    path = PROCESSED / "corpus.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner=False)
def load_emerging_themes() -> dict:
    """Charge les thèmes émergents."""
    path = PROCESSED / "emerging_themes.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner=False)
def load_consensus_report() -> dict:
    """Charge le rapport consensus/divergences."""
    path = PROCESSED / "consensus_divergence.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner=False)
def load_weak_signals() -> dict:
    """Charge les signaux faibles."""
    path = PROCESSED / "weak_signals.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner=False)
def load_cross_links() -> dict:
    """Charge l'index de cross-référence corpus ↔ tweets."""
    path = PROCESSED / "cross_reference_index.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner=False)
def load_thematic_bridges() -> list[dict]:
    """Charge les ponts thématiques."""
    path = PROCESSED / "thematic_bridges.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def check_data_status() -> dict:
    """Vérifie l'état des données processées — pour le status bar."""
    return {
        "tweets": (PROCESSED / "tweets_final.json").exists() or
                  (PROCESSED / "tweets_enriched.json").exists(),
        "corpus": (PROCESSED / "corpus.json").exists(),
        "embeddings": (PROCESSED / "tweet_embeddings.npy").exists(),
        "analysis": (PROCESSED / "emerging_themes.json").exists(),
        "pipeline_run": (PROCESSED / "corpus.json").exists(),
    }
