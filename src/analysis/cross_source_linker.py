"""
Cross-Source Linker — relie les tweets aux sections pertinentes du corpus.

L'idée : quand un tweet parle d'un sujet (ex: "Fed rate cut"), on veut
retrouver automatiquement les passages des rapports qui traitent du même
thème. C'est du retrieval sémantique basé sur les embeddings.

On utilise la similarité cosinus entre les embeddings du tweet et ceux
des paragraphes du corpus — ça fonctionne bien parce qu'on a normalisé
les vecteurs à l'encode (voir embeddings.py).
"""
from __future__ import annotations


import json
import numpy as np
from pathlib import Path
from typing import Optional

import pandas as pd


def link_tweets_to_corpus(df_tweets: pd.DataFrame,
                           tweet_embeddings: np.ndarray,
                           paragraphs: list[dict],
                           para_embeddings: np.ndarray,
                           top_k: int = 3,
                           min_similarity: float = 0.45) -> list[dict]:
    """
    Pour chaque tweet, trouve les K passages du corpus les plus pertinents.

    On ne garde que les matches au-dessus du seuil de similarité pour
    éviter les faux positifs — un match à 0.2 de similarité n'apporte
    aucune valeur.

    Retourne une liste de liens tweet → passages corpus.
    """
    if len(tweet_embeddings) == 0 or len(para_embeddings) == 0:
        return []

    # Matrice de similarité : (n_tweets, n_paragraphs)
    # Les embeddings sont déjà normalisés → cosine = dot product
    similarity_matrix = np.dot(tweet_embeddings, para_embeddings.T)

    links = []
    for i, (_, row) in enumerate(df_tweets.iterrows()):
        tweet_sims = similarity_matrix[i]
        # Trouve les top_k paragraphes les plus similaires
        top_indices = np.argsort(tweet_sims)[::-1][:top_k]

        matched_passages = []
        for idx in top_indices:
            sim_score = float(tweet_sims[idx])
            if sim_score < min_similarity:
                break
            para = paragraphs[idx]
            matched_passages.append({
                "doc_id": para.get("doc_id", ""),
                "doc_source": para.get("doc_source", ""),
                "page": para.get("page", 1),
                "text_excerpt": para.get("text", "")[:300],
                "similarity": round(sim_score, 3),
                "topic_label": para.get("topic_label", ""),
            })

        if matched_passages:
            links.append({
                "tweet_id": row.get("id", f"tweet_{i}"),
                "tweet_text": row.get("text_clean", ""),
                "tweet_date": str(row.get("date_str", "")),
                "matched_passages": matched_passages,
                "best_match_source": matched_passages[0]["doc_source"] if matched_passages else None,
                "best_similarity": matched_passages[0]["similarity"] if matched_passages else 0.0,
            })

    print(f"  {len(links)} tweets liés au corpus ({len(links)/max(len(df_tweets),1)*100:.1f}%)")
    return links


def build_cross_reference_index(links: list[dict]) -> dict:
    """
    Construit un index inversé : pour chaque document, quels tweets le référencent ?
    Utile pour la vue "corpus" du dashboard — on veut savoir quels tweets
    parlent du même sujet que chaque rapport.
    """
    index = {}
    for link in links:
        for passage in link.get("matched_passages", []):
            doc_id = passage["doc_id"]
            if doc_id not in index:
                index[doc_id] = []
            index[doc_id].append({
                "tweet_id": link["tweet_id"],
                "tweet_text": link["tweet_text"],
                "tweet_date": link["tweet_date"],
                "similarity": passage["similarity"],
                "page": passage["page"],
            })

    # Trie par similarité décroissante dans chaque doc
    for doc_id in index:
        index[doc_id].sort(key=lambda x: -x["similarity"])
        # Garde max 10 tweets par document
        index[doc_id] = index[doc_id][:10]

    return index


def find_thematic_bridges(df_tweets: pd.DataFrame,
                           tweet_embeddings: np.ndarray,
                           documents: list[dict],
                           doc_embeddings: np.ndarray,
                           threshold: float = 0.5) -> list[dict]:
    """
    Identifie les "ponts thématiques" : sujets qui apparaissent fortement
    dans PLUSIEURS documents ET dans les tweets.

    C'est différent des liens individuels — ici on cherche des thèmes
    transversaux, pas des références point à point.
    """
    if len(tweet_embeddings) == 0 or len(doc_embeddings) == 0:
        return []

    sim_matrix = np.dot(tweet_embeddings, doc_embeddings.T)

    bridges = []
    for j, doc in enumerate(documents):
        doc_sims = sim_matrix[:, j]
        relevant_tweets = np.where(doc_sims > threshold)[0]

        if len(relevant_tweets) < 3:  # Au moins 3 tweets pour parler de "pont"
            continue

        avg_sim = float(np.mean(doc_sims[relevant_tweets]))
        top_tweet_indices = relevant_tweets[np.argsort(doc_sims[relevant_tweets])[::-1][:3]]

        top_tweets = []
        for idx in top_tweet_indices:
            row = df_tweets.iloc[idx]
            top_tweets.append(row.get("text_clean", "")[:150])

        bridges.append({
            "doc_id": doc["id"],
            "doc_source": doc["source"],
            "doc_title": doc.get("title", ""),
            "n_relevant_tweets": int(len(relevant_tweets)),
            "avg_similarity": round(avg_sim, 3),
            "top_tweets": top_tweets,
        })

    return sorted(bridges, key=lambda x: -(x["n_relevant_tweets"]))


def save_cross_links(links: list[dict], index: dict,
                     bridges: list[dict], output_dir: Path):
    """Sauvegarde tous les résultats du cross-linking."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "cross_links.json", "w") as f:
        json.dump(links[:500], f, indent=2, ensure_ascii=False)  # Limite pour le fichier

    with open(output_dir / "cross_reference_index.json", "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    with open(output_dir / "thematic_bridges.json", "w") as f:
        json.dump(bridges, f, indent=2, ensure_ascii=False)

    print(f"  Cross-links sauvegardés → {output_dir}")
    print(f"  {len(bridges)} ponts thématiques identifiés")
