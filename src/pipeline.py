"""
Pipeline principal — orchestre tout le preprocessing et l'analyse.

Lance ce script une fois pour générer tous les fichiers dans processed/.
Après ça, le dashboard lit directement depuis processed/ sans avoir besoin
de relancer le pipeline, sauf si les données sources changent.

Usage :
    python src/pipeline.py
    python src/pipeline.py --skip-embeddings  # recharge les embeddings depuis le cache
    python src/pipeline.py --fast             # saute FinBERT (environ 5x plus rapide)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent


def run_pipeline(skip_embeddings: bool = False, fast_mode: bool = False):
    print("\n" + "="*60)
    print("  JCAP Macro Analysis — Pipeline de preprocessing")
    print("="*60)

    raw_corpus = BASE / "data" / "raw" / "corpus"
    raw_tweets = BASE / "data" / "raw" / "tweets" / "financial_juice_tweets.csv"
    processed = BASE / "data" / "processed"
    cache_dir = processed / "cache"

    # Phase 1 : Ingestion
    # On parse les PDFs une seule fois et on met en cache le résultat.
    # Si le fichier existe déjà, on le recharge directement.
    print("\n[1/5] Ingestion des données...")

    from src.ingestion.pdf_parser import parse_corpus
    corpus_path = processed / "corpus.json"
    if corpus_path.exists():
        print(f"  Corpus deja parse, chargement depuis {corpus_path}")
        with open(corpus_path) as f:
            documents = json.load(f)
    else:
        documents = parse_corpus(raw_corpus, corpus_path)

    from src.ingestion.tweet_loader import load_tweets
    tweets_path = processed / "tweets"
    parquet_path = tweets_path.with_suffix(".parquet")
    if parquet_path.exists():
        print(f"  Tweets deja charges, chargement depuis {parquet_path}")
        df_tweets = pd.read_parquet(parquet_path)
    else:
        df_tweets = load_tweets(raw_tweets, tweets_path)

    # Phase 2 : Embeddings
    # all-MiniLM-L6-v2 avec normalisation L2. Les embeddings sont sauvegardés
    # en .npy pour eviter de les recalculer a chaque relance.
    print("\n[2/5] Generation des embeddings...")

    from src.processing.embeddings import EmbeddingEngine, embed_corpus, embed_tweets
    engine = EmbeddingEngine(cache_dir=cache_dir)

    corpus_emb_path = processed / "corpus_embeddings.npy"
    tweet_emb_path = processed / "tweet_embeddings.npy"

    if skip_embeddings and corpus_emb_path.exists() and tweet_emb_path.exists():
        print("  Chargement des embeddings depuis le cache...")
        corpus_embeddings = np.load(corpus_emb_path)
        tweet_embeddings = np.load(tweet_emb_path)
    else:
        corpus_embeddings = embed_corpus(documents, engine, processed)
        tweet_embeddings = embed_tweets(df_tweets, engine, processed)

    # Phase 3 : Topic Modeling
    # BERTopic est tolerant aux echecs — si le corpus est trop petit pour
    # un topic coherent, il renvoie un topic unique. On continue dans tous les cas.
    print("\n[3/5] Topic Modeling (BERTopic)...")

    corpus_topics_path = processed / "corpus_topics.json"
    tweet_topics_path = processed / "tweet_topics.json"

    corpus_topics_result = None
    try:
        if corpus_topics_path.exists() and skip_embeddings:
            print("  Topics corpus deja calcules, skip...")
            with open(corpus_topics_path) as f:
                corpus_topics_result = json.load(f)
        else:
            from src.processing.topic_modeling import extract_corpus_topics
            from src.processing.topic_modeling import _split_into_paragraphs
            paragraphs = _split_into_paragraphs(documents)
            para_texts = [p["text"] for p in paragraphs]
            if para_texts:
                para_embeddings = engine.encode(para_texts, show_progress=True)
                np.save(processed / "para_embeddings.npy", para_embeddings)
                corpus_topics_result = extract_corpus_topics(
                    documents, para_embeddings, processed
                )
    except Exception as e:
        print(f"  Topic modeling corpus echoue (non bloquant) : {e}")

    tweet_topics_result = None
    try:
        if tweet_topics_path.exists() and skip_embeddings:
            print("  Topics tweets deja calcules, skip...")
            with open(tweet_topics_path) as f:
                tweet_topics_result = json.load(f)
        else:
            from src.processing.topic_modeling import extract_tweet_topics
            tweet_topics_result = extract_tweet_topics(df_tweets, tweet_embeddings, processed)
            if "df" in tweet_topics_result:
                df_tweets = tweet_topics_result["df"]
    except Exception as e:
        print(f"  Topic modeling tweets echoue (non bloquant) : {e}")

    # Phase 4 : NER + Sentiment
    # NER hybride (spaCy + regles regex) pour les entites macro.
    # FinBERT pour le sentiment — on le saute en mode fast car il prend 3 a 5 min.
    print("\n[4/5] NER + Sentiment Analysis...")

    enriched_path = processed / "tweets_enriched.json"

    if enriched_path.exists() and skip_embeddings:
        print("  Tweets enrichis deja calcules, chargement...")
        df_tweets = pd.read_json(enriched_path)
    else:
        from src.processing.ner_extraction import MacroNER, enrich_tweets_with_ner
        ner = MacroNER()
        df_tweets = enrich_tweets_with_ner(df_tweets, ner, processed)

        if not fast_mode:
            from src.processing.sentiment import SentimentAnalyzer, enrich_tweets_sentiment
            analyzer = SentimentAnalyzer()
            df_tweets = enrich_tweets_sentiment(df_tweets, analyzer, processed)

            from src.processing.sentiment import compute_corpus_sentiment
            documents = compute_corpus_sentiment(documents, analyzer, processed)
            with open(corpus_path, "w") as f:
                # On sauvegarde sans section_sentiments qui alourdit inutilement le fichier
                docs_light = [{k: v for k, v in d.items() if k != "section_sentiments"}
                               for d in documents]
                json.dump(docs_light, f, ensure_ascii=False, indent=2)
        else:
            print("  Fast mode : sentiment ignore, valeurs neutres par defaut")
            df_tweets["sentiment"] = "neutral"
            df_tweets["sentiment_score"] = 0.0

    # Phase 5 : Analyse avancee
    # Detection de themes emergents, consensus/divergences entre tweets et corpus,
    # signaux faibles, et cross-linking semantique tweets <-> rapports.
    print("\n[5/5] Analyse avancee (themes, consensus, signaux)...")

    from src.analysis.theme_detector import detect_emerging_themes, save_emerging_themes
    emerging = detect_emerging_themes(df_tweets)
    save_emerging_themes(emerging, processed)

    from src.analysis.consensus_divergence import build_consensus_report
    with open(corpus_path) as f:
        docs_final = json.load(f)
    consensus_report = build_consensus_report(df_tweets, docs_final, processed)

    from src.analysis.weak_signals import build_weak_signals_report
    weak_signals = build_weak_signals_report(df_tweets, processed)

    # Cross-linking semantique : pour chaque tweet, trouver les passages
    # du corpus les plus proches par similarite cosinus.
    para_emb_path = processed / "para_embeddings.npy"
    if para_emb_path.exists() and corpus_topics_result:
        from src.analysis.cross_source_linker import (
            link_tweets_to_corpus, build_cross_reference_index,
            find_thematic_bridges, save_cross_links
        )
        para_embeddings = np.load(para_emb_path)
        paragraphs = corpus_topics_result.get("paragraphs", [])
        if paragraphs:
            links = link_tweets_to_corpus(df_tweets, tweet_embeddings, paragraphs, para_embeddings)
            ref_index = build_cross_reference_index(links)
            bridges = find_thematic_bridges(df_tweets, tweet_embeddings, docs_final, corpus_embeddings)
            save_cross_links(links, ref_index, bridges, processed)

    # Sauvegarde finale du DataFrame enrichi
    # On convertit les colonnes listes en JSON serializable avant d'ecrire.
    df_tweets["datetime"] = df_tweets["datetime"].astype(str)
    for col in ["macro_keywords", "entities", "central_banks", "figures",
                "indicators", "assets", "rate_actions"]:
        if col in df_tweets.columns:
            df_tweets[col] = df_tweets[col].apply(
                lambda x: list(x) if hasattr(x, '__iter__') and not isinstance(x, str) else x
            )
    df_tweets.to_json(processed / "tweets_final.json", orient="records",
                       force_ascii=False, indent=2)

    print("\n" + "="*60)
    print("  Pipeline termine avec succes !")
    print(f"  {len(documents)} documents parses")
    print(f"  {len(df_tweets)} tweets enrichis")
    print(f"  {emerging.get('n_emerging', 0)} themes emergents")
    print(f"  {consensus_report.get('n_consensus', 0)} consensus, "
          f"{consensus_report.get('n_divergences', 0)} divergences")
    print("="*60)
    print(f"\nLancez le dashboard : streamlit run src/dashboard/app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JCAP Macro Analysis Pipeline")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Reutilise les embeddings du cache au lieu de les recalculer")
    parser.add_argument("--fast", action="store_true",
                        help="Mode rapide : saute FinBERT (sentiment neutre par defaut)")
    args = parser.parse_args()

    sys.path.insert(0, str(BASE))
    run_pipeline(skip_embeddings=args.skip_embeddings, fast_mode=args.fast)
