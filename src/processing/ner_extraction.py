"""
NER Extraction — détecte les entités macroéconomiques.

spaCy fait du bon travail sur les entités générales (pays, personnes, orgs),
mais pour les entités spécifiquement macro (indicateurs, banques centrales,
paires de devises), on complète avec des règles regex maison.

C'est une approche hybride classique : modèle pré-entraîné + règles métier.
"""
from __future__ import annotations


import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional

import spacy
import pandas as pd


# ─── Dictionnaires d'entités macro ────────────────────────────────────────────

CENTRAL_BANKS = {
    "Fed": ["fed", "federal reserve", "fomc", "federal open market"],
    "ECB": ["ecb", "european central bank", "bce"],
    "BoJ": ["boj", "bank of japan", "boj"],
    "BoE": ["boe", "bank of england"],
    "PBoC": ["pboc", "people's bank of china", "people bank china"],
    "RBA": ["rba", "reserve bank of australia"],
    "SNB": ["snb", "swiss national bank"],
    "BoC": ["boc", "bank of canada"],
    "Riksbank": ["riksbank"],
    "Norges Bank": ["norges bank"],
}

KEY_FIGURES = {
    "Powell": ["powell", "jerome powell"],
    "Lagarde": ["lagarde", "christine lagarde"],
    "Ueda": ["ueda", "kazuo ueda"],
    "Bailey": ["bailey", "andrew bailey"],
    "Waller": ["waller", "christopher waller"],
    "Williams": ["williams", "john williams"],
    "Yellen": ["yellen", "janet yellen"],
    "Bessent": ["bessent", "scott bessent"],
    "Trump": ["trump", "donald trump"],
    "Biden": ["biden"],
    "Xi Jinping": ["xi jinping", "xi"],
    "Putin": ["putin"],
}

MACRO_INDICATORS = {
    "CPI": ["cpi", "consumer price index", "inflation"],
    "PCE": ["pce", "personal consumption expenditure"],
    "GDP": ["gdp", "gross domestic product", "growth"],
    "NFP": ["nfp", "non-farm payroll", "payrolls", "jobs report"],
    "PMI": ["pmi", "purchasing managers index", "manufacturing pmi", "services pmi"],
    "ISM": ["ism", "institute supply management"],
    "JOLTS": ["jolts", "job openings"],
    "Jobless Claims": ["jobless claims", "unemployment claims", "initial claims"],
    "PPI": ["ppi", "producer price index"],
    "Retail Sales": ["retail sales"],
    "Housing": ["housing starts", "home sales", "existing home", "building permits"],
    "Trade Balance": ["trade balance", "trade deficit", "trade surplus"],
    "Current Account": ["current account"],
}

ASSET_CLASSES = {
    "US Treasuries": ["treasury", "treasuries", "10y", "2y yield", "yield curve",
                       "10-year", "30-year bond"],
    "Equities": ["s&p", "s&p 500", "nasdaq", "dow jones", "equity", "equities",
                  "stocks", "stock market"],
    "Oil": ["oil", "crude", "brent", "wti", "opec", "barrel"],
    "Gold": ["gold", "xau"],
    "USD": ["dollar", "dxy", "usd", "greenback"],
    "EUR": ["euro", "eur", "eurusd"],
    "JPY": ["yen", "jpy", "usdjpy"],
    "CNY": ["yuan", "cny", "rmb", "renminbi"],
    "Crypto": ["bitcoin", "btc", "crypto", "ethereum"],
}

GEOPOLITICAL = {
    "US": ["united states", "u.s.", "america", "washington"],
    "China": ["china", "chinese", "beijing", "prc"],
    "EU": ["europe", "european", "eurozone", "eu"],
    "Russia": ["russia", "russian", "moscow", "kremlin"],
    "Middle East": ["iran", "saudi arabia", "israel", "hormuz", "opec",
                     "middle east", "gulf"],
    "Japan": ["japan", "japanese", "tokyo"],
    "UK": ["uk", "united kingdom", "britain", "british", "london"],
}

RATE_ACTIONS = {
    "Hike": ["hike", "raise", "increase rate", "tightening", "hawkish"],
    "Cut": ["cut", "lower rate", "decrease rate", "easing", "dovish", "pivot"],
    "Hold": ["hold", "pause", "unchanged", "steady", "maintain"],
    "QE": ["qe", "quantitative easing", "asset purchase", "bond buying"],
    "QT": ["qt", "quantitative tightening", "balance sheet reduction"],
}


# ─── Extracteur principal ─────────────────────────────────────────────────────

class MacroNER:
    """
    Combine spaCy pour les entités générales et nos propres dictionnaires pour
    les termes macro spécifiques que spaCy ne connaît pas forcément bien —
    comme "FOMC", "QT" ou les noms de paires de devises.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        print(f"  Chargement du modèle spaCy : {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"  Modèle {spacy_model} non trouvé, tentative de téléchargement...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model], check=True)
            self.nlp = spacy.load(spacy_model)

        # on pré-compile une fois pour toutes plutôt qu'à chaque appel d'extract
        self._compile_patterns()

    def _compile_patterns(self):
        """Transforme les dictionnaires d'entités en regex compilées pour aller plus vite."""
        self.patterns = {}
        all_dicts = {
            "central_bank": CENTRAL_BANKS,
            "figure": KEY_FIGURES,
            "indicator": MACRO_INDICATORS,
            "asset": ASSET_CLASSES,
            "geo": GEOPOLITICAL,
            "rate_action": RATE_ACTIONS,
        }
        for category, entity_dict in all_dicts.items():
            self.patterns[category] = {}
            for canonical, variants in entity_dict.items():
                # on trie par longueur décroissante pour que "federal reserve" matche avant "fed"
                pattern_str = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
                self.patterns[category][canonical] = re.compile(
                    r"\b(" + pattern_str + r")\b", re.IGNORECASE
                )

    def extract(self, text: str) -> dict:
        """
        Extrait toutes les entités macro d'un texte et retourne un dict catégorisé,
        en combinant les règles dictionnaire et le pipeline spaCy.
        """
        text_lower = text.lower()
        entities = defaultdict(list)

        # ── Entités macro par règles ───────────────────────────────────
        for category, patterns in self.patterns.items():
            for canonical, pattern in patterns.items():
                if pattern.search(text_lower):
                    entities[category].append(canonical)

        # ── Entités spaCy (pays, personnes non couvertes) ──────────────
        doc = self.nlp(text[:1000])  # on coupe à 1000 chars pour éviter de ralentir sur les longs tweets threadés
        for ent in doc.ents:
            if ent.label_ == "GPE" and ent.text not in str(entities.get("geo", [])):
                entities["country"].append(ent.text)
            elif ent.label_ == "PERSON" and ent.text not in str(entities.get("figure", [])):
                entities["person"].append(ent.text)
            elif ent.label_ == "ORG" and ent.text not in str(entities.get("central_bank", [])):
                entities["org"].append(ent.text)
            elif ent.label_ in ("MONEY", "PERCENT"):
                entities["figure_numeric"].append(ent.text)

        # dict.fromkeys préserve l'ordre tout en dédupliquant, pratique ici
        return {k: list(dict.fromkeys(v)) for k, v in entities.items()}

    def extract_batch(self, texts: list[str], show_progress: bool = True) -> list[dict]:
        """Passe extract sur une liste de textes, avec une barre de progression optionnelle."""
        from tqdm import tqdm
        results = []
        iterator = tqdm(texts, desc="  NER") if show_progress else texts
        for text in iterator:
            results.append(self.extract(text))
        return results

    def get_entity_frequencies(self, entities_list: list[dict]) -> dict:
        """
        Agrège les entités sur tout le corpus pour savoir ce qui revient le plus.
        C'est utile pour identifier les sujets dominants avant même de regarder le sentiment.
        """
        freq = defaultdict(lambda: defaultdict(int))
        for entities in entities_list:
            for category, items in entities.items():
                for item in items:
                    freq[category][item] += 1
        # on trie par fréquence décroissante pour que le plus important soit en premier
        return {
            category: dict(sorted(counts.items(), key=lambda x: -x[1]))
            for category, counts in freq.items()
        }


def enrich_tweets_with_ner(df: pd.DataFrame, ner: MacroNER,
                            output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Ajoute les colonnes d'entités au DataFrame de tweets et sauvegarde en JSON si
    un output_dir est fourni. Les colonnes flatten (central_banks, figures, etc.)
    sont ajoutées pour faciliter les filtres en aval sans avoir à désimbriquer le dict.
    """
    print(f"  NER sur {len(df)} tweets...")
    entities_list = ner.extract_batch(df["text_clean"].tolist())

    df = df.copy()
    df["entities"] = entities_list
    # colonnes aplaties pour simplifier les requêtes sans désimbriquer entities à chaque fois
    df["central_banks"] = df["entities"].apply(lambda e: e.get("central_bank", []))
    df["figures"] = df["entities"].apply(lambda e: e.get("figure", []))
    df["indicators"] = df["entities"].apply(lambda e: e.get("indicator", []))
    df["assets"] = df["entities"].apply(lambda e: e.get("asset", []))
    df["rate_actions"] = df["entities"].apply(lambda e: e.get("rate_action", []))

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_json = df.copy()
        for col in ["entities", "central_banks", "figures", "indicators", "assets",
                    "rate_actions", "macro_keywords"]:
            if col in df_json.columns:
                df_json[col] = df_json[col].apply(
                    lambda x: x if isinstance(x, list) else list(x)
                )
        df_json["datetime"] = df_json["datetime"].astype(str)
        df_json.to_json(output_dir / "tweets_enriched.json", orient="records",
                        force_ascii=False, indent=2)

    return df


if __name__ == "__main__":
    ner = MacroNER()
    test = "Powell signals Fed may cut rates twice in 2024 as CPI falls below 3%"
    entities = ner.extract(test)
    print(f"Test : '{test}'")
    print(f"Entités : {entities}")
