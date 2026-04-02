"""
LLM Digest — génère une synthèse automatique via Groq (Llama 3.3 70B).

Groq est choisi pour sa latence ultra-faible (token streaming en temps réel).
Llama 3.3 70B est excellent pour ce genre de tâche de synthèse structurée.

Si Groq n'est pas disponible, on bascule sur Claude via l'API Anthropic.
Les deux sont optionnels — le dashboard tourne sans LLM, juste avec moins
de features sur la page Synthèse.
"""
from __future__ import annotations


import os
import json
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime


# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior macro analyst at a global investment fund (JCAP).
Your role is to synthesize real-time market intelligence from Twitter feeds and
research reports into actionable macro insights.

Write in a clear, professional style — like a morning briefing note.
Be direct, specific, and highlight what matters for traders and portfolio managers.
Do NOT be generic. Focus on what's actually in the data provided."""

DIGEST_PROMPT_TEMPLATE = """Based on the following macro intelligence data, write a structured daily briefing.

=== TWEET FEED ANALYSIS ({n_tweets} tweets, {date_range}) ===
Top themes: {top_themes}
Sentiment breakdown: {sentiment_breakdown}
High-priority signals: {high_priority}
Emerging trends: {emerging_trends}

=== CORPUS ANALYSIS ({n_docs} research reports) ===
Key topics from reports: {corpus_topics}
Consensus points (tweets + reports agree): {consensus}
Divergences (tweets vs reports disagree): {divergences}

=== WEAK SIGNALS ===
{weak_signals}

Write the following sections:
1. **EXECUTIVE SUMMARY** (3-4 sentences, the most important takeaway)
2. **KEY THEMES** (bullet list of 4-5 main macro themes with brief commentary)
3. **CONSENSUS VIEW** (what both tweets and research agree on)
4. **WATCH LIST** (emerging signals and potential surprises — what to monitor)
5. **POSITIONING IMPLICATIONS** (brief thoughts on market implications)

Be specific. Reference actual entities (Fed, ECB, Oil, etc.) mentioned in the data."""

QUESTION_PROMPT_TEMPLATE = """You are a macro analyst assistant with access to the following intelligence:

=== TWEET FEED ===
{tweet_context}

=== RESEARCH CORPUS EXTRACTS ===
{corpus_context}

=== ANALYSIS RESULTS ===
{analysis_context}

User question: {question}

Answer concisely and specifically based on the data provided. If the data doesn't
contain enough information to answer definitively, say so clearly."""


# ─── Client LLM ───────────────────────────────────────────────────────────────

class LLMDigest:
    """
    Gère l'accès aux LLMs pour la génération de synthèses macro.
    Groq est préféré pour sa rapidité, Claude sert de fallback si la clé Groq
    n'est pas configurée ou si le package n'est pas installé.
    """

    def __init__(self):
        self.groq_client = None
        self.anthropic_client = None
        self._init_clients()

    def _init_clients(self):
        """Tente d'initialiser Groq puis Anthropic selon les clés disponibles dans l'environnement."""
        # Groq d'abord — beaucoup plus rapide pour le streaming
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
                print("  ✓ Groq API initialisée (Llama 3.3 70B)")
            except ImportError:
                print("  ! Package groq non installé")

        # Anthropic en fallback
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                print("  ✓ Anthropic API initialisée (Claude)")
            except ImportError:
                print("  ! Package anthropic non installé")

        if not self.groq_client and not self.anthropic_client:
            print("  ⚠ Aucune API LLM disponible — synthèse désactivée")

    @property
    def is_available(self) -> bool:
        return self.groq_client is not None or self.anthropic_client is not None

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT,
                 max_tokens: int = 1500, stream: bool = False) -> str | Iterator[str]:
        """
        Envoie le prompt au LLM disponible et retourne soit le texte complet,
        soit un générateur pour le streaming — Groq est préféré si les deux sont configurés.
        """
        if self.groq_client:
            return self._groq_generate(prompt, system, max_tokens, stream)
        elif self.anthropic_client:
            return self._anthropic_generate(prompt, system, max_tokens, stream)
        else:
            return "⚠️ Aucune API LLM configurée. Ajoutez GROQ_API_KEY ou ANTHROPIC_API_KEY dans .env"

    def _groq_generate(self, prompt: str, system: str,
                        max_tokens: int, stream: bool) -> str | Iterator[str]:
        """Appelle Groq avec Llama 3.3 70B — temperature à 0.3 pour rester factuel sans être robotique."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        if stream:
            def stream_gen():
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    stream=True
                )
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            return stream_gen()
        else:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content

    def _anthropic_generate(self, prompt: str, system: str,
                             max_tokens: int, stream: bool) -> str | Iterator[str]:
        """Fallback Claude — même interface que Groq pour que le reste du code ne sache pas la différence."""
        if stream:
            def stream_gen():
                with self.anthropic_client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}]
                ) as s:
                    for text in s.text_stream:
                        yield text
            return stream_gen()
        else:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

    def build_digest_prompt(self, analysis_data: dict) -> str:
        """
        Transforme les résultats d'analyse (DataFrames, dicts) en un bloc de texte
        structuré que le LLM peut interpréter sans ambiguité pour générer le briefing.
        """
        tweets_df = analysis_data.get("tweets_df")
        corpus_docs = analysis_data.get("corpus_docs", [])
        emerging = analysis_data.get("emerging_themes", {})
        consensus_report = analysis_data.get("consensus_report", {})
        weak_signals = analysis_data.get("weak_signals", {})

        n_tweets = len(tweets_df) if tweets_df is not None else 0
        date_range = "N/A"
        if tweets_df is not None and "date_str" in tweets_df.columns:
            dates = sorted(tweets_df["date_str"].unique())
            date_range = f"{dates[0]} to {dates[-1]}" if dates else "N/A"

        # répartition des sentiments en pourcentages pour donner une vue d'ensemble rapide
        sent_breakdown = "N/A"
        if tweets_df is not None and "sentiment" in tweets_df.columns:
            counts = tweets_df["sentiment"].value_counts().to_dict()
            total = sum(counts.values())
            sent_breakdown = " | ".join(
                f"{k}: {v} ({v/total*100:.0f}%)" for k, v in counts.items()
            )

        high_priority_tweets = []
        if tweets_df is not None and "priority" in tweets_df.columns:
            hp = tweets_df[tweets_df["priority"] == "high"]["text_clean"].head(5).tolist()
            high_priority_tweets = [t[:150] for t in hp]

        top_emerging = emerging.get("top_emerging", [])

        corpus_topics = [f"{doc['source']}: {doc.get('title', '')[:60]}"
                         for doc in corpus_docs[:5]]

        consensus_items = [c["entity"] for c in consensus_report.get("consensus", [])[:5]]
        divergence_items = [f"{d['entity']} (tweets:{d.get('tweet_sentiment','?')} vs corpus:{d.get('corpus_sentiment','?')})"
                            for d in consensus_report.get("divergences", [])[:3]]

        rare_signals = [s["entity"] for s in weak_signals.get("rare_mentions", [])[:5]]
        shifts = [f"Narrative shift at {s['datetime'][:16]}: {s['direction']} (Δ={s['delta']:+.2f})"
                  for s in weak_signals.get("narrative_shifts", [])[:3]]

        weak_signals_text = "\n".join(
            [f"- Rare mention: {s}" for s in rare_signals] +
            [f"- {s}" for s in shifts]
        ) or "No significant weak signals detected"

        return DIGEST_PROMPT_TEMPLATE.format(
            n_tweets=n_tweets,
            date_range=date_range,
            top_themes=", ".join(top_emerging) or "See analysis below",
            sentiment_breakdown=sent_breakdown,
            high_priority="\n".join(f"• {t}" for t in high_priority_tweets) or "None",
            emerging_trends=", ".join(top_emerging) or "None detected",
            n_docs=len(corpus_docs),
            corpus_topics="\n".join(f"• {t}" for t in corpus_topics) or "None",
            consensus=", ".join(consensus_items) or "None detected",
            divergences="\n".join(f"• {d}" for d in divergence_items) or "None detected",
            weak_signals=weak_signals_text,
        )

    def generate_digest(self, analysis_data: dict, stream: bool = False) -> str | Iterator[str]:
        """Point d'entrée principal : construit le prompt et génère le digest, avec fallback si pas de LLM."""
        if not self.is_available:
            return self._fallback_digest(analysis_data)

        prompt = self.build_digest_prompt(analysis_data)
        return self.generate(prompt, stream=stream)

    def answer_question(self, question: str, analysis_data: dict,
                         stream: bool = False) -> str | Iterator[str]:
        """
        Répond à une question libre en lui donnant le contexte des tweets prioritaires,
        des extraits du corpus et des résultats d'analyse comme base de raisonnement.
        """
        if not self.is_available:
            return "⚠️ LLM non disponible. Configurez GROQ_API_KEY dans .env"

        tweets_df = analysis_data.get("tweets_df")
        corpus_docs = analysis_data.get("corpus_docs", [])

        # on prend les tweets haute priorité récents — les 20 derniers donnent assez de contexte sans noyer le LLM
        tweet_context = "No tweets loaded"
        if tweets_df is not None:
            hp = tweets_df[tweets_df.get("priority", pd.Series(dtype=str)) == "high"] \
                if "priority" in tweets_df.columns else tweets_df
            sample = hp.tail(20)["text_clean"].tolist() if not hp.empty else []
            tweet_context = "\n".join(f"• {t}" for t in sample)

        corpus_context = "\n\n".join(
            f"[{doc['source']}] {doc.get('preview', '')[:300]}"
            for doc in corpus_docs[:5]
        )

        emerging = analysis_data.get("emerging_themes", {})
        consensus = analysis_data.get("consensus_report", {})
        analysis_context = json.dumps({
            "emerging_themes": emerging.get("top_emerging", []),
            "consensus": [c["entity"] for c in consensus.get("consensus", [])[:5]],
            "divergences": [d["entity"] for d in consensus.get("divergences", [])[:3]],
        }, indent=2)

        prompt = QUESTION_PROMPT_TEMPLATE.format(
            tweet_context=tweet_context,
            corpus_context=corpus_context,
            analysis_context=analysis_context,
            question=question
        )
        return self.generate(prompt, max_tokens=800, stream=stream)

    def _fallback_digest(self, analysis_data: dict) -> str:
        """Génère un résumé minimal en texte brut quand aucun LLM n'est disponible, pour ne pas laisser la page vide."""
        emerging = analysis_data.get("emerging_themes", {})
        consensus = analysis_data.get("consensus_report", {})
        tweets_df = analysis_data.get("tweets_df")

        n = len(tweets_df) if tweets_df is not None else 0
        top = emerging.get("top_emerging", ["N/A"])
        cons = [c["entity"] for c in consensus.get("consensus", [])[:3]]

        return f"""## Macro Intelligence Brief — {datetime.now().strftime('%Y-%m-%d')}

> *LLM not available — configure GROQ_API_KEY for AI-generated synthesis*

**{n} tweets analyzed** over the past 4 days.

**Top Emerging Themes:** {', '.join(top)}

**Consensus Areas:** {', '.join(cons) if cons else 'Analysis in progress'}

*For full AI synthesis, add your GROQ_API_KEY to the .env file.*"""


# ─── Import conditionnel pour éviter erreur si pandas non importé ─────────────
try:
    import pandas as pd
except ImportError:
    pass
