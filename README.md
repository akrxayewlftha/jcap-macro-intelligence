# JCAP Macro Intelligence Dashboard

Un prototype d'analyse macro qui croise un flux de tweets financiers (Financial Juice) avec un corpus de 14 rapports de recherche institutionnelle. L'objectif est de détecter en temps quasi-réel les thèmes émergents, les points de consensus et les divergences entre ce que le marché dit et ce que les analystes écrivent.

## Ce que le dashboard fait

**Feed** — le flux brut de tweets, filtrable par date, priorité, sentiment et mot-clé. Les tweets haute priorité (marqués en rouge) sont automatiquement remontés. Chaque tweet porte son sentiment FinBERT et les entités macro extraites.

**Corpus** — les 14 rapports de recherche parsés, avec vue d'ensemble par taille et ton, et liens vers les tweets qui traitent du même sujet (cross-linking sémantique par embeddings).

**Insights** — l'analyse en quatre volets : thèmes émergents (vélocité de mention), consensus (tweets et recherche s'accordent), divergences (désaccord marché vs sell-side) et signaux faibles (mentions rares, ruptures de narrative).

**Synthesis** — un digest quotidien généré par LLM (Groq Llama 3.3 70B, avec Claude en fallback) et un Q&A ad-hoc sur les données.

## Lancer le projet

```bash
# Créer l'environnement
python -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le modèle spaCy
python -m spacy download en_core_web_sm

# Configurer les clés API (optionnel — le dashboard tourne sans LLM)
cp .env.example .env
# Éditer .env et ajouter GROQ_API_KEY

# Lancer le pipeline de traitement (une seule fois)
python src/pipeline.py --fast

# Lancer le dashboard
streamlit run src/dashboard/app.py
```

Le mode `--fast` saute FinBERT et tourne en 1 à 2 minutes. Sans `--fast`, le pipeline complet avec sentiment prend 5 à 10 minutes selon la machine.

## Structure du projet

```
data/
  raw/corpus/       14 PDFs de recherche macro
  raw/tweets/       CSV Financial Juice (497 tweets, Mar 27-31 2026)
  processed/        Résultats du pipeline (JSON, NPY)
src/
  ingestion/        Parsing PDF et chargement tweets
  processing/       Embeddings, BERTopic, NER, FinBERT
  analysis/         Thèmes émergents, consensus, signaux faibles
  synthesis/        Intégration LLM (Groq / Claude)
  dashboard/        Application Streamlit et composants
  pipeline.py       Orchestrateur principal
requirements.txt
```

## Données utilisées

Les tweets couvrent la période du 27 au 31 mars 2026, dans un contexte géopolitique marqué par les tensions autour du détroit d'Ormuz et les incertitudes sur la politique de la Fed. Les 14 rapports proviennent de Goldman Sachs, BofA, Macquarie, Natixis, SEB, Canaccord, DBS, Cavendish et plusieurs auteurs indépendants.

## API LLM

Le dashboard fonctionne entièrement sans LLM — toutes les analyses NLP tournent en local. La page Synthesis nécessite une clé pour le digest automatique.

```
GROQ_API_KEY=gsk_...        gratuit sur console.groq.com
ANTHROPIC_API_KEY=sk-ant-...  alternatif si Groq indisponible
```
