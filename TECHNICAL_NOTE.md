# Note technique — JCAP Macro Intelligence

## Architecture générale

Le système est découpé en deux temps distincts : un pipeline de traitement qui tourne une seule fois et produit des fichiers JSON en sortie, et un dashboard Streamlit qui lit ces fichiers à la demande. Cette séparation est intentionnelle — les étapes lourdes (embeddings, BERTopic, FinBERT) ne sont pas refaites à chaque interaction, ce qui rend le dashboard fluide même sur une machine modeste.

```
Ingestion → Embeddings → NLP → Analyse → Dashboard
```

Tous les résultats intermédiaires sont mis en cache dans `data/processed/`. Le dashboard utilise `st.cache_data` avec un TTL de 1h pour éviter les rechargements inutiles entre les pages.

---

## Choix techniques expliqués

### Parsing PDF : PyMuPDF plutôt que pdfplumber

PyMuPDF est environ trois fois plus rapide que pdfplumber sur des PDFs financiers denses, et gère nettement mieux les layouts complexes : colonnes multiples, tableaux imbriqués, notes de bas de page. Sur 14 rapports avec des structures très différentes (Goldman Sachs n'a pas le même format qu'un auteur indépendant), c'est ce qui fait la différence entre un parsing propre et du bruit dans le texte.

### Embeddings : all-MiniLM-L6-v2

Ce modèle de 22 millions de paramètres offre le meilleur ratio qualité/vitesse pour l'anglais financier. `all-mpnet-base-v2` donne de légèrement meilleurs résultats mais est trois fois plus lent — pas justifié ici. Les embeddings sont normalisés L2 à l'encodage, ce qui rend la similarité cosinus équivalente à un simple produit scalaire et accélère le calcul matriciel lors du cross-linking.

### Topic modeling : BERTopic plutôt que LDA

LDA suppose des sujets distribués selon Dirichlet sur des textes longs, ce qui ne colle pas avec des tweets de 140 caractères. BERTopic utilise les embeddings existants, les regroupe avec HDBSCAN et génère des labels par extraction de mots-clés avec KeyBERT. Le résultat est plus lisible, plus robuste au bruit et il réutilise les embeddings déjà calculés — pas de recalcul.

`min_topic_size=4` est choisi pour s'adapter au volume modeste du corpus (497 tweets sur 4 jours) sans fragmenter les topics.

### NER : hybride spaCy + règles métier

spaCy `en_core_web_sm` reconnaît bien les pays (GPE), les personnes et les organisations génériques. Mais pour les entités spécifiquement macro — Fed, BCE, CPI, JOLTS, PCE, QT — il est peu fiable car ces abréviations sont rares dans son corpus d'entraînement. Le système de règles regex maison prend le relais sur six catégories : banques centrales, personnalités, indicateurs, assets, actions sur les taux, géopolitique. Cette approche est plus transparente et plus facile à maintenir qu'un modèle fine-tuné.

### Sentiment : FinBERT plutôt que VADER

VADER a été conçu pour les tweets généraux. FinBERT a été fine-tuné sur Financial PhraseBank, un corpus de textes financiers annotés par des analystes. La différence est concrète : "hawkish" est neutre pour VADER mais négatif pour les obligations selon FinBERT, "beat expectations" est positif même formulé négativement. Sur notre corpus, c'est une différence qui compte.

### Détection de thèmes émergents : vélocité

La logique est simple : comparer la fréquence normalisée d'une entité dans la première moitié de la période vs la seconde. Une vélocité > 1.5 avec au moins deux mentions récentes est considérée comme un signal émergent. C'est délibérément simple — sur un corpus de 4 jours, les modèles plus sophistiqués surinterprètent le bruit. L'avantage de cette approche est qu'elle est totalement interprétable.

### Cross-linking tweets-corpus : similarité sémantique

Pour chaque tweet, on calcule sa similarité cosinus avec chaque paragraphe du corpus en utilisant les embeddings déjà calculés. Un seuil de 0.45 filtre les faux positifs — un score inférieur n'apporte pas d'information utile. L'index inversé (corpus → tweets pertinents) permet d'afficher dans chaque fiche de rapport les tweets qui traitent du même sujet.

### LLM pour la synthèse : Groq avec Llama 3.3 70B

Groq est choisi pour sa latence très faible — le streaming est quasi-instantané, ce qui rend l'expérience utilisateur nettement meilleure qu'une API classique. Llama 3.3 70B produit des synthèses structurées de qualité sur ce type de tâche. Claude (Anthropic) est maintenu comme fallback. Les deux sont optionnels : toutes les analyses NLP tournent en local.

---

## Limites connues

Le corpus est modeste : 4 jours de tweets et 14 documents. BERTopic peut produire peu de topics distincts, et les signaux faibles sont à interpréter avec prudence sur un si petit échantillon.

Le pipeline est entièrement anglais. Les quelques tweets en français ou espagnol dans le flux Financial Juice dégradent les résultats du sentiment et du topic modeling.

FinBERT est moins fiable sur les tweets très courts (moins de 5 mots) ou avec des abréviations inhabituelles non couvertes par son corpus d'entraînement.

---

## Pistes d'amélioration

Le passage en temps réel serait la priorité : un WebSocket Twitter alimentant les embeddings et les analyses incrémentalement, sans relancer le pipeline complet. Pour la scalabilité sur des volumes plus grands, remplacer la recherche numpy par FAISS ou ChromaDB. L'enrichissement du corpus avec Bloomberg ou Refinitiv permettrait un cross-linking plus dense. Un fine-tuning de FinBERT sur les tweets Financial Juice spécifiquement améliorerait la précision du sentiment sur ce type de langage financier très condensé.
