"""
PDF Parser — extrait le texte des rapports macro du corpus.

J'utilise pymupdf (fitz) plutôt que pdfplumber parce que c'est
beaucoup plus rapide sur des PDFs lourds avec beaucoup d'images,
et on récupère les métadonnées proprement.
"""
from __future__ import annotations


import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import fitz  # pymupdf


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Nettoie le texte brut extrait du PDF : espaces en trop, lignes vides, etc."""
    # Fusionne les coupures de lignes qui ne sont pas des fins de paragraphes
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", raw)
    # Écrase les espaces multiples
    text = re.sub(r" {2,}", " ", text)
    # Garde max 2 sauts de ligne consécutifs (séparateurs de sections)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Essaie de deviner la date à partir du nom de fichier.
    Les PDFs du corpus ont souvent le format ...-DD_Mon.pdf
    Exemples : BofA-The_Flow_Show-27_Mar.pdf → 2026-03-27
    """
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    # Pattern : DD_Mon ou DD_Month en fin de nom
    match = re.search(r"(\d{1,2})_([A-Za-z]{3,})", filename)
    if match:
        day = int(match.group(1))
        mon_str = match.group(2)[:3].lower()
        month = months.get(mon_str)
        if month:
            # On suppose l'année courante — les docs sont récents
            return f"2026-{month:02d}-{day:02d}"
    return None


def _extract_source_from_filename(filename: str) -> str:
    """Récupère le nom de l'institution/auteur depuis le nom de fichier."""
    # Format habituel : Institution-Titre-Date.pdf
    parts = filename.replace(".pdf", "").split("-")
    if parts:
        # Remplace les underscores par des espaces pour l'affichage
        return parts[0].replace("_", " ")
    return "Unknown"


# ─── Parser principal ──────────────────────────────────────────────────────────

def parse_pdf(pdf_path: Path) -> dict:
    """
    Parse un PDF et retourne un dict structuré avec :
    - texte complet + sections découpées
    - métadonnées (source, date, titre, nb pages)
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    full_text_parts = []
    sections = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        if page_text.strip():
            full_text_parts.append(page_text)
            sections.append({
                "page": page_num + 1,
                "text": _clean_text(page_text),
                "char_count": len(page_text)
            })

    doc.close()

    full_text = _clean_text("\n\n".join(full_text_parts))
    filename = pdf_path.name

    return {
        "id": pdf_path.stem,
        "filename": filename,
        "source": _extract_source_from_filename(filename),
        "date": _extract_date_from_filename(filename),
        "title": pdf_path.stem.replace("_", " ").replace("-", " — "),
        "full_text": full_text,
        "sections": sections,
        "page_count": len(sections),
        "word_count": len(full_text.split()),
        "parsed_at": datetime.now().isoformat(),
        # Premier paragraphe non-vide comme résumé de tête
        "preview": full_text[:500].strip() + "..." if len(full_text) > 500 else full_text
    }


def parse_corpus(corpus_dir: Path, output_path: Optional[Path] = None) -> list[dict]:
    """
    Parse tous les PDFs d'un répertoire et sauvegarde le résultat en JSON.
    Retourne la liste des documents parsés.
    """
    corpus_dir = Path(corpus_dir)
    pdf_files = sorted(corpus_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {corpus_dir}")

    print(f"  → {len(pdf_files)} PDFs à parser...")

    documents = []
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            doc = parse_pdf(pdf_path)
            documents.append(doc)
            print(f"  [{i:2d}/{len(pdf_files)}] ✓ {pdf_path.name} "
                  f"({doc['page_count']} pages, {doc['word_count']:,} mots)")
        except Exception as e:
            # On ne plante pas sur un PDF corrompu — on log et on continue
            print(f"  [{i:2d}/{len(pdf_files)}] ✗ {pdf_path.name} — Erreur: {e}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        print(f"\n  Corpus sauvegardé → {output_path}")

    print(f"\n  Total : {len(documents)} documents, "
          f"{sum(d['word_count'] for d in documents):,} mots au total")

    return documents


# ─── Point d'entrée standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parents[2]
    corpus_dir = BASE / "data" / "raw" / "corpus"
    output_path = BASE / "data" / "processed" / "corpus.json"

    print("=== Parsing du corpus PDF ===\n")
    docs = parse_corpus(corpus_dir, output_path)
    print(f"\nFait ! {len(docs)} docs dans {output_path}")
