"""
JCAP Macro Intelligence Dashboard — v2.
Design dark finance avec animations, glassmorphism et meilleure accessibilité.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

from src.dashboard.components.data_loader import (
    load_tweets, load_corpus, load_emerging_themes,
    load_consensus_report, load_weak_signals,
    load_cross_links, load_thematic_bridges,
    check_data_status,
)
from src.dashboard.visualizations import (
    sentiment_timeline, entity_frequency_bars, activity_heatmap,
    sentiment_donut, velocity_bubble_chart, consensus_scatter,
    corpus_overview_treemap, weak_signals_gauge,
    COLORS, SENTIMENT_COLORS,
)

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="JCAP Macro Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0e1117; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4a5568; }

/* ── App background ── */
.stApp { background: #080c14; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
    border-right: 1px solid #1e2d40;
}
section[data-testid="stSidebar"] .block-container { padding: 0 1rem 1rem; }

/* ── Metrics ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s;
}
div[data-testid="metric-container"]:hover {
    border-color: #00d4ff40;
    transform: translateY(-1px);
}
div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    opacity: 0.6;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 12px !important; font-weight: 500 !important; letter-spacing: 0.05em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #f0f4ff !important; font-size: 26px !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Expanders ── */
div[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1e2d40 !important;
    border-radius: 10px !important;
    overflow: hidden;
    transition: border-color 0.2s;
}
div[data-testid="stExpander"]:hover { border-color: #2d4a6a !important; }
div[data-testid="stExpander"] summary { padding: 0.75rem 1rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff12, #7c3aed12);
    border: 1px solid #00d4ff40;
    color: #00d4ff;
    border-radius: 8px;
    font-weight: 500;
    font-size: 13px;
    padding: 0.45rem 1rem;
    transition: all 0.2s;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff20, #7c3aed20);
    border-color: #00d4ff80;
    box-shadow: 0 0 16px #00d4ff20;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00d4ff, #0ea5e9);
    border: none;
    color: #080c14;
    font-weight: 600;
    box-shadow: 0 4px 15px #00d4ff30;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px #00d4ff50;
    transform: translateY(-2px);
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #111827 !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 8px !important;
    color: #e0e0e0 !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input:focus { border-color: #00d4ff60 !important; box-shadow: 0 0 0 2px #00d4ff15 !important; }

/* ── Tabs ── */
div[data-testid="stTabs"] > div:first-child { border-bottom: 1px solid #1e2d40; gap: 4px; }
button[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: #6b7280 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 6px 6px 0 0 !important;
    transition: all 0.2s !important;
}
button[data-baseweb="tab"]:hover { color: #cbd5e1 !important; background: #ffffff08 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #00d4ff !important;
    background: #00d4ff0d !important;
    border-bottom: 2px solid #00d4ff !important;
}
div[data-testid="stTabsContent"] { padding-top: 1.2rem; }

/* ── Dividers ── */
hr { border: none !important; border-top: 1px solid #1e2d40 !important; margin: 1rem 0; }

/* ── Info / warning / success ── */
div[data-testid="stAlert"] { border-radius: 8px !important; border: none !important; font-size: 13px; }

/* ── Radio (nav) ── */
div[data-testid="stRadio"] > div { gap: 4px; }
div[data-testid="stRadio"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    transition: all 0.2s;
    cursor: pointer;
    font-size: 14px !important;
    color: #9ca3af !important;
    width: 100%;
}
div[data-testid="stRadio"] label:hover { background: #ffffff08; border-color: #1e2d40; color: #e0e0e0 !important; }
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] input:checked + div {
    background: linear-gradient(135deg, #00d4ff12, #7c3aed12) !important;
    border-color: #00d4ff30 !important;
    color: #00d4ff !important;
}

/* ── Animations ── */
@keyframes pulse-dot { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
@keyframes fade-in { from { opacity:0; transform: translateY(8px); } to { opacity:1; transform: translateY(0); } }
@keyframes glow-border { 0%,100% { box-shadow: 0 0 0 0 #ff4b4b00; } 50% { box-shadow: 0 0 0 3px #ff4b4b20; } }
@keyframes slide-in { from { opacity:0; transform: translateX(-6px); } to { opacity:1; transform: translateX(0); } }

/* ── Custom components ── */
.page-header {
    display: flex; align-items: center; gap: 12px;
    padding: 0 0 1rem 0;
    border-bottom: 1px solid #1e2d40;
    margin-bottom: 1.5rem;
    animation: fade-in 0.4s ease;
}
.page-header-icon {
    font-size: 28px;
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #00d4ff15, #7c3aed15);
    border: 1px solid #00d4ff30;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
}
.page-header-title { font-size: 20px; font-weight: 700; color: #f0f4ff; line-height: 1.2; }
.page-header-sub { font-size: 12px; color: #6b7280; margin-top: 2px; }

/* ── Tweet card ── */
.tweet-card {
    background: linear-gradient(135deg, #111827, #131d2d);
    border: 1px solid #1e2d40;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 7px;
    transition: all 0.2s;
    animation: slide-in 0.3s ease;
}
.tweet-card:hover { border-color: #00d4ff30; transform: translateX(2px); box-shadow: 0 2px 12px #00000040; }
.tweet-card-high { border-left: 3px solid #ff4b4b; animation: slide-in 0.3s ease, glow-border 2s ease infinite; }
.tweet-card-medium { border-left: 3px solid #fbbf24; }
.tweet-card-normal { border-left: 3px solid #374151; }
.tweet-text { font-size: 13.5px; color: #d1d5db; line-height: 1.55; }
.tweet-meta { font-size: 11px; color: #4b5563; margin-top: 7px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.tweet-time { font-family: 'JetBrains Mono', monospace; color: #374151; font-size: 10.5px; }

/* ── Macro tags ── */
.macro-tag {
    display: inline-block; padding: 2px 7px;
    border-radius: 10px; font-size: 10.5px; font-weight: 500; letter-spacing: 0.02em;
    margin: 1px;
}
.tag-high    { background:#ff4b4b18; border:1px solid #ff4b4b50; color:#ff6b6b; }
.tag-medium  { background:#fbbf2418; border:1px solid #fbbf2450; color:#fcd34d; }
.tag-normal  { background:#60a5fa18; border:1px solid #60a5fa40; color:#93c5fd; }
.tag-low     { background:#37414118; border:1px solid #37414150; color:#6b7280; }
.tag-positive{ background:#22c55e18; border:1px solid #22c55e50; color:#4ade80; }
.tag-negative{ background:#ef444418; border:1px solid #ef444450; color:#f87171; }
.tag-neutral { background:#37414118; border:1px solid #37414150; color:#9ca3af; }
.tag-emerging{ background:#ffd70018; border:1px solid #ffd70050; color:#fde047; }
.tag-entity  { background:#7c3aed18; border:1px solid #7c3aed50; color:#c4b5fd; }

/* ── Stat card (signal / insight card) ── */
.stat-card {
    background: linear-gradient(135deg, #111827, #131d2d);
    border: 1px solid #1e2d40;
    border-radius: 10px;
    padding: 13px 15px;
    margin-bottom: 8px;
    transition: all 0.2s;
}
.stat-card:hover { border-color: #2d4a6a; box-shadow: 0 4px 16px #00000030; }
.stat-card-green  { border-left: 3px solid #22c55e; }
.stat-card-yellow { border-left: 3px solid #fbbf24; }
.stat-card-red    { border-left: 3px solid #ff4b4b; }
.stat-card-orange { border-left: 3px solid #f97316; }
.stat-card-gold   { border-left: 3px solid #ffd700; }
.stat-card-blue   { border-left: 3px solid #60a5fa; }
.stat-card-title  { font-size: 14px; font-weight: 600; color: #e2e8f0; }
.stat-card-sub    { font-size: 11.5px; color: #6b7280; margin-top: 5px; line-height: 1.5; }

/* ── Section label ── */
.section-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #4b5563;
    margin-bottom: 10px; margin-top: 4px;
}

/* ── Progress bar custom ── */
.prog-bar-bg {
    height: 5px; background: #1e2d40; border-radius: 3px; margin-top: 5px; overflow: hidden;
}
.prog-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }

/* ── Status dots ── */
.dot-ok   { display:inline-block; width:7px; height:7px; border-radius:50%; background:#22c55e; margin-right:7px; animation: pulse-dot 2s infinite; }
.dot-warn { display:inline-block; width:7px; height:7px; border-radius:50%; background:#fbbf24; margin-right:7px; animation: pulse-dot 2s infinite 0.5s; }
.dot-err  { display:inline-block; width:7px; height:7px; border-radius:50%; background:#ff4b4b; margin-right:7px; }

/* ── Filter bar ── */
.filter-bar {
    background: #111827; border: 1px solid #1e2d40; border-radius: 10px;
    padding: 12px 16px; margin-bottom: 16px;
}

/* ── KPI strip ── */
.kpi-strip { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:1.2rem; }
.kpi-box {
    flex: 1; min-width: 120px;
    background: linear-gradient(135deg, #111827, #141e2e);
    border: 1px solid #1e2d40; border-radius: 10px; padding: 14px 16px;
    position: relative; overflow: hidden;
    transition: all 0.2s;
}
.kpi-box:hover { border-color: #00d4ff30; transform: translateY(-1px); }
.kpi-box::after { content:''; position:absolute; inset:0; background:linear-gradient(135deg,#00d4ff05,transparent); pointer-events:none; }
.kpi-label { font-size: 10.5px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; color:#4b5563; margin-bottom:6px; }
.kpi-value { font-size: 24px; font-weight: 700; color: #f0f4ff; font-family:'JetBrains Mono',monospace; }
.kpi-sub   { font-size: 11px; color: #6b7280; margin-top: 4px; }

/* ── Preview block ── */
.preview-block {
    background: #0d1117; border-left: 3px solid #1e2d40;
    border-radius: 0 6px 6px 0; padding: 10px 14px;
    font-size: 12.5px; color: #6b7280; line-height: 1.65;
    font-family: 'JetBrains Mono', monospace;
    white-space: pre-wrap; word-break: break-word;
}

/* ── Velocity rank item ── */
.velocity-item {
    display:flex; align-items:center; justify-content:space-between;
    padding: 9px 12px; background: linear-gradient(135deg,#111827,#131d2d);
    border: 1px solid #1e2d40; border-radius: 8px; margin-bottom: 5px;
    transition: all 0.15s;
}
.velocity-item:hover { border-color: #2d4a6a; padding-left: 15px; }
.velocity-badge {
    font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:600;
    padding: 2px 8px; border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def tag(label: str, kind: str = "normal") -> str:
    return f'<span class="macro-tag tag-{kind}">{label}</span>'

def sentiment_badge(s: str) -> str:
    icons = {"positive": "▲", "negative": "▼", "neutral": "—"}
    return tag(f'{icons.get(s,"●")} {s.capitalize()}', s)

def entity_tags(entities: list, max_n: int = 4) -> str:
    shown = entities[:max_n]
    rest  = len(entities) - max_n
    out = "".join(tag(e, "entity") for e in shown)
    if rest > 0:
        out += tag(f"+{rest}", "neutral")
    return out

def progress_bar(pct: float, color: str = "#00d4ff") -> str:
    return (
        f'<div class="prog-bar-bg">'
        f'<div class="prog-bar-fill" style="width:{min(pct,100):.1f}%;background:{color};"></div>'
        f'</div>'
    )

def page_header(icon: str, title: str, subtitle: str = "") -> None:
    sub_html = f'<div class="page-header-sub">{subtitle}</div>' if subtitle else ""
    icon_html = f'<div class="page-header-icon">{icon}</div>' if icon else ""
    st.markdown(
        f'<div class="page-header">'
        f'{icon_html}'
        f'<div><div class="page-header-title">{title}</div>{sub_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def kpi(label: str, value: str, sub: str = "", accent: str = "#00d4ff") -> str:
    return (
        f'<div class="kpi-box" style="border-top:2px solid {accent}30;">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{accent};">{value}</div>'
        f'{"<div class=kpi-sub>" + sub + "</div>" if sub else ""}'
        f'</div>'
    )

def chart_caption(text: str, source: str = "") -> None:
    """Légende explicative sous un graphe avec optionnellement la source des données."""
    src_html = (
        f'<div style="margin-top:5px;font-size:10.5px;color:#374151;font-style:italic;">'
        f' Source : {source}</div>'
    ) if source else ""
    st.markdown(
        f'<div style="font-size:11.5px;color:#4b5563;line-height:1.6;'
        f'padding:6px 14px 12px;border-left:2px solid #1e2d40;margin-top:-6px;margin-bottom:4px;">'
        f'{text}{src_html}</div>',
        unsafe_allow_html=True,
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:20px 8px 16px;text-align:center;">
            
            <div style="font-size:17px;font-weight:700;color:#f0f4ff;letter-spacing:0.05em;">JCAP</div>
            <div style="font-size:11px;color:#4b5563;letter-spacing:0.12em;text-transform:uppercase;margin-top:3px;">
                Macro Intelligence
            </div>
            <div style="margin-top:10px;height:1px;background:linear-gradient(90deg,transparent,#1e2d40,transparent);"></div>
        </div>
        """, unsafe_allow_html=True)

        status = check_data_status()
        st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)
        labels = {"tweets": "Tweet Feed", "corpus": "Corpus", "embeddings": "Embeddings",
                  "analysis": "Analysis", "pipeline_run": "Pipeline"}
        for key, ok in status.items():
            dot = "dot-ok" if ok else "dot-warn"
            label = labels.get(key, key.replace("_", " ").title())
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:4px 0;">'
                f'<span class="{dot}"></span>'
                f'<span style="font-size:12px;color:{"#9ca3af" if ok else "#6b7280"};">{label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if not status["pipeline_run"]:
            st.warning("Run pipeline first:\n```\npython src/pipeline.py --fast\n```")

        st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#1e2d40,transparent);margin:12px 0;"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "nav",
            ["Feed", "Corpus", "Insights", "Synthesis"],
            label_visibility="collapsed",
        )

        st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#1e2d40,transparent);margin:12px 0;"></div>', unsafe_allow_html=True)

        # Stats rapides
        if status["tweets"]:
            try:
                df_s = load_tweets()
                if not df_s.empty:
                    dates = df_s["date_str"].unique() if "date_str" in df_s.columns else []
                    period = f"{min(dates)} → {max(dates)}" if len(dates) > 1 else (dates[0] if len(dates) else "—")
                    st.markdown(
                        f'<div style="font-size:11px;color:#374151;line-height:1.8;">'
                        f'<div style="color:#4b5563;">{period}</div>'
                        f'<div style="color:#4b5563;">{len(df_s):,} tweets</div>'
                        f'<div style="color:#4b5563;">14 research reports</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

        st.markdown(
            '<div style="font-size:10px;color:#1f2937;text-align:center;margin-top:16px;letter-spacing:0.05em;">'
            'JCAP Test Technique · 2026'
            '</div>',
            unsafe_allow_html=True,
        )

    return page.strip()


# ─── Page Feed ────────────────────────────────────────────────────────────────

def render_feed_page(df: pd.DataFrame):
    page_header("", "Tweet Feed", "Financial Juice · Real-time macro flow")

    if df.empty:
        st.info("No tweets loaded. Run: `python src/pipeline.py --fast`")
        return

    # KPIs
    hp    = len(df[df["priority"] == "high"]) if "priority" in df.columns else 0
    days  = df["date_str"].nunique() if "date_str" in df.columns else 0
    pos_p = int((df["sentiment"] == "positive").mean() * 100) if "sentiment" in df.columns else 0
    neg_p = int((df["sentiment"] == "negative").mean() * 100) if "sentiment" in df.columns else 0
    kw_n  = df["keyword_count"].sum() if "keyword_count" in df.columns else 0

    st.markdown(
        '<div class="kpi-strip">'
        + kpi("Total Tweets", f"{len(df):,}", f"{days} days", "#00d4ff")
        + kpi("Breaking", str(hp), "high priority", "#ff4b4b")
        + kpi("Bullish", f"{pos_p}%", "positive sentiment", "#22c55e")
        + kpi("Bearish", f"{neg_p}%", "negative sentiment", "#ef4444")
        + kpi("Macro Hits", f"{int(kw_n):,}", "keyword matches", "#a78bfa")
        + '</div>',
        unsafe_allow_html=True,
    )

    # Charts
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(sentiment_timeline(df), use_container_width=True)
        chart_caption(
            "Barres = volume de tweets par heure, colorées selon le sentiment dominant (vert haussier, rouge baissier, bleu neutre). "
            "Ligne cyan = score de sentiment lissé (moyenne mobile 4 h). "
            "Un score > 0 reflète une narrative de marché positive ; < 0 indique une pression baissière. "
            "La bande grisée (±0.12) représente la zone neutre sans signal directionnel.",
            source="Financial Juice — flux Twitter en temps réel · Sentiment calculé par FinBERT (modèle NLP entraîné sur textes financiers)"
        )
    with col2:
        st.plotly_chart(sentiment_donut(df), use_container_width=True)
        chart_caption(
            "Répartition du sentiment sur l'ensemble des tweets de la période. "
            "▲ Positive = ton haussier dominant · ▼ Negative = pression baissière · — Neutral = informatif/factuel. "
            "Un déséquilibre fort (ex. >60 % négatif) signal un stress de marché.",
            source="Financial Juice tweets · FinBERT sentiment"
        )

    st.plotly_chart(activity_heatmap(df), use_container_width=True)
    chart_caption(
        "Carte de chaleur croisant les heures UTC (axe Y) et les jours (axe X). "
        "Les zones les plus lumineuses (cyan intense) indiquent les pics d'activité. "
        "Interpréter en regard du calendrier macro : ouvertures de marchés (08h–09h UTC), "
        "données US (12h30–14h UTC : NFP, CPI, FOMC), déclarations de banquiers centraux.",
        source="Financial Juice tweets · horodatage UTC extrait à l'ingestion"
    )

    st.markdown("---")

    # Filter bar
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 2])
    with fc1:
        all_dates = sorted(df["date_str"].unique()) if "date_str" in df.columns else []
        sel_dates = st.multiselect(" Date", all_dates, default=all_dates, label_visibility="collapsed",
                                    placeholder="Filter by date…")
    with fc2:
        prio_f = st.selectbox(" Priority", ["all", "high", "medium", "normal", "low"],
                               label_visibility="collapsed")
    with fc3:
        sent_f = st.selectbox(" Sentiment", ["all", "positive", "negative", "neutral"],
                               label_visibility="collapsed")
    with fc4:
        search = st.text_input(" Search", placeholder="Fed, CPI, oil, Hormuz…",
                                label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Apply filters
    mask = pd.Series([True] * len(df), index=df.index)
    if sel_dates and "date_str" in df.columns:
        mask &= df["date_str"].isin(sel_dates)
    if prio_f != "all" and "priority" in df.columns:
        mask &= df["priority"] == prio_f
    if sent_f != "all" and "sentiment" in df.columns:
        mask &= df["sentiment"] == sent_f
    if search and "text_clean" in df.columns:
        mask &= df["text_clean"].str.contains(search, case=False, na=False)

    df_f = df[mask].sort_values("datetime", ascending=False) if "datetime" in df.columns else df[mask]

    n_match = len(df_f)
    st.markdown(
        f'<div style="font-size:12px;color:#6b7280;margin-bottom:10px;">'
        f'Showing <strong style="color:#00d4ff;">{n_match:,}</strong> tweets'
        + (f' · filtered from {len(df):,}' if n_match < len(df) else '')
        + '</div>',
        unsafe_allow_html=True,
    )

    # Tweet list
    for _, row in df_f.head(80).iterrows():
        prio     = row.get("priority", "normal")
        sent     = row.get("sentiment", "neutral")
        text     = row.get("text_clean", row.get("content", ""))
        dt_raw   = str(row.get("datetime", ""))
        dt       = dt_raw[:16].replace("T", " ").replace("+00:00", "")
        score    = row.get("sentiment_score", 0)
        score_s  = f"{score:+.2f}" if isinstance(score, float) else ""
        ents     = entity_tags(
            (row.get("central_banks") or []) + (row.get("indicators") or [])
        )
        card_cls  = f"tweet-card-{prio}" if prio in ("high", "medium") else "tweet-card-normal"
        prio_tag  = tag(prio.upper(), prio)
        sent_tag  = sentiment_badge(sent)
        score_el  = (f'<span style="font-family:JetBrains Mono,monospace;font-size:10.5px;'
                     f'color:{"#4ade80" if score > 0.1 else "#f87171" if score < -0.1 else "#6b7280"};">'
                     f'({score_s})</span>') if score_s else ""

        st.markdown(
            f'<div class="tweet-card {card_cls}">'
            f'<div class="tweet-text">{text}</div>'
            f'<div class="tweet-meta">'
            f'<span class="tweet-time">{dt} UTC</span>'
            f'<span style="color:#1e2d40;">·</span>'
            f'{prio_tag}{sent_tag}{score_el}'
            f'{"<span style=color:#1e2d40;>·</span>" + ents if ents else ""}'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ─── Page Corpus ──────────────────────────────────────────────────────────────

def render_corpus_page(documents: list, cross_ref: dict):
    page_header("", "Research Corpus", f"{len(documents)} reports · macro research & strategy")

    if not documents:
        st.info("No corpus loaded. Run pipeline first.")
        return

    st.plotly_chart(corpus_overview_treemap(documents), use_container_width=True)
    chart_caption(
        "Carte de la couverture du corpus de recherche. "
        "Chaque rectangle = un rapport, sa taille est proportionnelle au nombre de mots (profondeur d'analyse). "
        "<span style='color:#4ade80;'>Vert</span> = sentiment positif (bullish) · "
        "<span style='color:#f87171;'>Rouge</span> = négatif (bearish/risk-off) · "
        "<span style='color:#60a5fa;'>Bleu</span> = neutre/factuel. "
        "Cliquer sur une source pour zoomer. Survoler pour voir le titre et le nombre de mots.",
        source="14 rapports PDF parsés (Goldman Sachs, BofA, Macquarie, Natixis, SEB, Cavendish, DBS, Canaccord, Alexander Campbell, Michael Howell, Richard Bexelius…) · Sentiment calculé par FinBERT sur le texte intégral"
    )

    st.markdown("---")

    col_l, col_r = st.columns([5, 2])

    with col_l:
        st.markdown('<div class="section-label">Research Reports</div>', unsafe_allow_html=True)

        for doc in documents:
            src    = doc.get("source", "Unknown")
            title  = doc.get("title", doc.get("id", ""))
            date   = doc.get("date", "N/A")
            words  = doc.get("word_count", 0)
            pages  = doc.get("page_count", 0)
            sent   = doc.get("sentiment", "neutral")
            score  = doc.get("sentiment_score", 0)
            linked = cross_ref.get(doc.get("id", ""), [])

            sent_color = SENTIMENT_COLORS.get(sent, "#6b7280")
            score_s = f"{score:+.2f}" if isinstance(score, float) else ""

            with st.expander(f"**{src}** — {title[:65]}", expanded=False):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Date", date)
                m2.metric("Pages", pages)
                m3.metric("Words", f"{words:,}")
                m4.metric("Linked Tweets", len(linked))

                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:10px 0 6px;">'
                    f'<span style="font-size:11px;color:#4b5563;text-transform:uppercase;letter-spacing:.08em;">Tone</span>'
                    f'{sentiment_badge(sent)}'
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:{sent_color};">{score_s}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="section-label">Excerpt</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="preview-block">{doc.get("preview","")[:500]}</div>',
                    unsafe_allow_html=True,
                )

                if linked:
                    st.markdown('<div class="section-label" style="margin-top:12px;">Related Tweets</div>',
                                unsafe_allow_html=True)
                    for t in linked[:3]:
                        sim = t.get("similarity", 0)
                        sim_color = "#22c55e" if sim > 0.65 else "#fbbf24" if sim > 0.5 else "#6b7280"
                        st.markdown(
                            f'<div class="tweet-card tweet-card-normal">'
                            f'<div class="tweet-text" style="font-size:12.5px;">{t.get("tweet_text","")[:200]}</div>'
                            f'<div class="tweet-meta">'
                            f'<span class="tweet-time">{t.get("tweet_date","")}</span>'
                            f'<span style="color:#1e2d40;">·</span>'
                            f'<span style="font-family:JetBrains Mono,monospace;font-size:10.5px;color:{sim_color};">'
                            f'sim {sim:.2f}</span></div></div>',
                            unsafe_allow_html=True,
                        )

    with col_r:
        st.markdown('<div class="section-label">Coverage</div>', unsafe_allow_html=True)

        # Source breakdown
        src_counts = Counter(d.get("source", "Unknown") for d in documents)
        max_v = max(src_counts.values(), default=1)
        for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
            pct = cnt / len(documents) * 100
            st.markdown(
                f'<div style="margin-bottom:9px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                f'<span style="font-size:12px;color:#d1d5db;">{src}</span>'
                f'<span style="font-size:11px;color:#4b5563;">{cnt} doc{"s" if cnt>1 else ""}</span>'
                f'</div>'
                + progress_bar(pct, "#00d4ff50") +
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown('<div class="section-label">Corpus Sentiment</div>', unsafe_allow_html=True)

        sent_counts = Counter(d.get("sentiment", "neutral") for d in documents)
        total = len(documents)
        for s, cnt in sent_counts.items():
            c = SENTIMENT_COLORS.get(s, "#6b7280")
            pct = cnt / total * 100
            st.markdown(
                f'<div style="margin-bottom:8px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                f'<span style="color:{c};font-size:12px;">{s.capitalize()}</span>'
                f'<span style="color:#4b5563;font-size:11px;">{cnt}</span>'
                f'</div>'
                + progress_bar(pct, c) +
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown('<div class="section-label">Word Count</div>', unsafe_allow_html=True)
        total_words = sum(d.get("word_count", 0) for d in documents)
        st.markdown(
            f'<div class="stat-card stat-card-blue">'
            f'<div class="stat-card-title">{total_words:,}</div>'
            f'<div class="stat-card-sub">total words across {len(documents)} documents</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─── Page Insights ────────────────────────────────────────────────────────────

def render_insights_page(df: pd.DataFrame, emerging: dict,
                          consensus_report: dict, weak_signals: dict):
    page_header("", "Macro Insights",
                "Emerging themes · consensus · divergences · weak signals")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Emerging Themes",
        "Consensus",
        "Divergences",
        "Weak Signals",
    ])

    # ── Tab 1 : Emerging Themes ───────────────────────────────────
    with tab1:
        st.markdown(
            '<div style="background:#0d1f0d;border:1px solid #22c55e25;border-radius:8px;'
            'padding:9px 14px;font-size:11.5px;color:#4b5563;margin-bottom:14px;">'
            '<strong style="color:#6b7280;">Sources :</strong> '
            'Tweets Financial Juice (Mar 27–31 2026) · Entités extraites par NER hybride (spaCy + 6 dictionnaires macro) · '
            'Vélocité = ratio fréquence_récente / fréquence_ancienne (fenêtre glissante 2 jours)'
            '</div>',
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown('<div class="section-label">Velocity Ranking</div>', unsafe_allow_html=True)
            themes = emerging.get("emerging_themes", [])
            if themes:
                for i, th in enumerate(themes[:10]):
                    v = th.get("velocity", 0)
                    accent = "#ffd700" if v > 3 else "#22c55e" if v > 1.5 else "#6b7280"
                    badge_bg = f"{accent}18"
                    st.markdown(
                        f'<div class="velocity-item">'
                        f'<div style="display:flex;align-items:center;gap:8px;">'
                        f'<span style="font-size:10px;color:#374151;font-family:JetBrains Mono,monospace;">'
                        f'#{i+1:02d}</span>'
                        f'<span style="font-size:13px;color:#e2e8f0;">{th["entity"]}</span>'
                        f'<span style="font-size:10px;color:#6b7280;">{th.get("category","").replace("_"," ")}</span>'
                        f'</div>'
                        f'<span class="velocity-badge" style="background:{badge_bg};color:{accent};">'
                        f'↑ {v:.1f}×</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Run pipeline to compute emerging themes")

            st.markdown("---")
            st.markdown('<div class="section-label">Volume by Day</div>', unsafe_allow_html=True)
            vol = emerging.get("volume_by_day", {})
            if vol:
                max_v = max(vol.values(), default=1)
                for d_str, cnt in sorted(vol.items()):
                    pct = cnt / max_v * 100
                    st.markdown(
                        f'<div style="margin-bottom:8px;">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                        f'<span style="font-size:11.5px;color:#9ca3af;">{d_str}</span>'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:#00d4ff;">{cnt}</span>'
                        f'</div>'
                        + progress_bar(pct, "#00d4ff") +
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with col_b:
            vel_data = emerging.get("velocity_by_category", {})
            if vel_data:
                cat = st.selectbox("Category", list(vel_data.keys()), key="vel_cat")
                if cat:
                    vel_df = pd.DataFrame(vel_data[cat])
                    if not vel_df.empty:
                        st.plotly_chart(velocity_bubble_chart(vel_df, cat),
                                        use_container_width=True)
                        chart_caption(
                            "<strong>Comment lire ce graphe :</strong> "
                            "Axe X = fréquence de mention en début de période, axe Y = fréquence récente. "
                            "Les bulles <strong>au-dessus</strong> de la diagonale pointillée sont en <span style='color:#fde047;'>accélération (emerging)</span> — "
                            "le marché en parle de plus en plus. En dessous = sujet en déclin. "
                            "Taille = volume récent absolu · Couleur (jaune intense) = ratio de vitesse élevé.",
                            source="Financial Juice tweets · NER hybride (spaCy + règles regex) pour détecter les entités macro · Calcul de vélocité : freq_récente / (freq_ancienne + ε) sur fenêtre glissante"
                        )
                        counts = {r["entity"]: r.get("count_recent", 0)
                                  for _, r in vel_df.iterrows()}
                        st.plotly_chart(entity_frequency_bars({cat: counts}, cat),
                                        use_container_width=True)
                        chart_caption(
                            "Classement des entités par nombre de mentions sur la période récente dans cette catégorie. "
                            "Reflète l'intensité narrative — une entité très mentionnée domine le sentiment de marché. "
                            "Comparer avec la période précédente via le graphe de vélocité ci-dessus.",
                            source="Financial Juice tweets · NER extraction (banques centrales, indicateurs, assets, personnalités macro)"
                        )

    # ── Tab 2 : Consensus ────────────────────────────────────────
    with tab2:
        st.markdown(
            '<div style="background:#0d1a2e;border:1px solid #3b82f625;border-radius:8px;'
            'padding:9px 14px;font-size:11.5px;color:#4b5563;margin-bottom:14px;">'
            '<strong style="color:#6b7280;">Sources :</strong> '
            'Croisement de 497 tweets (FinBERT) × 14 rapports de recherche PDF · '
            'Entités communes identifiées par NER · Consensus = Δ sentiment tweets/corpus < 0.3 · '
            'Divergence = Δ > 0.3 (désaccord marché vs analystes sell-side)'
            '</div>',
            unsafe_allow_html=True,
        )
        cons_items = consensus_report.get("consensus", [])
        if not cons_items:
            st.info("Consensus analysis requires the full pipeline with sentiment.")
        else:
            st.markdown(
                f'<div style="font-size:12.5px;color:#6b7280;margin-bottom:1rem;">'
                f'<strong style="color:#22c55e;">{len(cons_items)}</strong> areas where '
                f'tweets and research corpus agree · '
                f'<strong style="color:#f97316;">{len(consensus_report.get("divergences",[]))}</strong> divergences</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(consensus_scatter(consensus_report), use_container_width=True)
            chart_caption(
                "<strong>Comment lire ce graphe :</strong> "
                "Chaque point = une entité macro (banque centrale, pays, asset). "
                "Axe X = score de sentiment moyen dans les tweets · Axe Y = score dans les rapports de recherche. "
                "<span style='color:#4ade80;'>Points verts (proche de la diagonale)</span> = consensus : le marché et les analystes sont alignés. "
                "<span style='color:#fb923c;'>Points orange (loin de la diagonale)</span> = divergence : signal d'alpha potentiel — "
                "le marché price différemment de ce que les analystes écrivent.",
                source="Tweets Financial Juice (FinBERT) croisés avec 14 rapports PDF (Goldman Sachs, BofA, Macquarie, Natixis, SEB, Cavendish, DBS, Canaccord…) · Matching par entité NER commune"
            )

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown('<div class="section-label">Consensus Points</div>', unsafe_allow_html=True)
                for item in cons_items[:8]:
                    t_s = item.get("tweet_sentiment", "?")
                    c_s = item.get("corpus_sentiment", "?")
                    n   = item.get("tweet_count", 0)
                    tc  = SENTIMENT_COLORS.get(t_s, "#6b7280")
                    cc  = SENTIMENT_COLORS.get(c_s, "#6b7280")
                    st.markdown(
                        f'<div class="stat-card stat-card-green">'
                        f'<div class="stat-card-title">{item["entity"]}</div>'
                        f'<div class="stat-card-sub">'
                        f' <span style="color:{tc};">{t_s}</span> &nbsp;·&nbsp; '
                        f' <span style="color:{cc};">{c_s}</span> &nbsp;·&nbsp; '
                        f'{n} mentions</div></div>',
                        unsafe_allow_html=True,
                    )

            with col_c2:
                st.markdown('<div class="section-label">Contested Topics</div>', unsafe_allow_html=True)
                contested = consensus_report.get("contested_in_tweets", [])
                if contested:
                    for item in contested[:8]:
                        std = item.get("std_score", 0)
                        n   = item.get("n_tweets", 0)
                        st.markdown(
                            f'<div class="stat-card stat-card-yellow">'
                            f'<div class="stat-card-title">{item["entity"]}</div>'
                            f'<div class="stat-card-sub">'
                            f'Dispersion {std:.2f} &nbsp;·&nbsp; {n} tweets &nbsp;·&nbsp; '
                            f'<span style="color:#fcd34d;">contested</span></div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No strongly contested topics detected")

    # ── Tab 3 : Divergences ──────────────────────────────────────
    with tab3:
        st.markdown(
            '<div style="background:#1a0d00;border:1px solid #f9731625;border-radius:8px;'
            'padding:9px 14px;font-size:11.5px;color:#4b5563;margin-bottom:14px;">'
            '<strong style="color:#6b7280;">Sources :</strong> '
            'Mêmes données que Consensus — seuls les cas avec Δ sentiment > 0.3 sont isolés ici. '
            'Score de divergence = |tweet_score − corpus_score|. '
            'Δ > 0.5 = désaccord fort (rouge) · Δ 0.3–0.5 = désaccord modéré (orange). '
            'Ces divergences représentent des opportunités d\'alpha : le marché et la recherche ne voient pas le même risque.'
            '</div>',
            unsafe_allow_html=True,
        )
        div_items = consensus_report.get("divergences", [])
        if not div_items:
            st.info("No divergences detected — or run with full sentiment pipeline.")
        else:
            st.markdown(
                '<div style="background:#f97316 10;border:1px solid #f9731630;border-radius:8px;'
                'padding:10px 14px;font-size:12.5px;color:#fed7aa;margin-bottom:1rem;">'
                '<strong>Divergences</strong> = where market participants and researchers disagree. '
                'Highest potential for alpha generation.</div>',
                unsafe_allow_html=True,
            )
            for item in div_items:
                dscore  = item.get("divergence_score", 0)
                t_s     = item.get("tweet_sentiment", "?")
                c_s     = item.get("corpus_sentiment", "?")
                tc      = SENTIMENT_COLORS.get(t_s, "#6b7280")
                cc      = SENTIMENT_COLORS.get(c_s, "#6b7280")
                bcolor  = "#ff4b4b" if dscore > 0.5 else "#f97316"
                card_cls = "stat-card-red" if dscore > 0.5 else "stat-card-orange"

                st.markdown(
                    f'<div class="stat-card {card_cls}">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<div class="stat-card-title">{item["entity"]}</div>'
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;color:{bcolor};">'
                    f'Δ {dscore:.2f}</span></div>'
                    f'<div class="stat-card-sub">'
                    f' Twitter: <span style="color:{tc};font-weight:500;">{t_s}</span> &nbsp;·&nbsp; '
                    f' Research: <span style="color:{cc};font-weight:500;">{c_s}</span> &nbsp;·&nbsp; '
                    f'{item.get("tweet_count",0)} tweets, {item.get("corpus_count",0)} refs'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    # ── Tab 4 : Weak Signals ────────────────────────────────────
    with tab4:
        st.markdown(
            '<div style="background:#120d1a;border:1px solid #a78bfa25;border-radius:8px;'
            'padding:9px 14px;font-size:11.5px;color:#4b5563;margin-bottom:14px;">'
            '<strong style="color:#6b7280;">Sources :</strong> '
            'Financial Juice tweets (497 tweets, Mar 27–31 2026) · '
            'Rare Signals = entités avec < 5 mentions totales, pondérées par % de tweets haute priorité () · '
            'Co-occurrences = lift > 2× par rapport à la fréquence attendue au hasard · '
            'Narrative Shifts = fenêtre glissante 3 h, détection de Δ sentiment > 0.4'
            '</div>',
            unsafe_allow_html=True,
        )
        n_rare   = weak_signals.get("n_rare_signals", 0)
        n_cooc   = weak_signals.get("n_unusual_cooc", 0)
        n_shifts = weak_signals.get("n_shifts", 0)

        st.markdown(
            '<div class="kpi-strip">'
            + kpi("Rare Signals", str(n_rare), "low-frequency mentions", "#ffd700")
            + kpi("Unusual Links", str(n_cooc), "co-occurrence lift > 2×", "#a78bfa")
            + kpi("Narrative Shifts", str(n_shifts), "sentiment ruptures", "#f97316")
            + '</div>',
            unsafe_allow_html=True,
        )

        if n_rare > 0 or n_shifts > 0:
            st.plotly_chart(weak_signals_gauge(n_rare, n_shifts), use_container_width=True)
            chart_caption(
                "<strong>Signaux faibles :</strong> indicateurs d'anticipation avant qu'un thème devienne dominant. "
                "<strong style='color:#fde047;'>Rare Signals</strong> = entités mentionnées < 5 fois sur la période — "
                "souvent les premières occurrences d'un risque émergent ou d'un retournement narratif. "
                "<strong style='color:#fb923c;'>Narrative Shifts</strong> = ruptures brutales du sentiment moyen (Δ > 0.4) "
                "sur une fenêtre de 3 heures — signal d'une information surprenante entrant dans le marché. "
                "La ligne rouge = seuil d'alerte (80 % du max).",
                source="Financial Juice tweets · Détection de ruptures par fenêtre glissante sur le score FinBERT · Co-occurrences inhabituelles calculées par coefficient lift (lift > 2×)"
            )

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown('<div class="section-label">Rare Mentions</div>', unsafe_allow_html=True)
            rare = weak_signals.get("rare_mentions", [])
            if rare:
                for sig in rare[:8]:
                    strength = sig.get("signal_strength", 0)
                    n_m      = sig.get("total_mentions", 0)
                    hp_m     = sig.get("high_priority_mentions", 0)
                    sent_v   = sig.get("avg_sentiment", 0)
                    sc       = "#4ade80" if sent_v > 0.1 else ("#f87171" if sent_v < -0.1 else "#6b7280")
                    examples = sig.get("example_tweets", [])
                    ex_html  = (
                        f'<div style="font-size:11px;color:#374151;margin-top:6px;'
                        f'font-style:italic;border-left:2px solid #1e2d40;padding-left:8px;">'
                        + (examples[0] or "")[:120] + "</div>"
                    ) if examples else ""

                    st.markdown(
                        f'<div class="stat-card stat-card-gold">'
                        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
                        f'<div class="stat-card-title">{sig["entity"]}</div>'
                        f'{tag(sig.get("category","").replace("_"," "), "entity")}'
                        f'</div>'
                        f'<div class="stat-card-sub">'
                        f'{n_m} mention{"s" if n_m>1 else ""} · '
                        f'{hp_m}  · '
                        f'tone <span style="color:{sc};">{sent_v:+.2f}</span> · '
                        f'signal <strong style="color:#fde047;">{strength:.2f}</strong>'
                        f'</div>'
                        f'{ex_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No rare signals detected")

        with col_s2:
            st.markdown('<div class="section-label">Narrative Shifts</div>', unsafe_allow_html=True)
            shifts = weak_signals.get("narrative_shifts", [])
            if shifts:
                for shift in shifts:
                    delta    = shift.get("delta", 0)
                    before   = shift.get("sentiment_before", 0)
                    after    = shift.get("sentiment_after", 0)
                    dt_s     = shift.get("datetime", "")[:16]
                    direction = shift.get("direction", "")
                    sc       = "#4ade80" if delta > 0 else "#f87171"
                    icon     = "▲" if delta > 0 else "▼"
                    card_cls = "stat-card-green" if delta > 0 else "stat-card-red"
                    pivot    = shift.get("pivot_tweet", "")
                    piv_html = (
                        f'<div style="font-size:11px;color:#374151;margin-top:6px;'
                        f'font-style:italic;border-left:2px solid #1e2d40;padding-left:8px;">'
                        + pivot[:100] + "</div>"
                    ) if pivot else ""

                    st.markdown(
                        f'<div class="stat-card {card_cls}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<div class="stat-card-title">'
                        f'<span style="color:{sc};">{icon}</span> '
                        f'{direction.replace("_"," ").title()}</div>'
                        f'<span style="font-family:JetBrains Mono,monospace;font-size:10.5px;color:#374151;">'
                        f'{dt_s}</span></div>'
                        f'<div class="stat-card-sub">'
                        f'{before:.2f} → <strong style="color:{sc};">{after:.2f}</strong> '
                        f'(Δ <span style="color:{sc};">{delta:+.2f}</span>)'
                        f'</div>{piv_html}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No significant narrative shifts detected")


# ─── Page Synthesis ───────────────────────────────────────────────────────────

def render_synthesis_page(df: pd.DataFrame, documents: list,
                           emerging: dict, consensus_report: dict,
                           weak_signals: dict):
    page_header("", "AI Synthesis", "Automated macro digest · Q&A on data")

    if "llm" not in st.session_state:
        from dotenv import load_dotenv
        load_dotenv(BASE / ".env")
        from src.synthesis.llm_digest import LLMDigest
        st.session_state.llm = LLMDigest()

    llm = st.session_state.llm

    if llm.is_available:
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:8px;background:#22c55e12;'
            'border:1px solid #22c55e30;border-radius:8px;padding:8px 14px;margin-bottom:1rem;'
            'font-size:12.5px;color:#4ade80;">'
            '<span class="dot-ok"></span>LLM Connected — Groq Llama 3.3 70B / Claude</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:8px;background:#fbbf2412;'
            'border:1px solid #fbbf2430;border-radius:8px;padding:8px 14px;margin-bottom:1rem;'
            'font-size:12.5px;color:#fcd34d;">'
            '<span class="dot-warn"></span>No LLM — add GROQ_API_KEY to .env</div>',
            unsafe_allow_html=True,
        )

    analysis_data = dict(tweets_df=df, corpus_docs=documents,
                         emerging_themes=emerging,
                         consensus_report=consensus_report,
                         weak_signals=weak_signals)

    # ── Daily Brief ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Daily Macro Brief</div>', unsafe_allow_html=True)
    if st.button(" Generate Digest", type="primary", disabled=df.empty):
        with st.spinner("Synthesizing macro intelligence…"):
            try:
                placeholder = st.empty()
                text = ""
                for chunk in llm.generate_digest(analysis_data, stream=llm.is_available):
                    text += chunk
                    placeholder.markdown(
                        f'<div style="background:#111827;border:1px solid #1e2d40;border-radius:10px;'
                        f'padding:20px 24px;line-height:1.75;font-size:14px;color:#d1d5db;">{text}</div>',
                        unsafe_allow_html=True,
                    )
                st.session_state["digest"] = text
            except Exception as e:
                st.error(f"Error: {e}")
    elif "digest" in st.session_state:
        st.markdown(
            f'<div style="background:#111827;border:1px solid #1e2d40;border-radius:10px;'
            f'padding:20px 24px;line-height:1.75;font-size:14px;color:#d1d5db;">'
            f'{st.session_state["digest"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Q&A ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)

    suggestions = [
        "What is the consensus view on Fed rate policy?",
        "How do tweets vs research differ on oil?",
        "Which EM risks are mentioned in the corpus?",
        "Any unusual patterns in tweet sentiment?",
    ]
    cols = st.columns(len(suggestions))
    for col, sugg in zip(cols, suggestions):
        with col:
            if st.button(sugg[:38] + "…", use_container_width=True):
                st.session_state["qa_question"] = sugg

    question = st.text_input(
        "question",
        value=st.session_state.get("qa_question", ""),
        placeholder="Ask anything about the macro data…",
        label_visibility="collapsed",
        key="qa_input",
    )
    # Sync input back to session state so it survives reruns
    if question != st.session_state.get("qa_question", ""):
        st.session_state["qa_question"] = question

    if question and st.button("Ask →", type="primary"):
        with st.spinner("Analyzing…"):
            try:
                ans_ph = st.empty()
                ans = ""
                for chunk in llm.answer_question(question, analysis_data,
                                                  stream=llm.is_available):
                    ans += chunk
                    ans_ph.markdown(
                        f'<div style="background:#111827;border:1px solid #1e2d40;border-radius:10px;'
                        f'padding:16px 20px;line-height:1.7;font-size:13.5px;color:#d1d5db;">{ans}</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown('<div class="section-label">Session Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="kpi-strip">'
        + kpi("Tweets", f"{len(df):,}" if not df.empty else "0", "loaded", "#00d4ff")
        + kpi("Reports", str(len(documents)), "research corpus", "#a78bfa")
        + kpi("Emerging", str(emerging.get("n_emerging", 0)), "themes detected", "#ffd700")
        + kpi("Signals", str(weak_signals.get("n_rare_signals", 0)), "weak signals", "#f97316")
        + '</div>',
        unsafe_allow_html=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    with st.spinner("Loading data…"):
        df           = load_tweets()
        documents    = load_corpus()
        emerging     = load_emerging_themes()
        consensus    = load_consensus_report()
        weak_sigs    = load_weak_signals()
        cross_ref    = load_cross_links()
        bridges      = load_thematic_bridges()

    page = render_sidebar()

    if "Feed" in page:
        render_feed_page(df)
    elif "Corpus" in page:
        render_corpus_page(documents, cross_ref)
    elif "Insights" in page:
        render_insights_page(df, emerging, consensus, weak_sigs)
    elif "Synthesis" in page:
        render_synthesis_page(df, documents, emerging, consensus, weak_sigs)


if __name__ == "__main__":
    main()
