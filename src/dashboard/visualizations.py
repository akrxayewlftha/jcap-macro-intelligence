"""
Visualizations — graphiques Plotly avec thème dark finance cohérent.
Toutes les figures partagent le même système de design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ─── Design system ────────────────────────────────────────────────────────────

BG       = "#080c14"
SURFACE  = "#111827"
SURFACE2 = "#141e2e"
BORDER   = "#1e2d40"
GRID     = "#1a2535"
TEXT     = "#d1d5db"
TEXT_DIM = "#6b7280"
CYAN     = "#00d4ff"
PURPLE   = "#a78bfa"
GREEN    = "#4ade80"
RED      = "#f87171"
ORANGE   = "#fb923c"
GOLD     = "#fde047"
MONO     = "JetBrains Mono, monospace"
SANS     = "Inter, sans-serif"

SENTIMENT_COLORS = {"positive": GREEN, "negative": RED, "neutral": TEXT_DIM}
PRIORITY_COLORS  = {"high": RED, "medium": GOLD, "normal": "#60a5fa", "low": TEXT_DIM}
COLORS = dict(
    positive=GREEN, negative=RED, neutral=TEXT_DIM,
    emerging=GOLD, consensus=GREEN, divergence=ORANGE,
    surface=SURFACE, grid=GRID,
)

_BASE_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family=SANS, color=TEXT, size=12),
    margin=dict(l=12, r=12, t=40, b=12),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor=SURFACE2, bordercolor=BORDER,
        font=dict(family=SANS, color=TEXT, size=12),
    ),
    legend=dict(
        bgcolor="rgba(17,24,39,0.85)", bordercolor=BORDER,
        borderwidth=1, font=dict(size=11, color=TEXT_DIM),
    ),
    xaxis=dict(
        gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER,
        tickfont=dict(family=MONO, size=10, color=TEXT_DIM),
        title_font=dict(family=SANS, size=11, color=TEXT_DIM),
    ),
    yaxis=dict(
        gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER,
        tickfont=dict(family=MONO, size=10, color=TEXT_DIM),
        title_font=dict(family=SANS, size=11, color=TEXT_DIM),
    ),
)


def _fig(extra: dict = None) -> go.Figure:
    """Crée une figure avec le layout de base appliqué."""
    fig = go.Figure()
    layout = dict(_BASE_LAYOUT)
    if extra:
        layout.update(extra)
    fig.update_layout(**layout)
    return fig


def _hex_alpha(hex_color: str, alpha: float) -> str:
    """Convertit un hex 6 chiffres + alpha en rgba() — compatible Plotly 5.9."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _title(text: str) -> dict:
    return dict(text=text, font=dict(family=SANS, size=13, color=TEXT), x=0, xanchor="left", pad=dict(l=4))


# ─── Sentiment Timeline ───────────────────────────────────────────────────────

def sentiment_timeline(df: pd.DataFrame) -> go.Figure:
    """Volume horaire (barres) + score sentiment lissé (ligne)."""
    if df.empty or "datetime" not in df.columns:
        return _fig()

    tmp = df.copy()
    tmp["bucket"] = pd.to_datetime(tmp["datetime"], utc=True).dt.floor("h")
    has_sent = "sentiment_score" in tmp.columns

    hourly = tmp.groupby("bucket").agg(
        count=("id", "count"),
        score=(("sentiment_score", "mean") if has_sent else ("id", "count")),
    ).reset_index()

    # Couleur des barres selon sentiment moyen
    bar_colors = []
    for s in (hourly["score"] if has_sent else [0] * len(hourly)):
        if s > 0.12:
            bar_colors.append(f"rgba(74,222,128,0.55)")
        elif s < -0.12:
            bar_colors.append(f"rgba(248,113,113,0.55)")
        else:
            bar_colors.append(f"rgba(96,165,250,0.4)")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=hourly["bucket"], y=hourly["count"],
        name="Volume",
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>%{y} tweets<extra></extra>",
    ), secondary_y=False)

    if has_sent:
        smooth = hourly["score"].rolling(4, min_periods=1, center=True).mean()
        fig.add_trace(go.Scatter(
            x=hourly["bucket"], y=smooth,
            name="Sentiment",
            mode="lines",
            line=dict(color=CYAN, width=2.5, shape="spline", smoothing=1.0),
            hovertemplate="Sentiment: %{y:.2f}<extra></extra>",
        ), secondary_y=True)

        # Bande neutre
        fig.add_hrect(y0=-0.12, y1=0.12, fillcolor="rgba(107,114,128,0.04)",
                       line_width=0, secondary_y=True)

    fig.update_layout(**_BASE_LAYOUT,
                       title=_title("Tweet Volume & Sentiment"),
                       barmode="overlay", showlegend=True)
    fig.update_layout(legend=dict(orientation="h", y=1.02, x=1, xanchor="right",
                                   bgcolor="rgba(0,0,0,0)",
                                   font=dict(size=11, color=TEXT_DIM)))
    fig.update_yaxes(title_text="Tweets", secondary_y=False, gridcolor=GRID)
    fig.update_yaxes(title_text="Score", secondary_y=True, gridcolor=GRID,
                     range=[-1, 1], zeroline=True, zerolinecolor=GRID,
                     tickformat=".1f")
    return fig


# ─── Entity Frequency Bars ───────────────────────────────────────────────────

def entity_frequency_bars(entity_freq: dict, category: str, top_n: int = 12) -> go.Figure:
    data = entity_freq.get(category, {})
    if not data:
        return _fig()

    items  = sorted(data.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels = [i[0] for i in items]
    values = [i[1] for i in items]

    max_v  = max(values, default=1)
    # Dégradé cyan → purple selon le rang
    clrs = [f"rgba({int(0+i*120/len(values))},{int(212-i*80/len(values))},{int(255-i*50/len(values))},0.75)"
            for i in range(len(values))]

    fig = _fig(dict(title=_title(category.replace("_", " ").title())))
    fig.add_trace(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(color=clrs, line=dict(width=0)),
        text=[str(v) for v in values],
        textposition="outside",
        textfont=dict(family=MONO, size=10, color=TEXT_DIM),
        hovertemplate="<b>%{y}</b>: %{x} mentions<extra></extra>",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis_title="Mentions",
        yaxis=dict(autorange="reversed", gridcolor=GRID, tickfont=dict(family=SANS, size=11)),
    )
    return fig


# ─── Activity Heatmap ────────────────────────────────────────────────────────

def activity_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty or "hour" not in df.columns or "date_str" not in df.columns:
        return _fig()

    pivot = df.groupby(["date_str", "hour"]).size().unstack(fill_value=0)

    fig = _fig(dict(title=_title("Activity Heatmap — Tweets by Hour × Day")))
    fig.add_trace(go.Heatmap(
        z=pivot.values.T,
        x=pivot.index.tolist(),
        y=[f"{h:02d}:00" for h in range(24)],
        colorscale=[
            [0.00, BG],
            [0.20, "#0d2035"],
            [0.50, "#0a4a6e"],
            [0.75, "#0080cc"],
            [1.00, CYAN],
        ],
        hovertemplate="<b>%{x} %{y}</b><br>%{z} tweets<extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="Tweets", font=dict(size=11, color=TEXT_DIM)),
            tickfont=dict(family=MONO, size=9, color=TEXT_DIM),
            thickness=12, len=0.7,
        ),
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        xaxis_title="",
        yaxis=dict(tickmode="array",
                   tickvals=[f"{h:02d}:00" for h in range(0, 24, 3)],
                   ticktext=[f"{h:02d}:00 UTC" for h in range(0, 24, 3)],
                   autorange="reversed",
                   gridcolor=GRID,
                   tickfont=dict(family=MONO, size=10)),
        height=280,
    )
    return fig


# ─── Sentiment Donut ─────────────────────────────────────────────────────────

def sentiment_donut(df: pd.DataFrame) -> go.Figure:
    if "sentiment" not in df.columns:
        return _fig()

    counts = df["sentiment"].value_counts()
    total  = counts.sum()

    fig = _fig(dict(title=_title("Sentiment"), showlegend=False))
    fig.add_trace(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.68,
        marker=dict(
            colors=[SENTIMENT_COLORS.get(l, TEXT_DIM) for l in counts.index],
            line=dict(color=BG, width=3),
        ),
        textinfo="percent",
        textfont=dict(family=MONO, size=11),
        insidetextorientation="horizontal",
        hovertemplate="<b>%{label}</b><br>%{value} tweets (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        annotations=[dict(
            text=f'<span style="font-size:18px;font-weight:700;color:{TEXT};">{total}</span>'
                 f'<br><span style="font-size:10px;color:{TEXT_DIM};">tweets</span>',
            x=0.5, y=0.5, showarrow=False, align="center",
        )],
        height=260,
        margin=dict(l=0, r=0, t=36, b=0),
    )
    return fig


# ─── Velocity Bubble ─────────────────────────────────────────────────────────

def velocity_bubble_chart(velocity_df: pd.DataFrame, category: str = "") -> go.Figure:
    if velocity_df is None or velocity_df.empty:
        return _fig()

    top = velocity_df.head(20).copy()
    top["size"] = (top["count_recent"].clip(1, 30) * 4 + 12).astype(float)

    fig = _fig(dict(
        title=_title(f"Theme Velocity — {category.replace('_',' ').title()}"),
        xaxis_title="Frequency (early period)",
        yaxis_title="Frequency (recent period)",
    ))

    # Axe de non-changement
    max_val = max(top["freq_old"].max(), top["freq_recent"].max(), 0.001)
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val * 1.1, y1=max_val * 1.1,
                  line=dict(color=BORDER, dash="dot", width=1))

    # Zones : emerging (haut-gauche) vs declining (bas-droite)
    fig.add_annotation(x=0, y=max_val, xref="x", yref="y",
                       text="Emerging ↑", showarrow=False,
                       font=dict(size=10, color=GOLD), xanchor="left")
    fig.add_annotation(x=max_val, y=0, xref="x", yref="y",
                       text="Declining ↓", showarrow=False,
                       font=dict(size=10, color=RED), xanchor="right")

    fig.add_trace(go.Scatter(
        x=top["freq_old"], y=top["freq_recent"],
        mode="markers+text",
        text=top["entity"],
        textposition="top center",
        textfont=dict(family=SANS, size=9.5, color=TEXT_DIM),
        marker=dict(
            size=top["size"],
            color=top["velocity"],
            colorscale=[[0, TEXT_DIM], [0.4, "#60a5fa"], [0.7, CYAN], [1, GOLD]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Velocity", font=dict(size=10, color=TEXT_DIM)),
                tickfont=dict(family=MONO, size=9, color=TEXT_DIM),
                thickness=10, len=0.6,
            ),
            line=dict(color=BG, width=1.5),
            opacity=0.9,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Old freq: %{x:.3f}<br>"
            "Recent freq: %{y:.3f}<extra></extra>"
        ),
    ))
    return fig


# ─── Consensus Scatter ───────────────────────────────────────────────────────

def consensus_scatter(consensus_report: dict) -> go.Figure:
    items = (consensus_report.get("consensus", []) +
             consensus_report.get("divergences", []))
    if not items:
        return _fig()

    df = pd.DataFrame(items).dropna(subset=["tweet_score", "corpus_score"])
    if df.empty:
        return _fig()

    colors = [GREEN if t == "consensus" else ORANGE for t in df["signal_type"]]
    sizes  = (df["tweet_count"].clip(3, 40) + 8).astype(float)

    fig = _fig(dict(
        title=_title("Tweets vs Corpus — Sentiment Alignment"),
        xaxis=dict(range=[-1, 1], gridcolor=GRID, zeroline=True, zerolinecolor=GRID,
                   title_text="Tweet Sentiment Score"),
        yaxis=dict(range=[-1, 1], gridcolor=GRID, zeroline=True, zerolinecolor=GRID,
                   title_text="Corpus Sentiment Score"),
    ))

    # Quadrant labels
    for txt, x, y, ha in [
        ("Both Positive", 0.7, 0.85, "center"),
        ("Both Negative", -0.7, -0.85, "center"),
        ("Market ↑ / Research ↓", 0.7, -0.85, "center"),
        ("Market ↓ / Research ↑", -0.7, 0.85, "center"),
    ]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                           font=dict(size=9, color="rgba(107,114,128,0.5)"), xanchor="center")

    # Diagonale consensus
    fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1,
                  line=dict(color="rgba(74,222,128,0.25)", dash="dot", width=1.5))

    fig.add_trace(go.Scatter(
        x=df["tweet_score"], y=df["corpus_score"],
        mode="markers+text",
        text=df["entity"],
        textposition="top right",
        textfont=dict(family=SANS, size=9, color=TEXT_DIM),
        marker=dict(size=sizes, color=colors, opacity=0.85,
                    line=dict(color=BG, width=1.5)),
        hovertemplate="<b>%{text}</b><br>Tweets: %{x:.2f}<br>Corpus: %{y:.2f}<extra></extra>",
    ))

    # Légende manuelle
    for label, color in [("Consensus", GREEN), ("Divergence", ORANGE)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=color),
            name=label, showlegend=True,
        ))

    fig.update_layout(showlegend=True)
    return fig


# ─── Corpus Treemap ───────────────────────────────────────────────────────────

def corpus_overview_treemap(documents: list) -> go.Figure:
    if not documents:
        return _fig()

    labels, parents, values, colors_list, hover = [], [], [], [], []

    # Racine — valeur = somme totale des mots (requis par branchvalues="total")
    total_words = sum(max(d.get("word_count", 0), 1) for d in documents)
    labels.append("Corpus")
    parents.append("")
    values.append(total_words)
    colors_list.append(BG)
    hover.append(f"Research Corpus · {total_words:,} words")

    # Sources
    unique_sources = list(dict.fromkeys(d.get("source", "Unknown") for d in documents))
    for src in unique_sources:
        src_words = sum(d.get("word_count", 0) for d in documents if d.get("source") == src)
        labels.append(src)
        parents.append("Corpus")
        values.append(src_words)
        colors_list.append(SURFACE2)
        hover.append(src)

    # Documents
    seen = set(labels)
    for i, doc in enumerate(documents):
        src   = doc.get("source", "Unknown")
        raw   = doc.get("title", doc.get("id", f"Doc {i}"))
        short = raw[:42]
        uniq  = short
        n = 2
        while uniq in seen:
            uniq = f"{short} ({n})"
            n += 1
        seen.add(uniq)

        words   = max(doc.get("word_count", 100), 1)
        score   = doc.get("sentiment_score") or 0   # None → 0
        date    = doc.get("date", "")
        s_label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
        # Couleur : teinte selon sentiment, intensité selon taille
        base_colors = {"positive": (74, 222, 128), "negative": (248, 113, 113), "neutral": (96, 165, 250)}
        r, g, b = base_colors[s_label]

        labels.append(uniq)
        parents.append(src)
        values.append(words)
        colors_list.append(f"rgba({r},{g},{b},0.45)")
        hover.append(f"{src} · {date}<br>{words:,} words")

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors_list,
            line=dict(width=2, color=BG),
            pad=dict(t=18, l=3, r=3, b=3),
        ),
        texttemplate="<b>%{label}</b>",
        textfont=dict(family=SANS, size=11, color=TEXT),
        hovertemplate="<b>%{label}</b><br>%{customdata}<extra></extra>",
        customdata=hover,
        pathbar=dict(
            visible=True, thickness=22,
            textfont=dict(family=SANS, size=11, color=TEXT_DIM),
        ),
        root_color=BG,
        branchvalues="total",
    ))

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        title=_title("Research Corpus — Size × Sentiment"),
        font=dict(family=SANS, color=TEXT),
        margin=dict(l=0, r=0, t=40, b=0),
        height=400,
    )
    return fig


# ─── Weak Signals Gauge ──────────────────────────────────────────────────────

def weak_signals_gauge(n_signals: int, n_shifts: int) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "indicator"}, {"type": "indicator"}]])

    for col, val, label, color, max_v in [
        (1, n_signals, "Rare Signals", GOLD, 20),
        (2, n_shifts,  "Narrative Shifts", ORANGE, 10),
    ]:
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            title=dict(text=label, font=dict(family=SANS, size=12, color=TEXT_DIM)),
            number=dict(font=dict(family=MONO, size=28, color=TEXT)),
            gauge=dict(
                axis=dict(range=[0, max_v],
                          tickfont=dict(family=MONO, size=9, color=TEXT_DIM),
                          tickcolor=GRID),
                bar=dict(color=color, thickness=0.25),
                bgcolor=SURFACE,
                borderwidth=1, bordercolor=BORDER,
                steps=[
                    dict(range=[0, max_v * 0.4],        color=_hex_alpha(color, 0.09)),
                    dict(range=[max_v * 0.4, max_v * 0.75], color=_hex_alpha(color, 0.16)),
                    dict(range=[max_v * 0.75, max_v],   color=_hex_alpha(color, 0.22)),
                ],
                threshold=dict(
                    line=dict(color=RED, width=2),
                    thickness=0.75,
                    value=max_v * 0.8,
                ),
            ),
        ), row=1, col=col)

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=SANS, color=TEXT),
        margin=dict(l=20, r=20, t=20, b=20),
        height=180,
    )
    return fig
