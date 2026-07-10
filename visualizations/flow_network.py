"""
flow_network.py — Multi-country schematic flow network (Tab 6).

Countries are positioned on a 2-D schematic canvas (0-100 × 0-100)
using approximate geographic centroids per UN M49 sub-region.
Bubbles are sized by total BACI trade volume (imports + exports across
all categories for the selected years).  Directed curved arrows show
bilateral lead-product flows between selected countries.

Public API
----------
build_flow_network(baci_df, active_years, countries,
    categories, min_flow, show_imports, show_exports) → go.Figure
"""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go

from model.regions import REGION_MAP

# ── Schematic region centroids (x: 0-100, y: 0-100, origin top-left) ─────────

_REGION_XY: dict[str, tuple[float, float]] = {
    "Western Africa":    (30, 55),
    "Eastern Africa":    (42, 58),
    "Northern Africa":   (35, 42),
    "Middle Africa":     (35, 52),
    "Southern Africa":   (38, 68),
    "Western Europe":    (40, 28),
    "Northern Europe":   (40, 22),
    "Southern Europe":   (43, 32),
    "Eastern Europe":    (52, 25),
    "Western Asia":      (58, 42),
    "Central Asia":      (62, 30),
    "Southern Asia":     (65, 48),
    "South-eastern Asia":(75, 52),
    "Eastern Asia":      (82, 35),
    "Northern America":  (15, 32),
    "Central America":   (18, 45),
    "Caribbean":         (22, 45),
    "South America":     (22, 65),
    "Oceania":           (85, 70),
    "Other":             (50, 50),
}

# ── Product category colours ──────────────────────────────────────────────────

_CAT_COLORS: dict[str, str] = {
    "BATT":  "#43A047",   # 🟩 green  — New Batteries
    "USED":  "#FDD835",   # 🟨 yellow — Used Batteries
    "SCRAP": "#FB8C00",   # 🟧 orange — Lead Scrap
    "FEED":  "#1E88E5",   # 🟦 blue   — Smelted Lead
    "ORE":   "#9E9E9E",   # ⬜ grey   — Ore & Concentrates
}

_CAT_DISPLAY: dict[str, str] = {
    "FEED":  "Smelted Lead (FEED)",
    "BATT":  "New Batteries (BATT)",
    "USED":  "Used Batteries (USED)",
    "SCRAP": "Lead Scrap (SCRAP)",
    "ORE":   "Ore & Concentrates (ORE)",
}

# ── Country label abbreviations (BACI names) ──────────────────────────────────

_ABBREV: dict[str, str] = {
    "Rep. of Korea":              "S. Korea",
    "Russian Federation":         "Russia",
    "Dem. People's Rep. of Korea":"N. Korea",
    "United Kingdom":             "UK",
    "United Arab Emirates":       "UAE",
    "Dem. Rep. of the Congo":     "DR Congo",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Lao People's Dem. Rep.":     "Laos",
    "Brunei Darussalam":          "Brunei",
    "Central African Rep.":       "CAR",
    "United Rep. of Tanzania":    "Tanzania",
    "Bosnia Herzegovina":         "Bosnia",
    "Rep. of Moldova":            "Moldova",
    "State of Palestine":         "Palestine",
}


def _label(country: str) -> str:
    if country in _ABBREV:
        return _ABBREV[country]
    return country if len(country) <= 13 else country[:12] + "…"


# ── Layout helpers ─────────────────────────────────────────────────────────────

def _spread_offsets(n: int, step: float = 4.5) -> list[tuple[float, float]]:
    """Evenly distribute n points in a small grid around (0, 0)."""
    if n == 1:
        return [(0.0, 0.0)]
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    offsets: list[tuple[float, float]] = []
    idx = 0
    for row in range(rows):
        row_count = min(cols, n - idx)
        for col in range(row_count):
            dx = (col - (row_count - 1) / 2) * step
            dy = (row - (rows - 1) / 2) * step
            offsets.append((dx, dy))
            idx += 1
    return offsets


# Approximate country centroids on the 0-100 schematic canvas (x = longitude-ish,
# y = latitude-ish; origin top-left). Coverage is best-effort — countries not in
# this table fall back to their UN sub-region centroid in _REGION_XY.
_COUNTRY_XY: dict[str, tuple[float, float]] = {
    "USA": (18, 32), "Canada": (18, 22), "Mexico": (16, 42),
    "Brazil": (28, 62), "Argentina": (24, 78), "Chile": (22, 76),
    "Colombia": (22, 52), "Peru": (22, 62), "Venezuela": (24, 50),
    "United Kingdom": (40, 22), "Ireland": (38, 22), "France": (42, 28),
    "Germany": (45, 25), "Netherlands": (44, 24), "Belgium": (43, 25),
    "Spain": (40, 33), "Portugal": (38, 33), "Italy": (46, 32),
    "Poland": (49, 24), "Sweden": (46, 18), "Norway": (44, 17),
    "Denmark": (45, 21), "Finland": (49, 17), "Switzerland": (45, 28),
    "Austria": (47, 28), "Greece": (50, 34), "Türkiye": (54, 34),
    "Russian Federation": (60, 20),
    "Egypt": (52, 42), "Morocco": (36, 40), "Algeria": (42, 40),
    "Tunisia": (46, 38), "Libya": (48, 42), "Sudan": (52, 48),
    "Ethiopia": (54, 52), "Kenya": (54, 58), "Tanzania": (54, 62),
    "Ghana": (40, 55), "Nigeria": (44, 55), "Côte d'Ivoire": (38, 55),
    "Senegal": (34, 52), "Togo": (42, 55), "Burkina Faso": (40, 52),
    "Mali": (38, 50), "Guinea": (35, 53), "Sierra Leone": (34, 55),
    "Liberia": (36, 56), "Cameroon": (46, 56), "South Africa": (50, 72),
    "Angola": (48, 65), "Zambia": (50, 65), "Zimbabwe": (52, 68),
    "Mozambique": (54, 68), "Namibia": (48, 70), "Botswana": (50, 70),
    "Dem. Rep. of the Congo": (48, 60),
    "Saudi Arabia": (58, 42), "United Arab Emirates": (60, 44),
    "Iran": (62, 38), "Iraq": (58, 38), "Israel": (55, 40),
    "Jordan": (56, 40), "Lebanon": (55, 38), "Qatar": (60, 43),
    "Oman": (61, 45), "Kuwait": (59, 40), "Yemen": (59, 47),
    "India": (68, 46), "Pakistan": (65, 42), "Bangladesh": (72, 46),
    "Sri Lanka": (69, 52), "Nepal": (70, 42),
    "China": (78, 34), "Japan": (88, 32), "Rep. of Korea": (85, 34),
    "Dem. People's Rep. of Korea": (85, 30),
    "Taiwan": (85, 40), "Mongolia": (78, 26), "Hong Kong": (82, 40),
    "Thailand": (75, 48), "Viet Nam": (77, 48), "Malaysia": (76, 54),
    "Singapore": (76, 55), "Indonesia": (80, 58), "Philippines": (82, 50),
    "Cambodia": (76, 50), "Myanmar": (73, 46), "Laos": (76, 46),
    "Australia": (86, 72), "New Zealand": (93, 78),
    "Kazakhstan": (65, 26), "Uzbekistan": (63, 32), "Ukraine": (52, 26),
}


def _country_positions_circle(
    countries: list[str], focal_country: str | None
) -> dict[str, tuple[float, float]]:
    cx, cy, r = 50.0, 50.0, 38.0
    non_focal = [c for c in countries if c != focal_country]
    n = len(non_focal)
    positions: dict[str, tuple[float, float]] = {}
    if focal_country and focal_country in countries:
        positions[focal_country] = (cx, cy)
    for i, c in enumerate(non_focal):
        angle = -math.pi / 2 + 2 * math.pi * i / max(n, 1)
        positions[c] = (cx + r * math.cos(angle), cy + r * math.sin(angle))
    return positions


def _country_positions_grid(
    countries: list[str], focal_country: str | None
) -> dict[str, tuple[float, float]]:
    ordered = ([focal_country] if focal_country and focal_country in countries else []) + [
        c for c in countries if c != focal_country
    ]
    n = len(ordered)
    if n == 0:
        return {}
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    x_pad, y_pad = 12.0, 12.0
    x_span, y_span = 100.0 - 2 * x_pad, 100.0 - 2 * y_pad
    dx = x_span / max(cols - 1, 1) if cols > 1 else 0
    dy = y_span / max(rows - 1, 1) if rows > 1 else 0
    positions: dict[str, tuple[float, float]] = {}
    for i, c in enumerate(ordered):
        row, col = divmod(i, cols)
        x = x_pad + col * dx if cols > 1 else 50.0
        y = y_pad + row * dy if rows > 1 else 50.0
        positions[c] = (x, y)
    return positions


def _country_positions_geographic(
    countries: list[str], focal_country: str | None
) -> dict[str, tuple[float, float]]:
    """Position each country at its approximate world centroid.
    Falls back to the country's UN sub-region centroid, then to (50, 50).
    Small offsets are added when multiple countries share a fallback."""
    positions: dict[str, tuple[float, float]] = {}
    fallback_buckets: dict[tuple[float, float], list[str]] = defaultdict(list)
    for c in countries:
        if c in _COUNTRY_XY:
            positions[c] = _COUNTRY_XY[c]
        else:
            region = REGION_MAP.get(c, "Other")
            base = _REGION_XY.get(region, (50.0, 50.0))
            fallback_buckets[base].append(c)
    for base, members in fallback_buckets.items():
        offsets = _spread_offsets(len(members), step=4.5)
        for c, (dx, dy) in zip(members, offsets):
            positions[c] = (base[0] + dx, base[1] + dy)
    return positions


def _country_positions_force(
    countries: list[str],
    focal_country: str | None,
    baci_df: pd.DataFrame,
    active_years: list[int],
    categories: list[str],
) -> dict[str, tuple[float, float]]:
    """Simple Fruchterman-Reingold style force layout.
    Edge weights ~ log(flow volume between the pair)."""
    n = len(countries)
    if n == 0:
        return {}
    if n == 1:
        return {countries[0]: (50.0, 50.0)}

    df = baci_df[baci_df["Year"].isin(active_years) & baci_df["category"].isin(categories)]
    pair_weight: dict[tuple[str, str], float] = defaultdict(float)
    country_set = set(countries)
    subset = df[df["Exporter"].isin(country_set) & df["Importer"].isin(country_set)]
    for _, r in subset.iterrows():
        a, b = r["Exporter"], r["Importer"]
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        pair_weight[key] += float(r["actual_lead"])

    idx = {c: i for i, c in enumerate(countries)}
    # Seed with the circle layout so nothing starts on top of the focal country
    seed = _country_positions_circle(countries, focal_country)
    pos = [[seed[c][0], seed[c][1]] for c in countries]

    area = 100.0 * 100.0
    k = math.sqrt(area / n) * 0.75
    iterations = 120
    t_start = 15.0

    for step in range(iterations):
        t = t_start * (1 - step / iterations)
        disp = [[0.0, 0.0] for _ in range(n)]

        # Repulsion (all pairs)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.hypot(dx, dy) or 0.01
                force = (k * k) / dist
                ux, uy = dx / dist, dy / dist
                disp[i][0] += ux * force
                disp[i][1] += uy * force
                disp[j][0] -= ux * force
                disp[j][1] -= uy * force

        # Attraction (edges)
        for (a, b), w in pair_weight.items():
            if a not in idx or b not in idx:
                continue
            i, j = idx[a], idx[b]
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            dist = math.hypot(dx, dy) or 0.01
            weight_log = math.log1p(w)
            force = (dist * dist) / k * (weight_log / 10.0)
            ux, uy = dx / dist, dy / dist
            disp[i][0] -= ux * force
            disp[i][1] -= uy * force
            disp[j][0] += ux * force
            disp[j][1] += uy * force

        # Focal pin
        for i, c in enumerate(countries):
            d = math.hypot(disp[i][0], disp[i][1]) or 0.01
            step_x = disp[i][0] / d * min(d, t)
            step_y = disp[i][1] / d * min(d, t)
            pos[i][0] += step_x
            pos[i][1] += step_y
            # Keep inside canvas
            pos[i][0] = max(8.0, min(92.0, pos[i][0]))
            pos[i][1] = max(8.0, min(92.0, pos[i][1]))
            if focal_country and c == focal_country:
                pos[i][0] = 50.0
                pos[i][1] = 50.0

    return {c: (pos[i][0], pos[i][1]) for i, c in enumerate(countries)}


def _country_positions(
    countries: list[str],
    focal_country: str | None = None,
    layout: str = "circle",
    baci_df: pd.DataFrame | None = None,
    active_years: list[int] | None = None,
    categories: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Dispatch to the selected layout algorithm."""
    if layout == "grid":
        return _country_positions_grid(countries, focal_country)
    if layout == "geographic":
        return _country_positions_geographic(countries, focal_country)
    if layout == "force" and baci_df is not None and active_years is not None and categories is not None:
        return _country_positions_force(countries, focal_country, baci_df, active_years, categories)
    return _country_positions_circle(countries, focal_country)


# ── Curve helpers ─────────────────────────────────────────────────────────────

def _bezier(
    x1: float, y1: float, x2: float, y2: float,
    offset: float = 5.0, n: int = 25,
) -> tuple[list[float], list[float], tuple[float, float]]:
    """
    Quadratic Bezier from (x1,y1) to (x2,y2) with a perpendicular control-point
    offset (positive = left of direction of travel).

    Returns (xs, ys, (ctrl_x, ctrl_y)).
    """
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        # Coincident points — return a tiny stub
        return [x1, x2], [y1, y2], (mx, my)
    # Left perpendicular unit vector
    px, py = -dy / length, dx / length
    cx, cy = mx + px * offset, my + py * offset
    t_vals = [i / (n - 1) for i in range(n)]
    xs = [(1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2 for t in t_vals]
    ys = [(1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2 for t in t_vals]
    return xs, ys, (cx, cy)


# ── Bubble sizes ──────────────────────────────────────────────────────────────

def _bubble_sizes(
    countries: list[str],
    baci_df: pd.DataFrame,
    active_years: list[int],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Return (sizes_px, imp_vols, exp_vols) where sizes_px maps country →
    pixel diameter (20-80), and imp_vols/exp_vols map country → annualised
    total imports/exports across all categories.
    """
    subset = baci_df[baci_df["Year"].isin(active_years)]
    n_years = len(active_years)

    imp_vols: dict[str, float] = {}
    exp_vols: dict[str, float] = {}
    for c in countries:
        imp_vols[c] = float(subset[subset["Importer"] == c]["actual_lead"].sum()) / n_years
        exp_vols[c] = float(subset[subset["Exporter"] == c]["actual_lead"].sum()) / n_years

    total_vols = {c: imp_vols[c] + exp_vols[c] for c in countries}
    sqrt_vals  = {c: math.sqrt(v) for c, v in total_vols.items()}
    min_s = min(sqrt_vals.values())
    max_s = max(sqrt_vals.values())

    sizes: dict[str, float] = {}
    for c, s in sqrt_vals.items():
        t = (s - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        sizes[c] = 20.0 + t * 60.0
    return sizes, imp_vols, exp_vols


# ── Public: figure ────────────────────────────────────────────────────────────

def build_flow_network(
    baci_df: pd.DataFrame,
    active_years: list[int],
    countries: list[str],
    categories: list[str],
    min_flow: float,
    show_imports: bool,
    show_exports: bool,
    focal_country: str | None = None,
    layout: str = "circle",
    focal_only: bool = False,
    hidden_pairs: set[frozenset[str]] | None = None,
    prune_isolated: bool = False,
) -> go.Figure:
    """
    Build a schematic flow-network figure.

    Parameters
    ----------
    categories     : list of category codes to include (subset of FEED/BATT/USED/SCRAP)
    min_flow       : annualised t Pb threshold — flows below this are hidden
    show_imports   : include arrows where the Importer is a selected country
    show_exports   : include arrows where the Exporter is a selected country
    layout         : "circle" | "grid" | "geographic" | "force"
    focal_only     : if True and focal_country is set, only arrows touching the
                     focal country are drawn (arrows between two non-focal
                     countries are suppressed)
    hidden_pairs   : set of frozenset({country_a, country_b}) — any arrow between
                     the two countries (in either direction) is suppressed
    prune_isolated : if True and focal_country is set, drop any country that has
                     no remaining arrow (after all other filters) with the focal
                     country. Focal country itself is always kept.
    """
    df = baci_df[baci_df["Year"].isin(active_years)]
    n_years = len(active_years)
    country_set = set(countries)

    # ── Flows between selected countries ──────────────────────────────────────
    flows = (
        df[
            df["Exporter"].isin(country_set)
            & df["Importer"].isin(country_set)
            & df["category"].isin(categories)
            & (df["Exporter"] != df["Importer"])
        ]
        .groupby(["Exporter", "Importer", "category"])["actual_lead"]
        .sum()
        .reset_index()
    )
    flows["actual_lead"] = flows["actual_lead"] / n_years

    # Min-flow threshold
    flows = flows[flows["actual_lead"] >= min_flow]

    # Direction filter — relative to the focal country. Every arrow between two
    # selected countries is simultaneously an import (for its Importer) and an
    # export (for its Exporter), so "imports vs exports" is only well-defined
    # relative to a chosen focal country. Without one, both flags are True
    # ("Both") and nothing is filtered.
    if focal_country and not (show_imports and show_exports):
        if show_imports and not show_exports:
            flows = flows[flows["Importer"] == focal_country]
        elif show_exports and not show_imports:
            flows = flows[flows["Exporter"] == focal_country]
        else:  # neither direction selected → show nothing
            flows = flows.iloc[0:0]

    # Focal-only filter: keep arrows only if the focal country is at one end
    if focal_only and focal_country:
        flows = flows[
            (flows["Exporter"] == focal_country)
            | (flows["Importer"] == focal_country)
        ]

    # Manual pair hider: drop any arrow whose {exporter, importer} pair is hidden
    if hidden_pairs and not flows.empty:
        _hidden_key = flows.apply(
            lambda r: frozenset([r["Exporter"], r["Importer"]]),
            axis=1,
        )
        flows = flows[~_hidden_key.isin(hidden_pairs)]

    # Prune countries with no remaining arrow to the focal country
    if prune_isolated and focal_country and focal_country in country_set:
        connected = (
            set(flows.loc[flows["Exporter"] == focal_country, "Importer"])
            | set(flows.loc[flows["Importer"] == focal_country, "Exporter"])
        )
        connected.add(focal_country)
        countries = [c for c in countries if c in connected]
        country_set = set(countries)
        flows = flows[
            flows["Exporter"].isin(country_set) & flows["Importer"].isin(country_set)
        ]

    # ── Layout ────────────────────────────────────────────────────────────────
    positions = _country_positions(
        countries,
        focal_country=focal_country,
        layout=layout,
        baci_df=baci_df,
        active_years=active_years,
        categories=categories,
    )
    sizes, imp_vols, exp_vols = _bubble_sizes(countries, baci_df, active_years)

    max_flow = flows["actual_lead"].max() if not flows.empty else 1.0

    # Detect bidirectional flows (per category) to adjust curve offset
    flow_keys: set[tuple[str, str, str]] = set(
        (r["Exporter"], r["Importer"], r["category"])
        for _, r in flows.iterrows()
    )

    fig = go.Figure()

    # ── Arrow traces ──────────────────────────────────────────────────────────
    legend_cats_shown: set[str] = set()

    for _, row in flows.iterrows():
        exp, imp, cat = row["Exporter"], row["Importer"], row["category"]
        vol = float(row["actual_lead"])

        x1, y1 = positions[exp]
        x2, y2 = positions[imp]
        color = _CAT_COLORS.get(cat, "#555")

        # Arrow width scaled by volume (sqrt)
        line_width = max(1.0, 1.0 + math.sqrt(vol / max_flow) * 7.0)

        # Curve offset: larger when both directions exist (so they separate visually)
        bidir = (imp, exp, cat) in flow_keys
        curve_offset = 7.0 if bidir else 3.5

        xs, ys, _ = _bezier(x1, y1, x2, y2, offset=curve_offset)

        hover = (
            f"<b>{exp} → {imp}</b><br>"
            f"{_CAT_DISPLAY.get(cat, cat)}: {vol:,.0f} t Pb"
        )

        first_of_cat = cat not in legend_cats_shown
        legend_cats_shown.add(cat)

        # Curve trace
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=color, width=line_width),
            hovertemplate=hover + "<extra></extra>",
            showlegend=first_of_cat,
            legendgroup=cat,
            name=_CAT_DISPLAY.get(cat, cat) if first_of_cat else "",
            legendrank=list(_CAT_COLORS).index(cat) if cat in _CAT_COLORS else 99,
        ))

        # Arrowhead via annotation (data-coordinate ax/ay for accurate direction)
        fig.add_annotation(
            x=xs[-1], y=ys[-1],
            ax=xs[-4], ay=ys[-4],
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.9,
            arrowwidth=max(1.5, line_width * 0.75),
            arrowcolor=color,
            text="",
            opacity=0.85,
        )

    # ── Bubble traces (one per country so hover works cleanly) ────────────────
    for c in countries:
        x, y = positions[c]
        sz = sizes[c]

        total_trade = imp_vols[c] + exp_vols[c]
        hover = (
            f"<b>{c}</b><br>"
            f"Total imports: {imp_vols[c]:,.0f} t Pb<br>"
            f"Total exports: {exp_vols[c]:,.0f} t Pb<br>"
            f"Total trade volume: {total_trade:,.0f} t Pb"
        )

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                size=sz,
                color="#37474F",
                line=dict(width=1.5, color="#ffffff"),
                opacity=0.88,
            ),
            text=[_label(c)],
            textposition="top center",
            textfont=dict(size=10, color="#111111"),
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
            name=c,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=30, b=60),
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
        xaxis=dict(visible=False, range=[-5, 105]),
        yaxis=dict(visible=False, range=[5, 95]),
        legend=dict(
            title=dict(text="Product category", font=dict(size=12)),
            orientation="h",
            x=0.0,
            y=-0.06,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
        hovermode="closest",
    )

    return fig
