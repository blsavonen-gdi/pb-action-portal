"""
provenance_map.py — Supply Chain Provenance visualization for Tab 5.

Traces three pathways of input flows into a selected end-market country.
Colors match the app-wide product-category palette (same as square emoji):

  PATHWAY A — Battery supply chain (greens 🟩)
    Tier A1 (#43A047): Countries supplying NEW BATTERIES (BATT) to selected country
    Tier A2 (#A5D6A7): Countries supplying FEED/SCRAP/USED to Tier A1 countries
    Tier A3 (#C8E6C9): Countries supplying SCRAP/USED to Tier A2 countries

  PATHWAY B — Direct smelted lead supply (blues 🟦)
    Tier B1 (#1E88E5): Countries supplying SMELTED LEAD (FEED) to selected country
    Tier B2 (#90CAF9): Countries supplying SCRAP/USED to Tier B1 countries

  PATHWAY C — Direct scrap/waste supply (orange 🟧)
    Tier C1 (#FB8C00): Countries supplying SCRAP/USED to selected country

Priority (highest first): selected > A1 > B1 > C1 > A2 > B2 > A3

Public API
----------
build_provenance_map(baci_df, active_years, selected_country, top_n) → go.Figure
get_provenance_tables(baci_df, active_years, selected_country, top_n)
    → (a1_df, a2_df, a3_df, b1_df, b2_df, c1_df)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from model.data_loader import BACI_TO_STANDARD_NAME

# ── Color constants ───────────────────────────────────────────────────────────
# Colors match the app-wide category palette (🟩🟨🟧🟦⬜ square emojis).

_GRAY     = "#E8E8E8"   # no trade relationship
_SELECTED = "#1565C0"   # selected end market (distinct dark blue, black border)

# Pathway A — New Batteries (green 🟩)
_A1 = "#43A047"   # direct battery suppliers
_A2 = "#A5D6A7"   # upstream feedstock suppliers
_A3 = "#C8E6C9"   # deep upstream

# Pathway B — Smelted Lead (blue 🟦)
_B1 = "#1E88E5"   # direct smelted lead suppliers
_B2 = "#90CAF9"   # upstream scrap suppliers

# Pathway C — Lead Scrap / Used Batteries (orange 🟧)
_C1 = "#FB8C00"   # direct scrap/waste battery suppliers

# Slot order: index → colour (used in _make_discrete_colorscale)
# 0=gray, 1=selected, 2=A1, 3=B1, 4=C1, 5=A2, 6=B2, 7=A3
_ALL_COLORS = [_GRAY, _SELECTED, _A1, _B1, _C1, _A2, _B2, _A3]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_discrete_colorscale(colors: list[str]) -> list[list]:
    """Step-function colorscale for N discrete slots. Pair with zmin=0, zmax=N-1."""
    n = len(colors)
    cs: list[list] = []
    for i, c in enumerate(colors):
        cs.append([i / n, c])
        cs.append([(i + 1) / n, c])
    return cs


# ── Tier computation ──────────────────────────────────────────────────────────

def _compute_tiers(
    df: pd.DataFrame,
    n_years: int,
    selected_country: str,
    top_n: int,
    max_layer: int = 3,
) -> dict:
    """
    Compute all six supply-chain tiers from year-filtered BACI data.

    Exclusion logic: each tier excludes countries already in higher-priority tiers,
    so the top-N slots surface new countries not yet visible on the map.

    Priority: A1 > B1 > C1 > A2 > B2 > A3

    Returns a dict with keys:
      a1_vol, a2_vol, a3_vol  — Series[baci_name → annualised t Pb] for pathway A
      b1_vol, b2_vol          — same for pathway B
      c1_vol                  — same for pathway C
      all_memberships         — dict[baci_name → list of (tier_label, vol)]
    """

    # ── Pathway A ─────────────────────────────────────────────────────────────

    # A1: new battery (BATT) suppliers to selected country
    a1_raw = df[
        (df["Importer"] == selected_country) & (df["category"] == "BATT")
    ]
    a1_vol = (
        a1_raw.groupby("Exporter")["actual_lead"].sum().nlargest(top_n) / n_years
    ).rename("vol")
    shown_a1 = set(a1_vol.index)

    # A2/B2/A3 only computed when max_layer allows
    _empty = pd.Series(dtype=float, name="vol")

    if max_layer >= 2:
        excl_a2 = {selected_country} | shown_a1
        a2_raw = df[
            df["Importer"].isin(shown_a1)
            & df["category"].isin(["FEED", "SCRAP", "USED"])
            & ~df["Exporter"].isin(excl_a2)
        ]
        a2_vol = (
            a2_raw.groupby("Exporter")["actual_lead"].sum().nlargest(top_n) / n_years
        ).rename("vol")
    else:
        a2_vol = _empty.copy()
    shown_a2 = set(a2_vol.index)

    # ── Pathway B ─────────────────────────────────────────────────────────────

    # B1: smelted lead (FEED) suppliers directly to selected country
    b1_raw = df[
        (df["Importer"] == selected_country) & (df["category"] == "FEED")
    ]
    b1_vol = (
        b1_raw.groupby("Exporter")["actual_lead"].sum().nlargest(top_n) / n_years
    ).rename("vol")
    shown_b1 = set(b1_vol.index)

    if max_layer >= 2:
        excl_b2 = {selected_country} | shown_a1 | shown_b1 | shown_a2
        b2_raw = df[
            df["Importer"].isin(shown_b1)
            & df["category"].isin(["SCRAP", "USED"])
            & ~df["Exporter"].isin(excl_b2)
        ]
        b2_vol = (
            b2_raw.groupby("Exporter")["actual_lead"].sum().nlargest(top_n) / n_years
        ).rename("vol")
    else:
        b2_vol = _empty.copy()
    shown_b2 = set(b2_vol.index)

    # ── Pathway C ─────────────────────────────────────────────────────────────

    # C1: SCRAP or USED suppliers directly to selected country
    c1_raw = df[
        (df["Importer"] == selected_country)
        & df["category"].isin(["SCRAP", "USED"])
    ]
    c1_vol = (
        c1_raw.groupby("Exporter")["actual_lead"].sum().nlargest(top_n) / n_years
    ).rename("vol")
    shown_c1 = set(c1_vol.index)

    # ── A3: after C1 so we can exclude it ─────────────────────────────────────

    if max_layer >= 3:
        excl_a3 = {selected_country} | shown_a1 | shown_b1 | shown_c1 | shown_a2 | shown_b2
        a3_raw = df[
            df["Importer"].isin(shown_a2)
            & df["category"].isin(["SCRAP", "USED"])
            & ~df["Exporter"].isin(excl_a3)
        ]
        a3_vol = (
            a3_raw.groupby("Exporter")["actual_lead"].sum().nlargest(top_n) / n_years
        ).rename("vol")
    else:
        a3_vol = _empty.copy()

    # ── Membership tracking (for hover tooltips) ──────────────────────────────

    all_memberships: dict[str, list[tuple[str, float]]] = {}

    def _record(vol_series: pd.Series, label: str) -> None:
        for country, vol in vol_series.items():
            all_memberships.setdefault(country, []).append((label, float(vol)))

    _record(a1_vol, "Tier A1 — new battery supplier")
    _record(a2_vol, "Tier A2 — feedstock to battery mfrs")
    _record(a3_vol, "Tier A3 — scrap to feedstock smelters")
    _record(b1_vol, "Tier B1 — smelted lead supplier")
    _record(b2_vol, "Tier B2 — scrap to lead producers")
    _record(c1_vol, "Tier C1 — scrap/waste supplier")

    return {
        "a1_vol": a1_vol,
        "a2_vol": a2_vol,
        "a3_vol": a3_vol,
        "b1_vol": b1_vol,
        "b2_vol": b2_vol,
        "c1_vol": c1_vol,
        "all_memberships": all_memberships,
    }


# ── Public: choropleth ────────────────────────────────────────────────────────

def build_provenance_map(
    baci_df: pd.DataFrame,
    active_years: list[int],
    selected_country: str,
    top_n: int,
    max_layer: int = 3,
) -> go.Figure:
    """
    Build a provenance choropleth tracing input supply chains into selected_country.

    Color coding — pathway-based, matching app category palette:
    Green  (_A1/_A2/_A3) — Pathway A: battery supply chain (darker = closer)
    Blue   (_B1/_B2)     — Pathway B: smelted lead supply (darker = closer)
    Orange (_C1)         — Pathway C: direct scrap/waste suppliers
    Dark blue(_SELECTED) — selected end-market country
    Gray  (_GRAY)        — no supply relationship shown
    """
    df = baci_df[baci_df["Year"].isin(active_years)]
    n_years = len(active_years)
    tiers = _compute_tiers(df, n_years, selected_country, top_n, max_layer=max_layer)

    a1_set  = set(tiers["a1_vol"].index)
    a2_set  = set(tiers["a2_vol"].index)
    a3_set  = set(tiers["a3_vol"].index)
    b1_set  = set(tiers["b1_vol"].index)
    b2_set  = set(tiers["b2_vol"].index)
    c1_set  = set(tiers["c1_vol"].index)
    memberships = tiers["all_memberships"]

    # Slot index → color defined by _ALL_COLORS
    # 0=gray, 1=selected, 2=A1, 3=B1, 4=C1, 5=A2, 6=B2, 7=A3

    def _z(baci_name: str) -> int:
        if baci_name == selected_country:
            return 1
        if baci_name in a1_set:
            return 2
        if baci_name in b1_set:
            return 3
        if baci_name in c1_set:
            return 4
        if baci_name in a2_set:
            return 5
        if baci_name in b2_set:
            return 6
        if baci_name in a3_set:
            return 7
        return 0

    def _hover(baci_name: str) -> str:
        if baci_name == selected_country:
            return f"<b>{baci_name}</b><br>Selected end market"
        mems = memberships.get(baci_name, [])
        if not mems:
            return f"<b>{baci_name}</b>"
        lines = [f"<b>{baci_name}</b>"]
        for tier_label, vol in mems:
            lines.append(f"{tier_label}: {vol:,.0f} t Pb")
        return "<br>".join(lines)

    # Collect all BACI countries visible in the filtered dataset
    all_baci = set(df["Exporter"].unique()) | set(df["Importer"].unique())

    locations: list[str] = []
    z_values:  list[int] = []
    hovers:    list[str] = []

    for baci_name in all_baci:
        plotly_name = BACI_TO_STANDARD_NAME.get(baci_name, baci_name)
        if plotly_name is None:
            continue
        locations.append(plotly_name)
        z_values.append(_z(baci_name))
        hovers.append(_hover(baci_name))

    n_slots = len(_ALL_COLORS)
    colorscale = _make_discrete_colorscale(_ALL_COLORS)

    main = go.Choropleth(
        locations=locations,
        z=z_values,
        locationmode="country names",
        colorscale=colorscale,
        zmin=0,
        zmax=n_slots - 1,
        showscale=False,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
        marker_line_color="#ffffff",
        marker_line_width=0.5,
    )

    traces: list[go.BaseTraceType] = [main]

    # Black border on selected country
    sel_plotly = BACI_TO_STANDARD_NAME.get(selected_country, selected_country)
    if sel_plotly:
        traces.append(go.Choropleth(
            locations=[sel_plotly],
            z=[1],
            locationmode="country names",
            colorscale=colorscale,
            zmin=0,
            zmax=n_slots - 1,
            showscale=False,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=[_hover(selected_country)],
            marker_line_color="#000000",
            marker_line_width=2,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=500,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#aaaaaa",
            showland=True,
            landcolor="#f5f5f5",
            showocean=True,
            oceancolor="#e3f2fd",
            showlakes=False,
            projection_type="natural earth",
        ),
    )
    return fig


# ── Public: summary tables ────────────────────────────────────────────────────

def get_provenance_tables(
    baci_df: pd.DataFrame,
    active_years: list[int],
    selected_country: str,
    top_n: int,
    max_layer: int = 3,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame,
    pd.DataFrame,
]:
    """
    Return (a1_df, a2_df, a3_df, b1_df, b2_df, c1_df) summary tables.

    Pathway A — Battery supply chain:
      a1_df  columns: Country, t Pb (new batteries)
      a2_df  columns: Country, t Pb (feedstock to battery mfrs)
      a3_df  columns: Country, t Pb (scrap to smelters)

    Pathway B — Direct smelted lead:
      b1_df  columns: Country, t Pb (smelted lead)
      b2_df  columns: Country, t Pb (scrap to lead producers)

    Pathway C — Direct scrap/waste:
      c1_df  columns: Country, t Pb (scrap/waste)

    All volumes annualised (sum / n_years), rounded to integer tonnes.
    """
    df = baci_df[baci_df["Year"].isin(active_years)]
    n_years = len(active_years)
    tiers = _compute_tiers(df, n_years, selected_country, top_n, max_layer=max_layer)

    def _to_df(vol_series: pd.Series, vol_col: str) -> pd.DataFrame:
        if vol_series.empty:
            return pd.DataFrame(columns=["Country", vol_col])
        return pd.DataFrame({
            "Country": vol_series.index.tolist(),
            vol_col:   vol_series.round(0).astype(int).values,
        })

    a1_df = _to_df(tiers["a1_vol"], "t Pb (new batteries)")
    a2_df = _to_df(tiers["a2_vol"], "t Pb (feedstock to battery mfrs)")
    a3_df = _to_df(tiers["a3_vol"], "t Pb (scrap to smelters)")
    b1_df = _to_df(tiers["b1_vol"], "t Pb (smelted lead)")
    b2_df = _to_df(tiers["b2_vol"], "t Pb (scrap to lead producers)")
    c1_df = _to_df(tiers["c1_vol"], "t Pb (scrap/waste)")

    return a1_df, a2_df, a3_df, b1_df, b2_df, c1_df
