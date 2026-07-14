"""
lead_accumulation.py — Lead mass accumulation tab for the Pb Action Streamlit app.

Methodology: docs/lead_accumulation_tab.md

Computes annual net lead balance for a country or sub-region:
    Net(year) = Mining(year) + Imports_Pb(year) − Exports_Pb(year)

Mining comes from BGS/USGS mine-production data (already in tonnes Pb).
Trade quantities are converted to lead content using adjustable Pb factors.

All internal quantities: metric tonnes of lead (t Pb).
All display quantities: kilotonnes (kt Pb).
"""

from __future__ import annotations

from pathlib import Path

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data"


def _sig2(v: float) -> float:
    """Round a model-derived value to 2 significant figures (UI display)."""
    if v is None or v == 0 or (isinstance(v, float) and math.isnan(v)):
        return 0.0
    exp = math.floor(math.log10(abs(v)))
    factor = 10 ** (exp - 1)
    return round(v / factor) * factor


def _fmt_kt(kt: float) -> str:
    """Signed 2-significant-figure kt string, e.g. '+460' or '+8.3'."""
    r = _sig2(kt)
    if abs(r) >= 10 or r == round(r):
        return f"{r:+,.0f}"
    return f"{r:+,.1f}"

# ── HS code metadata ──────────────────────────────────────────────────────────
# lo / hi define the adjustable range; default = midpoint (or fixed value).
HS_META: dict[int, dict] = {
    260700: {"name": "Lead Ore & Concentrates",          "cat": "Mining Outputs",       "default": 0.600, "lo": 0.45, "hi": 0.75},
    262021: {"name": "Leaded Gasoline Residues",         "cat": "Slag",                 "default": 0.400, "lo": 0.20, "hi": 0.60},
    262029: {"name": "Other Lead-Containing Residues",   "cat": "Slag",                 "default": 0.550, "lo": 0.30, "hi": 0.80},
    282410: {"name": "Lead Monoxide",                    "cat": "Battery Inputs",       "default": 0.930, "lo": 0.93, "hi": 0.93},
    282490: {"name": "Other Lead Oxides",                "cat": "Battery Inputs",       "default": 0.900, "lo": 0.87, "hi": 0.93},
    780110: {"name": "Refined Unwrought Lead",           "cat": "Battery Inputs",       "default": 0.990, "lo": 0.99, "hi": 0.99},
    780191: {"name": "Unwrought Lead (w/ Antimony)",     "cat": "Battery Inputs",       "default": 0.960, "lo": 0.94, "hi": 0.98},
    780199: {"name": "Other Unwrought Lead",             "cat": "Battery Inputs",       "default": 0.970, "lo": 0.95, "hi": 0.99},
    780200: {"name": "Lead Waste & Scrap",               "cat": "Battery Waste",        "default": 0.725, "lo": 0.50, "hi": 0.95},
    780411: {"name": "Lead Sheet/Strip/Foil (PCB)",      "cat": "Other Lead Products",  "default": 0.990, "lo": 0.99, "hi": 0.99},
    780419: {"name": "Lead Sheet/Strip/Foil (Other)",    "cat": "Other Lead Products",  "default": 0.990, "lo": 0.99, "hi": 0.99},
    780420: {"name": "Lead Tubes, Pipes & Fittings",     "cat": "Other Lead Products",  "default": 0.985, "lo": 0.98, "hi": 0.99},
    780600: {"name": "Other Lead Articles",              "cat": "Other Lead Products",  "default": 0.970, "lo": 0.95, "hi": 0.99},
    850710: {"name": "Lead-Acid Starter Batteries",      "cat": "New Batteries",        "default": 0.600, "lo": 0.55, "hi": 0.65},
    850720: {"name": "Other Lead-Acid Batteries",        "cat": "New Batteries",        "default": 0.600, "lo": 0.55, "hi": 0.65},
    850790: {"name": "Battery Parts",                    "cat": "Battery Inputs",       "default": 0.625, "lo": 0.30, "hi": 0.95},
    854810: {"name": "Used Lead-Acid Batteries",         "cat": "Battery Waste",        "default": 0.600, "lo": 0.55, "hi": 0.65},
    854911: {"name": "Spent Lead-Acid Battery Waste",    "cat": "Battery Waste",        "default": 0.600, "lo": 0.55, "hi": 0.65},
}

CATEGORIES_ORDERED: list[str] = [
    "Mining Outputs",
    "Battery Inputs",
    "New Batteries",
    "Battery Waste",
    "Slag",
    "Other Lead Products",
]

_CAT_COLORS: dict[str, str] = {
    "Mining Outputs":      "#78909C",
    "Battery Inputs":      "#1E88E5",
    "New Batteries":       "#43A047",
    "Battery Waste":       "#F9A825",
    "Slag":                "#8D6E63",
    "Other Lead Products": "#7E57C2",
}

# HS codes excluded from the calculation by default (user can re-enable).
# Slag (262021/262029) and Other Lead Products (780411/780419/780420/780600).
_DEFAULT_OFF: set[int] = {262021, 262029, 780411, 780419, 780420, 780600}

# ── Easy-mode (5-category) grouping ───────────────────────────────────────────
# Mirrors the five trade categories used on the Trade tabs. Battery Waste is
# split into Used Batteries + Lead Scrap; Battery Inputs -> Smelted Lead;
# Mining Outputs -> Ore & Concentrates.
_EASY_CATS_ORDERED: list[str] = [
    "Ore & Concentrates", "Smelted Lead", "New Batteries",
    "Used Batteries", "Lead Scrap",
]
_EASY_CAT_COLORS: dict[str, str] = {
    "Ore & Concentrates": "#9E9E9E",  # grey
    "Smelted Lead":       "#1E88E5",  # blue
    "New Batteries":      "#43A047",  # green
    "Used Batteries":     "#FDD835",  # yellow
    "Lead Scrap":         "#FB8C00",  # orange
}
_HS_EASY_CAT: dict[int, str] = {
    260700: "Ore & Concentrates",
    282410: "Smelted Lead", 282490: "Smelted Lead",
    780110: "Smelted Lead", 780191: "Smelted Lead",
    780199: "Smelted Lead", 850790: "Smelted Lead",
    850710: "New Batteries", 850720: "New Batteries",
    854810: "Used Batteries", 854911: "Used Batteries",
    780200: "Lead Scrap",
}

# Per-grouping settings: (HS->category map, ordered category list, mining-label).
# "advanced" uses the six material-flow categories from HS_META; the mine
# production folds into "Mining Outputs". "easy" uses the five trade categories;
# mine production folds into "Ore & Concentrates".
def _grouping_config(grouping: str):
    if grouping == "easy":
        return (
            lambda hs: _HS_EASY_CAT.get(int(hs)),
            _EASY_CATS_ORDERED,
            _EASY_CAT_COLORS,
            "Ore & Concentrates",
        )
    return (
        lambda hs: HS_META.get(int(hs), {}).get("cat", "Other"),
        CATEGORIES_ORDERED,
        _CAT_COLORS,
        "Mining Outputs",
    )

# mined.csv (USGS) country names → BACI country names
_MINING_TO_BACI: dict[str, str] = {
    "Republic Of Korea": "Rep. of Korea",
    "Burma":             "Myanmar",
    "North Korea":       "Dem. People's Rep. of Korea",
    "Macedonia":         "North Macedonia",
    "Russia":            "Russian Federation",
    "Laos":              "Lao People's Dem. Rep.",
    "Vietnam":           "Viet Nam",
    "Bolivia":           "Bolivia (Plurinational State of)",
}

# BGS country_trans names → BACI country names
_BGS_TO_BACI_MINING: dict[str, str] = {
    "Bolivia":                    "Bolivia (Plurinational State of)",
    "Bosnia & Herzegovina":       "Bosnia Herzegovina",
    "Congo, Democratic Republic": "Dem. Rep. of the Congo",
    "Czech Republic":             "Czechia",
    "Dominican Republic":         "Dominican Rep.",
    "Ireland, Republic of":       "Ireland",
    "Korea (Rep. of)":            "Rep. of Korea",
    "Korea, Dem. P.R. of":        "Dem. People's Rep. of Korea",
    "Laos":                       "Lao People's Dem. Rep.",
    "Russia":                     "Russian Federation",
    "Tanzania":                   "United Rep. of Tanzania",
    "Trinidad & Tobago":          "Trinidad and Tobago",
    "Turkey":                     "Türkiye",
    "Vietnam":                    "Viet Nam",
}


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading expanded trade data...")
def load_master_baci(dataset: str = "hs12") -> pd.DataFrame:
    """Load the full-lead-universe BACI file (17 HS codes for hs12, 20 for hs22)."""
    if dataset == "hs22":
        path = DATA_DIR / "BACI_HS22_lead_trade_2022_2024.csv"
        df = pd.read_csv(path)
    else:
        path = DATA_DIR / "BACI_lead_trade_2012_2024_modified_vHS_4_master.csv"
        df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for col in ("Exporter", "Importer"):
        df[col] = df[col].replace({"TÃƒÂ¼rkiye": "Türkiye", "TÃ¼rkiye": "Türkiye"})
    return df


@st.cache_data(show_spinner=False)
def load_mined() -> pd.DataFrame:
    """Load USGS mine-production data (2015–2023), mapping country names to BACI convention."""
    path = DATA_DIR / "Reference" / "USGS mined.csv"
    if not path.exists():
        path = DATA_DIR / "mined.csv"
    if not path.exists():
        return pd.DataFrame(columns=["country", "year", "value_metric_t", "country_baci"])
    df = pd.read_csv(path)
    df["country_baci"] = df["country"].map(lambda c: _MINING_TO_BACI.get(c, c))
    return df


@st.cache_data(show_spinner=False)
def load_bgs_mined() -> pd.DataFrame:
    """Load BGS mine-production data (1971–2023), harmonised to (country_baci, year, value_metric_t)."""
    path = DATA_DIR / "Reference" / "BGS Refined and Smelted.csv"
    if not path.exists():
        path = DATA_DIR / "BGS Refined and Smelted.csv"
    if not path.exists():
        return pd.DataFrame(columns=["country_baci", "year", "value_metric_t"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    mine = df[df["bgs_commodity_trans"] == "lead, mine"][["country_trans", "year", "Mass of Pb"]].copy()
    mine = mine.rename(columns={"Mass of Pb": "value_metric_t"})
    mine["country_baci"] = mine["country_trans"].map(lambda c: _BGS_TO_BACI_MINING.get(c, c))
    return mine[["country_baci", "year", "value_metric_t"]].dropna(subset=["value_metric_t"])


# ── Core computation ──────────────────────────────────────────────────────────

def compute_balance(
    baci: pd.DataFrame,
    mined: pd.DataFrame,
    selection: str,
    is_region: bool,
    region_map: dict[str, str],
    pb_factors: dict[int, float],
    grouping: str = "advanced",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute annual net lead balance and HS-level detail.

    Returns
    -------
    annual_df : Year | Mining | Lead Feedstock | ... | Total  (tonnes Pb)
    hs_year_df: Year | Product | hs_name | category | imports_t | exports_t | net_t  (tonnes Pb)
    """
    # ── Row selection ─────────────────────────────────────────────────────────
    if is_region:
        rc = {c for c, r in region_map.items() if r == selection}
        imp_mask = baci["Importer"].isin(rc) & ~baci["Exporter"].isin(rc)
        exp_mask = baci["Exporter"].isin(rc) & ~baci["Importer"].isin(rc)
        mine_mask = mined["country_baci"].isin(rc)
    else:
        imp_mask = baci["Importer"] == selection
        exp_mask = baci["Exporter"] == selection
        mine_mask = mined["country_baci"] == selection

    imp = baci[imp_mask].copy()
    exp = baci[exp_mask].copy()
    mine = mined[mine_mask].copy()

    # ── Convert raw trade quantities to lead content ──────────────────────────
    imp["pb_t"] = imp["Product"].map(pb_factors).fillna(0.70) * imp["Quantity"]
    exp["pb_t"] = exp["Product"].map(pb_factors).fillna(0.70) * exp["Quantity"]

    # ── HS × year aggregation ─────────────────────────────────────────────────
    imp_agg = (
        imp.groupby(["Year", "Product"])["pb_t"].sum()
        .reset_index().rename(columns={"pb_t": "imports_t"})
    )
    exp_agg = (
        exp.groupby(["Year", "Product"])["pb_t"].sum()
        .reset_index().rename(columns={"pb_t": "exports_t"})
    )

    hs_year = imp_agg.merge(exp_agg, on=["Year", "Product"], how="outer").fillna(0)
    hs_year["hs_name"]  = hs_year["Product"].map(lambda x: HS_META.get(int(x), {}).get("name", str(x)))
    # hs_year.category stays on the 6-cat scheme (drives the Advanced HS table).
    hs_year["category"] = hs_year["Product"].map(lambda x: HS_META.get(int(x), {}).get("cat", "Other"))
    hs_year["net_t"]    = hs_year["imports_t"] - hs_year["exports_t"]

    # Active grouping (5-cat Easy or 6-cat Advanced) drives the annual bucketing.
    cat_of, cats_ordered, _grp_colors, mining_label = _grouping_config(grouping)

    # ── Annual BGS mining ─────────────────────────────────────────────────────
    mining_annual = (
        mine.groupby("year")["value_metric_t"].sum()
        .reset_index()
        .rename(columns={"year": "Year", "value_metric_t": "_bgs"})
    )

    # ── Annual category net totals (from trade), bucketed per active grouping ──
    _grp = hs_year.assign(_gcat=hs_year["Product"].map(cat_of)).dropna(subset=["_gcat"])
    cat_annual = (
        _grp.groupby(["Year", "_gcat"])["net_t"]
        .sum().unstack(fill_value=0).reset_index()
    )

    # ── Assemble annual_df ────────────────────────────────────────────────────
    all_years = sorted(int(y) for y in baci["Year"].unique())
    annual = pd.DataFrame({"Year": all_years})
    annual = annual.merge(cat_annual, on="Year", how="left").fillna(0)
    annual = annual.merge(mining_annual, on="Year", how="left").fillna({"_bgs": 0.0})

    # Mining category = BGS/USGS mine production + net ore trade (260700)
    if mining_label in annual.columns:
        annual[mining_label] = annual["_bgs"] + annual[mining_label]
    else:
        annual[mining_label] = annual["_bgs"]
    annual = annual.drop(columns=["_bgs"])

    for cat in cats_ordered:
        if cat not in annual.columns:
            annual[cat] = 0.0

    trade_cats = [c for c in cats_ordered if c != mining_label]
    annual["Total"] = annual[mining_label] + annual[trade_cats].sum(axis=1)
    annual = (
        annual[["Year", mining_label] + trade_cats + ["Total"]]
        .sort_values("Year").reset_index(drop=True)
    )

    return annual, hs_year


# ── Chart builders ────────────────────────────────────────────────────────────

_BASE_YEAR = 2012
_INCOMPLETE_NOTE = dict(
    text="⚠ 2024 data incomplete",
    x=0.99, y=0.02, xref="paper", yref="paper",
    showarrow=False, align="right",
    font=dict(size=11, color="#E65100"),
    bgcolor="rgba(255,255,255,0.8)",
    borderpad=3,
)


def _cumulative_line_chart(annual_df: pd.DataFrame, sidebar_year: int, label: str) -> go.Figure:
    """Cumulative net lead balance since _BASE_YEAR (base = 0)."""
    df = annual_df.sort_values("Year")
    years = df["Year"].tolist()
    totals_kt = [v / 1000 for v in df["Total"].tolist()]

    # Build cumulative: base year = 0; each subsequent year adds its annual net
    cumulative: list[float | None] = []
    running = 0.0
    for y, net in zip(years, totals_kt):
        if y < _BASE_YEAR:
            cumulative.append(None)
        elif y == _BASE_YEAR:
            cumulative.append(0.0)
        else:
            running += net
            cumulative.append(running)

    # Confine displayed values to 2 significant figures (model-derived).
    cumulative = [None if v is None else _sig2(v) for v in cumulative]

    fig = go.Figure()

    # Green / red area fills
    fig.add_trace(go.Scatter(
        x=years, y=[max(0.0, v) if v is not None else None for v in cumulative],
        mode="none", fill="tozeroy",
        fillcolor="rgba(67,160,71,0.15)", showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=[min(0.0, v) if v is not None else None for v in cumulative],
        mode="none", fill="tozeroy",
        fillcolor="rgba(229,57,53,0.15)", showlegend=False, hoverinfo="skip",
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=years, y=cumulative,
        mode="lines+markers",
        line=dict(color="#1E88E5", width=2.5),
        marker=dict(size=5),
        name="Cumulative net",
        hovertemplate="%{x}: %{y:+,.1f} kt Pb cumulative<extra></extra>",
    ))

    # Sidebar-year highlight
    if sidebar_year in years:
        idx = years.index(sidebar_year)
        val = cumulative[idx]
        if val is not None:
            fig.add_vline(x=sidebar_year, line_dash="dot", line_color="#FB8C00", line_width=1.5)
            fig.add_trace(go.Scatter(
                x=[sidebar_year], y=[val],
                mode="markers",
                marker=dict(size=14, color="#FB8C00", symbol="diamond"),
                name=f"{sidebar_year} (detail)",
                hovertemplate=f"{sidebar_year}: %{{y:+,.1f}} kt Pb cumulative<extra></extra>",
            ))

    fig.add_hline(y=0, line_color="#555", line_width=1)

    fig.update_layout(
        title=dict(text=f"Cumulative Net Lead Balance Since {_BASE_YEAR} — {label}", font_size=15),
        xaxis=dict(title="Year", tickmode="linear", dtick=1),
        yaxis_title="Cumulative Net Lead (kt Pb)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=40),
        plot_bgcolor="#fafafa",
        annotations=[
            dict(
                text=f"Base year = {_BASE_YEAR} (0)",
                x=_BASE_YEAR, y=0,
                xanchor="left", yanchor="bottom",
                showarrow=True, arrowhead=2, arrowsize=0.8,
                ax=25, ay=-35,
                font=dict(size=11, color="#555"),
            ),
            _INCOMPLETE_NOTE,
        ],
    )
    return fig


def _annual_bar_chart(annual_df: pd.DataFrame, sidebar_year: int, label: str) -> go.Figure:
    """Annual net lead balance as colored bars (green = gain, red = loss)."""
    df = annual_df.sort_values("Year")
    years = df["Year"].tolist()
    totals_kt = [_sig2(v / 1000) for v in df["Total"].tolist()]

    colors = [
        "#FB8C00" if y == sidebar_year else ("#43A047" if v >= 0 else "#E53935")
        for y, v in zip(years, totals_kt)
    ]

    fig = go.Figure(go.Bar(
        x=years, y=totals_kt,
        marker_color=colors,
        hovertemplate="%{x}: %{y:+,.1f} kt Pb<extra></extra>",
        name="Annual net",
    ))

    fig.add_hline(y=0, line_color="#555", line_width=1)

    fig.update_layout(
        title=dict(text=f"Annual Net Lead Balance — {label}", font_size=15),
        xaxis=dict(title="Year", tickmode="linear", dtick=1),
        yaxis_title="Net Lead (kt Pb)",
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=40),
        plot_bgcolor="#fafafa",
        annotations=[_INCOMPLETE_NOTE],
    )
    return fig


def _bar_chart(
    annual_df: pd.DataFrame,
    selected_year: int,
    label: str,
    cats_ordered: list[str] | None = None,
    cat_colors: dict[str, str] | None = None,
) -> go.Figure:
    cats_ordered = cats_ordered or CATEGORIES_ORDERED
    cat_colors = cat_colors or _CAT_COLORS
    row = annual_df[annual_df["Year"] == selected_year]
    if row.empty:
        return go.Figure()

    fig = go.Figure()
    total_kt = 0.0

    for cat in cats_ordered:
        val = float(row[cat].values[0]) if cat in row.columns else 0.0
        val_kt = _sig2(val / 1000)
        total_kt += val_kt
        fig.add_trace(go.Bar(
            x=[cat], y=[val_kt],
            name=cat,
            marker_color=cat_colors.get(cat, "#9E9E9E"),
            hovertemplate=f"<b>{cat}</b><br>%{{y:+,.2f}} kt Pb<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="#555", line_width=1)

    annotations = [
        dict(
            text=f"Total net: {_fmt_kt(total_kt)} kt Pb",
            x=0.99, y=0.97, xref="paper", yref="paper",
            showarrow=False, align="right",
            font=dict(size=13, color="#333"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#aaa", borderwidth=1,
        ),
    ]
    if selected_year == 2024:
        annotations.append(dict(
            text="⚠ 2024 data incomplete",
            x=0.99, y=0.87, xref="paper", yref="paper",
            showarrow=False, align="right",
            font=dict(size=11, color="#E65100"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=3,
        ))

    fig.update_layout(
        title=dict(text=f"Lead Gains & Losses by Category — {label}, {selected_year}", font_size=15),
        xaxis_title="",
        yaxis_title="Net Lead (kt Pb)",
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=80),
        plot_bgcolor="#fafafa",
        annotations=annotations,
    )
    return fig


def _net_balance_world_map(
    baci: pd.DataFrame,
    mined_bgs: pd.DataFrame,
    mined_usgs: pd.DataFrame,
    mining_source: str,
    year: int,
    pb_factors: dict[int, float],
) -> go.Figure:
    """
    Choropleth of every country's net lead balance for *year*:
        net = mine production + Pb imports - Pb exports   (tonnes Pb -> kt Pb)
    Diverging red/green scale (red = net loser, green = net gainer), centered on
    zero and clipped at the 95th percentile of |value| for contrast.
    """
    t = baci[baci["Year"] == year].copy()
    if t.empty:
        return go.Figure()
    t["pb_t"] = t["Product"].map(pb_factors).fillna(0.70) * t["Quantity"]
    imp = t.groupby("Importer")["pb_t"].sum()
    exp = t.groupby("Exporter")["pb_t"].sum()
    net = imp.subtract(exp, fill_value=0.0)

    # Mine production: preferred source, filled per-country from the other.
    pref, alt = (mined_bgs, mined_usgs) if mining_source == "BGS" else (mined_usgs, mined_bgs)
    pref_y = pref[pref["year"] == year].groupby("country_baci")["value_metric_t"].sum()
    alt_y = alt[alt["year"] == year].groupby("country_baci")["value_metric_t"].sum()
    mining = pref_y.combine_first(alt_y)

    total_kt = net.add(mining, fill_value=0.0) / 1000.0
    total_kt = total_kt[total_kt.abs() > 1e-9]
    if total_kt.empty:
        return go.Figure()

    countries = list(total_kt.index)
    vals = [_sig2(float(v)) for v in total_kt.values]
    _bound = float(np.percentile(np.abs(vals), 95))
    if _bound <= 0:
        _bound = max(abs(min(vals)), abs(max(vals)), 1e-6)

    fig = go.Figure(go.Choropleth(
        locations=countries,
        z=vals,
        locationmode="country names",
        colorscale="RdYlGn",
        zmid=0, zmin=-_bound, zmax=_bound,
        colorbar=dict(title="kt Pb", len=0.7, thickness=14),
        hovertemplate="%{location}: %{z:+,.1f} kt Pb<extra></extra>",
        marker_line_color="#ffffff", marker_line_width=0.4,
    ))
    fig.update_layout(
        title=dict(text=f"Net Lead Balance by Country — {year}", x=0.5, font=dict(size=14)),
        height=430,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        geo=dict(
            showframe=False, showcoastlines=True, coastlinecolor="#aaaaaa",
            showland=True, landcolor="#d8d8d8", showocean=True, oceancolor="#e3f2fd",
            showlakes=False, projection_type="natural earth",
        ),
    )
    return fig


# ── Table builder ─────────────────────────────────────────────────────────────

def _build_table(
    hs_year_df: pd.DataFrame,
    mined: pd.DataFrame,
    selection: str,
    is_region: bool,
    region_map: dict[str, str],
    year: int,
) -> pd.DataFrame:
    """
    Build the HS-level breakdown table for *year*.

    Columns: Category | HS Code | Description | Imports (kt Pb) | Exports (kt Pb) | Net (kt Pb) | _is_subtotal
    """
    year_hs = hs_year_df[hs_year_df["Year"] == year].copy()

    # Lookup BGS mining for this specific year
    if is_region:
        rc = {c for c, r in region_map.items() if r == selection}
        mine_sel = mined[mined["country_baci"].isin(rc)]
    else:
        mine_sel = mined[mined["country_baci"] == selection]
    bgs_mining = float(mine_sel[mine_sel["year"] == year]["value_metric_t"].sum())

    rows: list[dict] = []

    for cat in CATEGORIES_ORDERED:
        cat_hs = year_hs[year_hs["category"] == cat]

        # BGS mine production row (Mining Outputs category only)
        if cat == "Mining Outputs" and bgs_mining > 0:
            rows.append({
                "Category": cat,
                "HS Code": "—",
                "Description": "Mine Production (BGS/USGS)",
                "Imports (kt Pb)": round(bgs_mining / 1000, 3),
                "Exports (kt Pb)": 0.0,
                "Net (kt Pb)": round(bgs_mining / 1000, 3),
                "_is_subtotal": False,
            })

        # Individual HS code rows (skip if both round to 0.000 kt at 3 dp)
        for _, hs_row in cat_hs.iterrows():
            if round(hs_row["imports_t"] / 1000, 3) == 0 and round(hs_row["exports_t"] / 1000, 3) == 0:
                continue
            rows.append({
                "Category": cat,
                "HS Code": str(int(hs_row["Product"])),
                "Description": hs_row["hs_name"],
                "Imports (kt Pb)": round(hs_row["imports_t"] / 1000, 3),
                "Exports (kt Pb)": round(hs_row["exports_t"] / 1000, 3),
                "Net (kt Pb)": round(hs_row["net_t"] / 1000, 3),
                "_is_subtotal": False,
            })

        # Category subtotal
        cat_imp = float(cat_hs["imports_t"].sum()) + (bgs_mining if cat == "Mining Outputs" else 0.0)
        cat_exp = float(cat_hs["exports_t"].sum())
        cat_net = float(cat_hs["net_t"].sum()) + (bgs_mining if cat == "Mining Outputs" else 0.0)

        has_data = cat_imp > 0 or cat_exp > 0 or (cat == "Mining Outputs" and bgs_mining > 0)
        if has_data:
            rows.append({
                "Category": cat,
                "HS Code": "",
                "Description": f"{cat} — Subtotal",
                "Imports (kt Pb)": round(cat_imp / 1000, 3),
                "Exports (kt Pb)": round(cat_exp / 1000, 3),
                "Net (kt Pb)": round(cat_net / 1000, 3),
                "_is_subtotal": True,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def _style_table(df: pd.DataFrame):
    """
    Return a pandas Styler: green/red gradient on Net column; gray bold on subtotals.
    Darker = larger magnitude.
    """
    net_col = "Net (kt Pb)"
    display = df.drop(columns=["_is_subtotal"]).copy()

    max_gain = max(float(df[net_col].max()), 1e-6)
    max_loss = max(float(abs(df[net_col].min())), 1e-6)

    # Capture subtotal mask via closure
    is_sub = df["_is_subtotal"].values

    def _net_color(val: float) -> str:
        if val > 0:
            t = min(1.0, val / max_gain)
            r = int(232 - t * (232 - 27))
            g = int(245 - t * (245 - 94))
            b = int(233 - t * (233 - 32))
            fg = "white" if t > 0.55 else "#333"
            return f"background-color: rgb({r},{g},{b}); color: {fg}; font-weight: 600"
        if val < 0:
            t = min(1.0, abs(val) / max_loss)
            r = int(255 - t * (255 - 183))
            g = int(235 - t * (235 - 28))
            b = int(238 - t * (238 - 28))
            fg = "white" if t > 0.55 else "#333"
            return f"background-color: rgb({r},{g},{b}); color: {fg}; font-weight: 600"
        return ""

    def _row_style(row: "pd.Series") -> list[str]:
        n = len(display.columns)
        if is_sub[row.name]:
            return ["background-color: #ECEFF1; font-weight: bold; border-top: 1px solid #90A4AE"] * n
        return [""] * n

    styler = (
        display.style
        .map(_net_color, subset=[net_col])
        .apply(_row_style, axis=1)
        .format({
            "Imports (kt Pb)": "{:,.3f}",
            "Exports (kt Pb)": "{:,.3f}",
            net_col: "{:+,.3f}",
        })
        .set_properties(**{"font-size": "13px"})
        .hide(axis="index")
    )
    return styler


# ── Main render function ──────────────────────────────────────────────────────

def render_lead_accumulation_tab(
    region_map: dict[str, str],
    regions_ordered: list[str],
    sidebar_year: int = 2022,
    dataset: str = "hs12",
    pb_factors: dict[int, float] | None = None,
    mining_source: str = "BGS",
    major_region_map: dict[str, str] | None = None,
    major_regions_ordered: list[str] | None = None,
    advanced: bool = True,
) -> None:
    """Entry point — call from within a `with tab:` block in streamlit_app.py."""

    master_baci  = load_master_baci(dataset)
    _mined_usgs  = load_mined()       # USGS 2015–2023
    _mined_bgs   = load_bgs_mined()   # BGS  1971–2023

    all_countries = sorted(
        set(master_baci["Exporter"].unique()) | set(master_baci["Importer"].unique())
    )

    # ── Country / region selector (regions first, then countries A-Z) ─────────
    _major = list(major_regions_ordered or [])
    _regions_all = _major + list(regions_ordered)
    _sel_options = _regions_all + all_countries
    sel_col, _ = st.columns([2, 3])
    with sel_col:
        _default_idx = _sel_options.index("Ghana") if "Ghana" in _sel_options else 0
        selection = st.selectbox(
            "Select country or region",
            _sel_options,
            index=_default_idx,
            key="accum_selection",
            help="Regions (continents and UN sub-regions) are listed first, then countries A-Z.",
        )
    is_region = selection in _regions_all
    # Resolve the region membership map (major regions use their own map).
    if is_region and selection in _major and major_region_map:
        region_map = major_region_map

    # ── Resolve mining dataset (preferred source with auto-fallback) ──────────
    _mine_countries = (
        {c for c, r in region_map.items() if r == selection} if is_region else {selection}
    )
    if pb_factors is None:
        pb_factors = {hs: meta["default"] for hs, meta in HS_META.items()}

    if mining_source == "BGS":
        if _mined_bgs["country_baci"].isin(_mine_countries).any():
            mined_df = _mined_bgs
            _mine_src_note: str | None = None
        elif _mined_usgs["country_baci"].isin(_mine_countries).any():
            mined_df = _mined_usgs
            _mine_src_note = (
                f"No BGS mine data found for **{selection}** — "
                "using USGS data (2015–2023) instead."
            )
        else:
            mined_df = _mined_bgs
            _mine_src_note = None
    else:  # USGS preferred
        if _mined_usgs["country_baci"].isin(_mine_countries).any():
            mined_df = _mined_usgs
            _mine_src_note = None
        elif _mined_bgs["country_baci"].isin(_mine_countries).any():
            mined_df = _mined_bgs
            _mine_src_note = (
                f"No USGS mine data found for **{selection}** — "
                "using BGS data (1971–2023) instead."
            )
        else:
            mined_df = _mined_usgs
            _mine_src_note = None

    if _mine_src_note:
        st.info(_mine_src_note)

    # ── Compute ───────────────────────────────────────────────────────────────
    _grouping = "advanced" if advanced else "easy"
    _cats_ordered, _cat_colors = (
        (CATEGORIES_ORDERED, _CAT_COLORS) if advanced
        else (_EASY_CATS_ORDERED, _EASY_CAT_COLORS)
    )
    annual_df, hs_year_df = compute_balance(
        master_baci, mined_df, selection, is_region, region_map, pb_factors,
        grouping=_grouping,
    )

    if annual_df.empty or annual_df["Total"].abs().sum() < 0.001:
        st.warning(f"No lead trade or mining data found for **{selection}**.")
        return

    # Clamp sidebar_year to available range
    all_years = annual_df["Year"].tolist()
    selected_year = max(min(sidebar_year, max(all_years)), min(all_years))

    # ── Chart 1: Cumulative net lead balance ──────────────────────────────────
    st.plotly_chart(
        _cumulative_line_chart(annual_df, selected_year, selection),
        use_container_width=True,
    )
    st.caption(
        f"Running total of annual net lead since {_BASE_YEAR} (base = 0). "
        "Positive = cumulative surplus; negative = cumulative deficit. "
        "Orange diamond marks the year selected in the sidebar."
    )

    # ── Annual net lead balance bar (Advanced only) ───────────────────────────
    if advanced:
        st.plotly_chart(
            _annual_bar_chart(annual_df, selected_year, selection),
            use_container_width=True,
        )
        st.caption(
            "Each bar shows one year's net lead balance (mining + imports − exports). "
            "Green = net inflow; red = net outflow. Orange = sidebar-selected year."
        )

    # ── Category breakdown (half) + global net-balance map (half) ─────────────
    _cat_col, _map_col = st.columns(2)
    with _cat_col:
        st.plotly_chart(
            _bar_chart(annual_df, selected_year, selection, _cats_ordered, _cat_colors),
            use_container_width=True,
        )
        _mining_lbl = "Mining Outputs" if advanced else "Ore & Concentrates"
        st.caption(
            "Each bar shows one category's net contribution to the total balance in the "
            f"selected year ({selected_year}, set via the sidebar Year slider). "
            f"{_mining_lbl} = BGS/USGS mine production + net ore (HS 260700) trade."
        )
    with _map_col:
        st.plotly_chart(
            _net_balance_world_map(
                master_baci, _mined_bgs, _mined_usgs, mining_source,
                selected_year, pb_factors,
            ),
            use_container_width=True,
        )
        st.caption(
            f"Net lead balance by country for {selected_year}: "
            "**green** = net gainer (more lead in than out), **red** = net loser. "
            "Scale clipped at the 95th percentile of magnitude for contrast."
        )

    # ── Key metrics + HS-code breakdown table (Advanced only) ─────────────────
    if advanced:
        yr_row = annual_df[annual_df["Year"] == selected_year]
        if not yr_row.empty:
            metric_cols = st.columns(len(CATEGORIES_ORDERED) + 1)
            total_val = float(yr_row["Total"].values[0])
            with metric_cols[0]:
                st.metric("Total Net", f"{_fmt_kt(total_val/1000)} kt Pb")
            for i, cat in enumerate(CATEGORIES_ORDERED):
                with metric_cols[i + 1]:
                    val = float(yr_row[cat].values[0]) if cat in yr_row.columns else 0.0
                    st.metric(cat[:12], f"{_fmt_kt(val/1000)} kt Pb")

        st.subheader(f"HS-Code Breakdown — {selection}, {selected_year}")
        table_df = _build_table(
            hs_year_df, mined_df, selection, is_region, region_map, selected_year
        )
        if table_df.empty:
            st.info("No trade or mining data for this selection in the selected year.")
        else:
            st.dataframe(
                _style_table(table_df),
                use_container_width=True,
                height=min(600, 40 + len(table_df) * 36),
            )
        st.caption(
            "Subtotal rows (bold gray) summarise each category. "
            "Net color: darker green = larger gain; darker red = larger loss. "
            "All values in kt Pb (kilotonnes of lead content)."
        )
