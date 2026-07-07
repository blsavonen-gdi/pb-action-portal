"""
streamlit_app.py — Pb Action Mass Balance Model
Main entry point. Tabs: Overview | Trade Analysis | Trade Trends | Flow Network |
Production & Capacity | Supply Chain Provenance | Lead Accumulation | Trade Composition.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from model.data_loader import load_baci, load_smelter_capacity
from visualizations.trade_map import (
    build_total_volume_map, build_bilateral_map, build_bilateral_balance_map,
    build_region_bilateral_map, build_region_balance_map,
)
from visualizations.provenance_map import build_provenance_map, get_provenance_tables
from visualizations.flow_network import build_flow_network
from visualizations.flow_network_interactive import build_flow_network_html
import streamlit.components.v1 as components
from model.regions import REGIONS_ORDERED, REGION_MAP, MAJOR_REGIONS_ORDERED, MAJOR_REGION_MAP
from visualizations.lead_accumulation import (
    render_lead_accumulation_tab, HS_META, CATEGORIES_ORDERED, _DEFAULT_OFF,
)
from visualizations.trade_composition_map import render_trade_composition_map
from visualizations.mass_balance_sankey import render_mass_balance_sankey_tab
from visualizations.india_calibration_explorer_tab import (
    render_india_calibration_explorer_tab,
)
from visualizations.india_v4_tab import render_india_v4_tab
from india_model.india_mass_balance import (
    load_india_csvs,
    build_net_trade as india_build_net_trade,
    prepare_model_inputs as india_prepare_inputs,
    forward_model as india_forward_model,
)

st.set_page_config(
    page_title="Global Lead Analysis Toolkit",
    page_icon="🔋",
    layout="wide",
)

# ── App-wide category color palette ───────────────────────────────────────────
# These colors are used consistently across multiselect pills, the provenance
# map, and the flow network arrows. They match the colored-square emoji.
CAT_COLORS = {
    "BATT":  "#43A047",  # 🟩 green  — New Batteries
    "USED":  "#FDD835",  # 🟨 yellow — Used Batteries
    "SCRAP": "#FB8C00",  # 🟧 orange — Lead Scrap
    "FEED":  "#1E88E5",  # 🟦 blue   — Smelted Lead
    "ORE":   "#9E9E9E",  # ⬜ grey   — Ore & Concentrates
}

st.markdown("""
<style>
/* ── Tab group visual separators ──────────────────────────────────────── */
/* Tab 6 = Supply Chain Provenance (first of Analytical Tools group)       */
/* Tab 8 = Trade Composition (first of Experimental group)                 */
/* Tab 9 = Mass Balance (also Experimental)                                */
[data-baseweb="tab-list"] button:nth-child(6),
[data-baseweb="tab-list"] button:nth-child(8) {
    margin-left: 10px !important;
    padding-left: 14px !important;
    border-left: 2px solid rgba(180, 180, 180, 0.5) !important;
}

/* ── Product-category multiselect pill colors ─────────────────────────── */
[data-baseweb="tag"]:has([aria-label*="Ore"]) {
    background-color: rgba(158,158,158,0.18) !important;
    border: 1px solid #9E9E9E !important;
}
[data-baseweb="tag"]:has([aria-label*="New Batteries"]) {
    background-color: rgba(67,160,71,0.15) !important;
    border: 1px solid #43A047 !important;
}
[data-baseweb="tag"]:has([aria-label*="Used Batteries"]) {
    background-color: rgba(253,216,53,0.20) !important;
    border: 1px solid #F9A825 !important;
}
[data-baseweb="tag"]:has([aria-label*="Lead Scrap"]) {
    background-color: rgba(251,140,0,0.15) !important;
    border: 1px solid #FB8C00 !important;
}
[data-baseweb="tag"]:has([aria-label*="Smelted Lead"]) {
    background-color: rgba(30,136,229,0.15) !important;
    border: 1px solid #1E88E5 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading trade data...")
def _load_baci_cached(dataset: str = "hs12") -> pd.DataFrame:
    return load_baci(dataset)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: id(df)})
def _build_flow_network_cached(
    baci_df: pd.DataFrame,
    active_years: tuple[int, ...],
    countries: tuple[str, ...],
    categories: tuple[str, ...],
    min_flow: float,
    show_imports: bool,
    show_exports: bool,
    focal_country: str | None,
    layout: str,
    focal_only: bool,
    hidden_pairs: frozenset,
    prune_isolated: bool,
):
    return build_flow_network(
        baci_df=baci_df,
        active_years=list(active_years),
        countries=list(countries),
        categories=list(categories),
        min_flow=min_flow,
        show_imports=show_imports,
        show_exports=show_exports,
        focal_country=focal_country,
        layout=layout,
        focal_only=focal_only,
        hidden_pairs=set(hidden_pairs),
        prune_isolated=prune_isolated,
    )


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: id(df)})
def _build_flow_network_html_cached(
    baci_df: pd.DataFrame,
    active_years: tuple[int, ...],
    countries: tuple[str, ...],
    categories: tuple[str, ...],
    min_flow: float,
    show_imports: bool,
    show_exports: bool,
    focal_country: str | None,
    layout: str,
    physics_enabled: bool,
    height_px: int,
    focal_only: bool,
    hidden_pairs: frozenset,
    prune_isolated: bool,
) -> str:
    return build_flow_network_html(
        baci_df=baci_df,
        active_years=list(active_years),
        countries=list(countries),
        categories=list(categories),
        min_flow=min_flow,
        show_imports=show_imports,
        show_exports=show_exports,
        focal_country=focal_country,
        layout=layout,
        physics_enabled=physics_enabled,
        height_px=height_px,
        focal_only=focal_only,
        hidden_pairs=set(hidden_pairs),
        prune_isolated=prune_isolated,
    )


_BGS_TO_BACI_NAME: dict[str, str] = {
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
_BACI_TO_BGS_NAME: dict[str, str] = {v: k for k, v in _BGS_TO_BACI_NAME.items()}

_BACI_TO_USGS_NAME: dict[str, str] = {
    "Rep. of Korea":                    "Republic Of Korea",
    "Myanmar":                          "Burma",
    "Dem. People's Rep. of Korea":      "North Korea",
    "North Macedonia":                  "Macedonia",
    "Russian Federation":               "Russia",
    "Lao People's Dem. Rep.":           "Laos",
    "Viet Nam":                         "Vietnam",
    "Bolivia (Plurinational State of)": "Bolivia",
}
# USGS country names → Plotly-recognised display names (for choropleth)
_USGS_TO_DISPLAY: dict[str, str] = {
    "Burma":             "Myanmar",
    "Republic Of Korea": "South Korea",
    "Macedonia":         "North Macedonia",
    "Russia":            "Russian Federation",
}


@st.cache_data(show_spinner=False)
def _load_bgs() -> pd.DataFrame:
    import os
    path = "data/Reference/BGS Refined and Smelted.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["year", "country_trans", "bgs_commodity_trans", "Mass of Pb"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df[["year", "country_trans", "bgs_commodity_trans", "Mass of Pb"]].copy()


@st.cache_data(show_spinner=False)
def _load_capacity_cached() -> pd.DataFrame:
    return load_smelter_capacity()


@st.cache_data(show_spinner=False)
def _load_usgs_mined() -> pd.DataFrame:
    import os
    path = "data/Reference/USGS mined.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["country", "year", "value_metric_t"])
    return pd.read_csv(path, usecols=["country", "year", "value_metric_t"])


@st.cache_data(show_spinner=False)
def _load_usgs_refined() -> pd.DataFrame:
    import os
    if not os.path.exists("data/Reference/USGS refined_primary.csv") or not os.path.exists("data/Reference/USGS refined_secondary.csv"):
        return pd.DataFrame(columns=["country", "year", "value_metric_t"])
    rp = pd.read_csv("data/Reference/USGS refined_primary.csv", usecols=["country", "year", "value_metric_t"])
    rs = pd.read_csv("data/Reference/USGS refined_secondary.csv", usecols=["country", "year", "value_metric_t"])
    return pd.concat([rp, rs]).groupby(["country", "year"])["value_metric_t"].sum().reset_index()


@st.cache_data(show_spinner=False)
def _load_india_csvs_cached() -> dict:
    return load_india_csvs()


@st.cache_data(show_spinner=False)
def _build_india_net_trade_cached(_baci_df: pd.DataFrame) -> tuple:
    return india_build_net_trade(_baci_df)


@st.cache_data(show_spinner=False)
def _load_mining_refining() -> pd.DataFrame:
    return pd.read_csv("data/country_year_mining_refining.csv")


@st.cache_data(show_spinner="Computing model estimates…")
def _compute_model_map_data(
    baci_df: pd.DataFrame,
    mining_df: pd.DataFrame,
    year: int,
    dataset: str,
    pb_factors_tuple: tuple,
    mining_source: str,
    eta_secondary: float,
    eta_break: float,
    delta_pb: float,
    beta: float,
    eta_mfg: float,
    eta_ore: float,
    gamma: float,
) -> dict[str, dict]:
    """
    Run the mass-balance model for every eligible country for a single year.
    Returns {country: model_outputs_dict}.  Cached by all parameters.
    """
    from visualizations.mass_balance_sankey import build_sankey, _load_master_baci

    pb_factors = dict(pb_factors_tuple)

    refining_cols = ["refined_bgs_t", "refined_primary_usgs_t", "refined_secondary_usgs_t"]
    has_refining = mining_df[
        [c for c in refining_cols if c in mining_df.columns]
    ].notna().any(axis=1)
    eligible = sorted(set(mining_df.loc[has_refining, "country"].unique()))

    master_df = _load_master_baci() if dataset == "hs12" else pd.DataFrame(
        columns=["Year", "Exporter", "Importer", "Product", "Quantity"]
    )

    results: dict[str, dict] = {}
    for country in eligible:
        try:
            _, outputs, warn = build_sankey(
                baci_df       = baci_df,
                master_df     = master_df,
                mining_df     = mining_df,
                country       = country,
                active_years  = [year],
                dataset       = dataset,
                pb_factors    = pb_factors,
                mining_source = mining_source,
                eta_secondary = eta_secondary,
                eta_break     = eta_break,
                delta_pb      = delta_pb,
                beta          = beta,
                eta_mfg       = eta_mfg,
                eta_ore       = eta_ore,
                gamma         = gamma,
                min_flow      = 0.0,
            )
            if warn != "__no_refining__":
                results[country] = outputs
        except Exception:
            pass
    return results


@st.cache_data(show_spinner=False)
def _load_eurostat_collection() -> dict:
    from model.data_loader import load_eurostat_collection
    return load_eurostat_collection()


_dataset = st.session_state.get("baci_dataset", "HS12 (2012–2024)")
_dataset_key = "hs22" if _dataset.startswith("HS22") else "hs12"
baci_df = _load_baci_cached(_dataset_key)

all_years = sorted(baci_df["Year"].unique().tolist())
min_year, max_year = all_years[0], all_years[-1]

# All countries appearing in BACI trade data (exporters or importers)
all_baci_countries = sorted(
    set(baci_df["Exporter"].unique()) | set(baci_df["Importer"].unique())
)


# ── First-visit disclaimer ────────────────────────────────────────────────────

if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    _d_logo_col, _d_title_col = st.columns([1, 9])
    with _d_logo_col:
        st.image("Pb Action Logo - Primary.png", width=90)
    with _d_title_col:
        st.markdown("## Global Lead Analysis Toolkit")
        st.caption("Partnership for Battery Action (Pb Action / GDI)")
    st.divider()
    st.markdown("### ⚠ A note on data quality before you begin")
    st.write(
        "These tools draw on large, multi-source datasets — BACI trade statistics, "
        "BGS/USGS production figures, and model-derived estimates — each with known gaps "
        "and reporting errors.\n\n"
        "The results are useful for **high-level analysis, rough estimates, and pattern "
        "detection**, but should be treated as the **beginning of research, not the end**. "
        "Cross-check important findings against primary sources before drawing conclusions.\n\n"
        "This toolkit is actively evolving — we are continuously adding data, refining methods, "
        "and building new analysis tools, so check back regularly. We welcome feedback and "
        "contributions: reach out to [ben.savonen@globaldevincubator.org](mailto:ben.savonen@globaldevincubator.org)."
    )
    st.divider()
    if st.button("I understand — enter the app", type="primary"):
        st.session_state.disclaimer_accepted = True
        st.rerun()
    st.stop()

# ── Tab layout ────────────────────────────────────────────────────────────────

_logo_col, _title_col = st.columns([1, 9])
with _logo_col:
    st.image("Pb Action Logo - Primary.png", width=90)
with _title_col:
    st.title("Global Lead Analysis Toolkit")
    st.caption(
        "Partnership for Battery Action (Pb Action / GDI). "
        "All values in metric tonnes of lead content."
    )

_LIFECYCLE_SVG = """
<div style="margin: 4px 0 10px 0;">
<p style="font-size:12px;color:#888;margin:0 0 6px 0;">Lead changes product forms across its lifecycle. These tools track each stage by <em>lead content</em> — the tonnes of lead embedded in each product, not the gross weight.</p>
<svg viewBox="0 0 810 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;max-height:115px;">
  <defs>
    <marker id="arr" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#666"/>
    </marker>
    <marker id="arr-blue" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#1E88E5"/>
    </marker>
  </defs>

  <!-- Box 1: Ore & Concentrates -->
  <rect x="1" y="28" width="118" height="44" rx="5" fill="rgba(158,158,158,0.15)" stroke="#9E9E9E" stroke-width="1.5"/>
  <text x="60" y="45" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#555">Ore &amp;</text>
  <text x="60" y="59" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#555">Concentrates</text>

  <!-- Arrow 1: Primary Smelting -->
  <line x1="120" y1="50" x2="165" y2="50" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="143" y="19" text-anchor="middle" font-size="10" font-family="sans-serif" fill="#666">Primary</text>
  <text x="143" y="30" text-anchor="middle" font-size="10" font-family="sans-serif" fill="#666">Smelting</text>

  <!-- Box 2: Smelted Lead -->
  <rect x="168" y="28" width="118" height="44" rx="5" fill="rgba(30,136,229,0.12)" stroke="#1E88E5" stroke-width="1.5"/>
  <text x="227" y="52" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#555">Smelted Lead</text>

  <!-- Arrow 2: Manufacturing -->
  <line x1="287" y1="50" x2="332" y2="50" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="310" y="23" text-anchor="middle" font-size="10" font-family="sans-serif" fill="#666">Manu-</text>
  <text x="310" y="34" text-anchor="middle" font-size="10" font-family="sans-serif" fill="#666">facturing</text>

  <!-- Box 3: New Batteries -->
  <rect x="335" y="28" width="118" height="44" rx="5" fill="rgba(67,160,71,0.12)" stroke="#43A047" stroke-width="1.5"/>
  <text x="394" y="45" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#555">New</text>
  <text x="394" y="59" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#555">Batteries</text>

  <!-- Arrow 3: Use -->
  <line x1="454" y1="50" x2="499" y2="50" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="477" y="34" text-anchor="middle" font-size="10" font-family="sans-serif" fill="#666">Use</text>

  <!-- Box 4: Used Batteries -->
  <rect x="502" y="28" width="118" height="44" rx="5" fill="rgba(253,216,53,0.18)" stroke="#F9A825" stroke-width="1.5"/>
  <text x="561" y="45" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#555">Used</text>
  <text x="561" y="59" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#555">Batteries</text>

  <!-- Arrow 4: Breaking -->
  <line x1="621" y1="50" x2="666" y2="50" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="644" y="34" text-anchor="middle" font-size="10" font-family="sans-serif" fill="#666">Breaking</text>

  <!-- Box 5: Lead Scrap -->
  <rect x="669" y="28" width="118" height="44" rx="5" fill="rgba(251,140,0,0.12)" stroke="#FB8C00" stroke-width="1.5"/>
  <text x="728" y="52" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#555">Lead Scrap</text>

  <!-- Secondary Smelting arc (Lead Scrap → Smelted Lead, curves below) -->
  <path d="M 728,72 C 728,113 227,113 227,72" fill="none" stroke="#1E88E5" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arr-blue)"/>
  <text x="478" y="126" text-anchor="middle" font-size="10" font-family="sans-serif" fill="#1E88E5">Secondary Smelting</text>
</svg>
</div>
"""

# ── Global sidebar: dataset + time period + year (apply to all tabs) ─────────
with st.sidebar:
    st.radio(
        "BACI dataset",
        options=["HS12 (2012–2024)", "HS22 (2022–2024)"],
        index=0,
        key="baci_dataset",
        help=(
            "HS12 (2012–2024): the full historical dataset using HS 2017 product codes. "
            "Waste batteries = HS 854810.\n\n"
            "HS22 (2022–2024): the newer BACI release using HS 2022 product codes. "
            "Waste batteries = HS 854911 (a reclassified code that separates waste batteries "
            "more cleanly). Coverage limited to 2022–2024."
        ),
    )
    st.caption(
        "HS22 adds a dedicated code for waste batteries (854911), improving classification. "
        "HS12 offers the full 2012–2024 history. For trend analysis, use HS12."
    )

    st.divider()

    _mining_source_sel = st.radio(
        "Mining & refining data",
        options=["BGS (default)", "USGS"],
        index=0,
        key="mining_source",
        help=(
            "**BGS** (British Geological Survey): 1971–2023. Covers mine production and "
            "total refined lead. No primary/secondary split.\n\n"
            "**USGS** (US Geological Survey Mineral Yearbooks): 2015–2023. Separates "
            "primary and secondary refined lead.\n\n"
            "When a country/year is missing from the selected source, the app automatically "
            "falls back to the other source and shows a note."
        ),
    )
    st.caption(
        "BGS and USGS use similar methods but sometimes report different figures for the "
        "same country-year. Try both to check sensitivity — large differences may indicate "
        "reporting gaps or scope differences."
    )

    st.divider()

    time_period = st.radio(
        "Time period",
        options=["Single year", "3-year average (recommended)"],
        index=1,
        help=(
            "Single year: show model outputs for the selected year only.\n\n"
            "3-year average: average [year−1, year, year+1], clipped to data range. "
            "Smooths year-to-year trade volatility."
        ),
    )

    _year_label = "Center year" if time_period.startswith("3-year") else "Year"
    _default_year = min(2022, max_year)
    year = st.slider(
        _year_label, min_value=min_year, max_value=max_year, value=_default_year, step=1,
    )

    if time_period.startswith("3-year"):
        active_years = [y for y in [year - 1, year, year + 1] if min_year <= y <= max_year]
        period_label = f"{active_years[0]}–{active_years[-1]} average"
        st.caption(f"Window: {period_label}")
    else:
        active_years = [year]
        period_label = str(year)

    st.divider()

    with st.expander("Pb content factors", expanded=False):
        st.caption(
            "Lead content fractions applied to BACI trade quantities across all tabs. "
            "Uncheck a code to exclude it from all calculations."
        )
        _pb_overrides: dict[int, float] = {}
        _pb_disabled: set[int] = set()
        for _cat in CATEGORIES_ORDERED:
            _cat_codes = [(hs, m) for hs, m in HS_META.items() if m["cat"] == _cat]
            if not _cat_codes:
                continue
            st.markdown(f"**{_cat}**")
            for _hs, _meta in _cat_codes:
                _chk_col, _ctrl_col = st.columns([5, 4])
                _default_on = _hs not in _DEFAULT_OFF
                with _chk_col:
                    _enabled = st.checkbox(
                        f"{_hs} — {_meta['name']}",
                        value=_default_on,
                        key=f"pb_chk_{_hs}",
                    )
                with _ctrl_col:
                    if not _enabled:
                        _pb_disabled.add(_hs)
                        st.caption("excluded")
                    elif _meta["lo"] < _meta["hi"]:
                        _val = st.slider(
                            "Pb fraction",
                            min_value=float(_meta["lo"]),
                            max_value=float(_meta["hi"]),
                            value=float(_meta["default"]),
                            step=0.005,
                            format="%.3f",
                            key=f"pb_factor_{_hs}",
                            label_visibility="collapsed",
                        )
                        _pb_overrides[_hs] = _val
                    else:
                        st.caption(f"fixed: {_meta['default']:.0%}")
            st.divider()

    pb_factors: dict[int, float] = {hs: meta["default"] for hs, meta in HS_META.items()}
    pb_factors.update(_pb_overrides)
    for _hs in _pb_disabled:
        pb_factors[_hs] = 0.0

    _mining_pref: str = "BGS" if _mining_source_sel.startswith("BGS") else "USGS"


tab_overview, tab_trade, tab_massbal = st.tabs([
    "Overview", "Trade Toolkit", "Material Flow Toolkit"
])

# ── Overview tab ─────────────────────────────────────────────────────────────

with tab_overview:
    st.markdown(_LIFECYCLE_SVG, unsafe_allow_html=True)

    st.markdown(
        "Select a toolkit above to begin exploring. "
        "The **Trade Toolkit** provides direct views into bilateral trade data. "
        "The **Material Flow Toolkit** builds model-derived estimates of what happens inside each country."
    )
    st.write("")

    ov_left, ov_right = st.columns(2)

    with ov_left:
        with st.container(border=True):
            st.markdown("#### 🔄 Trade Toolkit")
            st.caption(
                "Direct views into BACI bilateral lead trade data, filterable by "
                "country, year, and product. Use these to explore observed trade flows."
            )
            st.markdown(
                "**Trade Analysis** — Map bilateral lead flows on a world map. "
                "Filter by product category and direction; click a country to see its "
                "top trading partners. Includes a regional aggregation view."
            )
            st.markdown(
                "**Trade Trends** — Time-series chart of a country's imports and exports "
                "by product category from 2012 to the present, with optional YoY % change view."
            )
            st.markdown(
                "**Flow Network** — Sankey-style node-link diagram of bilateral flows "
                "between selected countries. Arrows are scaled by volume; "
                "bubble size reflects total trade."
            )
            st.markdown(
                "**Trade Composition** — Color every country's trade portfolio using a "
                "triangular RYB model (blue = battery inputs, yellow = new batteries, "
                "red = waste/scrap). Spot structural patterns at a glance."
            )
            st.markdown(
                "**Supply Chain Provenance** — Trace the multi-tier upstream origins of "
                "a country's battery supply chain across three pathways: battery, "
                "smelted lead, and scrap/waste."
            )

    with ov_right:
        with st.container(border=True):
            st.markdown("#### ⚖ Material Flow Toolkit")
            st.caption(
                "Model-derived estimates built on top of trade and production data. "
                "These tools estimate what happens inside countries, not just across borders."
            )
            st.markdown(
                "**Production & Capacity** — Choropleth maps showing where lead is mined, "
                "where it is refined, where batteries are manufactured, and where batteries "
                "enter service. Logarithmic scale; animated playback available."
            )
            st.markdown(
                "**Recycling Economy Snapshot** — Summary scorecard of a country's "
                "recycling economy: collection rate, secondary smelting share, "
                "battery self-sufficiency, and comparison baselines."
            )
            st.markdown(
                "**Lead Accumulation** — A country's annual net lead balance: "
                "mining + imports − exports, broken down by product category. "
                "Highlights whether a country is a net lead accumulator or depleter."
            )
            st.markdown(
                "**Process Estimates** — Material-flow Sankey for any country, using "
                "trade plus mining/refining anchors. Broad coverage, but not reconciled "
                "against external observations."
            )
            st.markdown(
                "**Mass Balance** _(under construction)_ — Fully reconciled "
                "material-flow model for a country, calibrated against external "
                "observations (USGS primary/secondary refining, stock estimates). "
                "Currently being built out for India; not yet ready for use."
            )

    st.divider()
    st.caption(
        "Sidebar controls (left): switch between BACI HS12/HS22 datasets, BGS/USGS "
        "mining data, and time period. All tabs respond to these selections."
    )


# Category label ↔ internal code mapping (used in Trade Analysis)
# Emoji prefix provides color-coded visual cue in dropdowns and selected pills.
# CSS above adds a matching colored border/tint to selected pills.
_CAT_LABELS: dict[str, str] = {
    "🟩 New Batteries":       "BATT",
    "🟨 Used Batteries":       "USED",
    "🟧 Lead Scrap":           "SCRAP",
    "🟦 Smelted Lead":         "FEED",
    "⬜ Ore & Concentrates":   "ORE",
}
_CAT_LABEL_LIST = list(_CAT_LABELS.keys())


def _strip_cat_emoji(label: str) -> str:
    """Return the category name without its emoji prefix (for chart titles)."""
    return label.split(" ", 1)[1] if " " in label else label


# ── Trade Trends: series and chart helpers ────────────────────────────────────

_TT_HS_ALL   = {260700, 780110, 780191, 780199, 780200, 850710, 850720, 850790, 854810, 282410, 282490}
_TT_HS_SMELT = {780110, 780191, 780199}
_TT_HS_REF   = {780110}
_TT_HS_SCRAP = {780200}
_TT_HS_UBATT = {854810}
_TT_HS_NBATT = {850710, 850720}
_TT_HS_ORE   = {260700}

# Swap waste battery HS code for HS22 dataset
if _dataset_key == "hs22":
    _TT_HS_UBATT = {854911}
    _TT_HS_ALL   = {260700, 780110, 780191, 780199, 780200, 850710, 850720, 850790, 854911, 282410, 282490}

_TT_SERIES_NAMES: list[str] = [
    "Total Imports", "Total Exports",
    "Smelted Lead Imports", "Smelted Lead Exports",
    "Refined Lead Imports", "Refined Lead Exports",
    "Lead Scrap Imports", "Lead Scrap Exports",
    "Waste Battery Imports", "Waste Battery Exports",
    "New Battery Imports", "New Battery Exports",
    "Ore Imports", "Ore Exports",
    "Lead Mined", "Lead Smelted/Refined",
]

_TT_SERIES_COLORS: dict[str, str] = {
    "Total Imports":               "#1f77b4",
    "Total Exports":               "#ff7f0e",
    "Smelted Lead Imports":        "#2ca02c",
    "Smelted Lead Exports":        "#d62728",
    "Refined Lead Imports":        "#9467bd",
    "Refined Lead Exports":        "#8c564b",
    "Lead Scrap Imports":          "#e377c2",
    "Lead Scrap Exports":          "#7f7f7f",
    "Waste Battery Imports":       "#bcbd22",
    "Waste Battery Exports":       "#17becf",
    "New Battery Imports":         "#aec7e8",
    "New Battery Exports":         "#ffbb78",
    "Ore Imports":                 "#98df8a",
    "Ore Exports":                 "#ff9896",
    "Lead Mined":                  "#6a3d9a",
    "Lead Smelted/Refined":        "#b15928",
}

_TT_SLOT_COLORS = ["#1565C0", "#E65100", "#1B5E20"]
_TT_SEPARATOR   = "── Regions ──"
_RTV_REGION_SEP = "── UN Sub-regions ──"


def _tt_baci_series(
    baci: pd.DataFrame, countries: list[str], series: str
) -> pd.Series:
    is_import = "Import" in series
    side = baci["Importer"].isin(countries) if is_import else baci["Exporter"].isin(countries)
    if   series.startswith("Total"):         hs = _TT_HS_ALL
    elif series.startswith("Smelted Lead"):  hs = _TT_HS_SMELT
    elif series.startswith("Refined Lead"):  hs = _TT_HS_REF
    elif series.startswith("Lead Scrap"):    hs = _TT_HS_SCRAP
    elif series.startswith("Waste Battery"): hs = _TT_HS_UBATT
    elif series.startswith("New Battery"):   hs = _TT_HS_NBATT
    elif series.startswith("Ore"):           hs = _TT_HS_ORE
    else:                                    return pd.Series(dtype=float)
    return baci[side & baci["Product"].isin(hs)].groupby("Year")["actual_lead"].sum()


def _tt_bgs_series(
    bgs: pd.DataFrame, countries: list[str], series: str
) -> pd.Series:
    commodity = "lead, mine" if series == "Lead Mined" else "lead, refined"
    bgs_names = {_BACI_TO_BGS_NAME.get(c, c) for c in countries}
    mask = bgs["country_trans"].isin(bgs_names) & (bgs["bgs_commodity_trans"] == commodity)
    filtered = bgs[mask]
    if filtered.empty:
        return pd.Series(dtype=float)
    return filtered.groupby("year")["Mass of Pb"].sum().rename_axis("Year")


def _tt_usgs_series(
    usgs_mined: pd.DataFrame,
    usgs_refined: pd.DataFrame,
    countries: list[str],
    series: str,
) -> pd.Series:
    usgs_df = usgs_mined if series == "Lead Mined" else usgs_refined
    usgs_names = {_BACI_TO_USGS_NAME.get(c, c) for c in countries}
    filtered = usgs_df[usgs_df["country"].isin(usgs_names)]
    if filtered.empty:
        return pd.Series(dtype=float)
    return filtered.groupby("year")["value_metric_t"].sum().rename_axis("Year")


def _tt_compute_slot(
    baci: pd.DataFrame,
    bgs: pd.DataFrame,
    usgs_mined: pd.DataFrame,
    usgs_refined: pd.DataFrame,
    selection: str,
    series_list: list[str],
    year_idx: pd.Index,
    mining_pref: str = "BGS",
) -> tuple[dict[str, pd.Series], list[str]]:
    """Returns (data_dict, fallback_notes)."""
    is_region = selection in REGIONS_ORDERED
    countries = (
        [c for c, r in REGION_MAP.items() if r == selection]
        if is_region else [selection]
    )
    result: dict[str, pd.Series] = {}
    notes: list[str] = []
    for s in series_list:
        if s in ("Lead Mined", "Lead Smelted/Refined"):
            if mining_pref == "BGS":
                raw = _tt_bgs_series(bgs, countries, s)
                if raw.dropna().empty:
                    raw = _tt_usgs_series(usgs_mined, usgs_refined, countries, s)
                    if not raw.dropna().empty:
                        notes.append(
                            f"**{s}**: No BGS data for {selection} — using USGS (2015–2023)."
                        )
            else:
                raw = _tt_usgs_series(usgs_mined, usgs_refined, countries, s)
                if raw.dropna().empty:
                    raw = _tt_bgs_series(bgs, countries, s)
                    if not raw.dropna().empty:
                        notes.append(
                            f"**{s}**: No USGS data for {selection} — using BGS (1971–2023)."
                        )
        else:
            raw = _tt_baci_series(baci, countries, s)
        result[s] = raw.reindex(year_idx)
    return result, notes


def _tt_build_chart(
    series_data: dict[str, pd.Series],
    title: str,
    slot_color: str,
    view_mode: str,
    yrange: list | None,
) -> go.Figure:
    fig = go.Figure()
    for sname, s in series_data.items():
        y = (s.pct_change() * 100) if view_mode.startswith("Relative") else s
        color = _TT_SERIES_COLORS.get(sname, "#333")
        fig.add_trace(go.Scatter(
            x=list(s.index), y=list(y),
            mode="lines+markers", name=sname,
            line=dict(color=color, width=2),
            marker=dict(color=color, size=5),
            connectgaps=False,
        ))
    y_label = "% change YoY" if view_mode.startswith("Relative") else "tonnes Pb"
    fig.update_layout(
        title=dict(
            text=f'<span style="color:{slot_color}"><b>{title}</b></span>',
            x=0, font=dict(size=15),
        ),
        xaxis=dict(title="Year", tickmode="linear", dtick=2),
        yaxis=dict(title=y_label, range=yrange),
        height=400,
        margin=dict(l=60, r=20, t=50, b=140),
        legend=dict(orientation="h", yanchor="top", y=-0.32, xanchor="left", x=0),
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8")
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8")
    return fig



def _build_display_baci(
    baci_df: pd.DataFrame, active_years: list[int], center_year: int
) -> pd.DataFrame:
    """
    Annualise BACI flows across active_years: sum then divide by n_years.
    Returns a DataFrame with Year = center_year.
    """
    if len(active_years) == 1:
        return baci_df
    subset = baci_df[baci_df["Year"].isin(active_years)]
    agg = (
        subset.groupby(["Exporter", "Importer", "Product", "category"], as_index=False)
        ["actual_lead"].sum()
    )
    agg["actual_lead"] = agg["actual_lead"] / len(active_years)
    agg["Year"] = center_year
    return agg



with tab_trade:
    _t_analysis, _t_trends, _t_flow, _t_comp, _t_prov = st.tabs([
        "Trade Analysis", "Trade Trends", "Flow Network",
        "Trade Composition", "Supply Chain Provenance",
    ])

    # ── Trade Analysis ────────────────────────────────────────────────────────

    with _t_analysis:
        _ta_desc_col, _ta_lm_col = st.columns([8, 1])
        with _ta_desc_col:
            st.write(
                "Explore how lead products move between countries. "
                "The Country Trade View shows bilateral flows for a selected product and country. "
                "The Regional Flow View colors countries by UN M49 sub-region and shows which "
                "regions supply inputs to — or receive outputs from — a selected region."
            )
        with _ta_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data source:** BACI (Base pour l'Analyse du Commerce International), "
                    "published by CEPII. Harmonizes UN Comtrade import and export reports to "
                    "resolve bilateral discrepancies. Values are converted to tonnes of lead "
                    "content using the Pb fraction factors in the sidebar.\n\n"
                    "**How it's used:** Each row in BACI represents a bilateral flow "
                    "(exporter → importer, product, year). This tab aggregates those flows "
                    "by product category and displays them on a world map."
                )

        t3_view = st.radio(
            "View",
            options=["Country Trade View", "Regional Flow View"],
            index=0,
            horizontal=True,
            key="t3_view_radio",
        )

        st.divider()

        # ══ Country Trade View ═══════════════════════════════════════════════════════
        if t3_view == "Country Trade View":
            map_col1, map_col2 = st.columns([2, 3])

            with map_col1:
                map_cat_labels = st.multiselect(
                    "Product categories",
                    options=_CAT_LABEL_LIST,
                    default=_CAT_LABEL_LIST,
                    key="map_categories",
                    help=(
                        "New Batteries = HS 850710/850720 | "
                        "Used Batteries = HS 854810 | "
                        "Lead Scrap = HS 780200 | "
                        "Smelted Lead = HS 780110/780191/780199/850790/282410/282490 "
                        "(unwrought lead alloys; not all forms are technically refined) | "
                        "Ore & Concentrates = HS 260700. "
                        "Select multiple to sum Pb quantities across products."
                    ),
                )
                if not map_cat_labels:
                    map_cat_labels = _CAT_LABEL_LIST
            map_categories = [_CAT_LABELS[l] for l in map_cat_labels]

            with map_col2:
                _map_options = ["— All countries (total volume) —"] + all_baci_countries
                map_inspect = st.selectbox(
                    "Select country to inspect bilateral trade partners",
                    options=_map_options,
                    index=0,
                    key="map_inspect",
                )

            map_color_mode = st.radio(
                "Color scale mode",
                options=["Global volume scale", "Bilateral trade balance"],
                index=0,
                horizontal=True,
                key="map_color_mode",
                help=(
                    "**Global volume scale**: color by net flow direction, intensity by volume. "
                    "Vivid = large flow, pale = small flow.\n\n"
                    "**Bilateral trade balance**: color by the ratio of trade with the selected "
                    "country, regardless of volume. Two countries with the same color have the "
                    "same import/export ratio, even if their volumes differ."
                ),
            )

            display_baci = _build_display_baci(baci_df, active_years, year)
            _no_country_selected = (map_inspect == "— All countries (total volume) —")
            _cat_display_label = ", ".join(_strip_cat_emoji(l) for l in map_cat_labels)

            if _no_country_selected:
                period_note = f"  ({period_label})" if len(active_years) > 1 else ""
                st.caption(
                    f"Showing annualised trade volume (imports + exports) per country{period_note}. "
                    "Select a country above to see its bilateral trade partners."
                )
                fig_map = build_total_volume_map(display_baci, year, map_categories, _cat_display_label)
            else:
                period_note = f"  ({period_label})" if len(active_years) > 1 else ""
                if map_color_mode == "Bilateral trade balance":
                    fig_map = build_bilateral_balance_map(
                        display_baci, year, map_categories, map_inspect, _cat_display_label
                    )
                    st.caption(
                        f"**Orange**: {map_inspect} is net *receiver* from partner (partner dominates sending).  "
                        f"**Purple**: {map_inspect} is net *sender* to partner (partner dominates receiving).  "
                        f"**White**: balanced trade. Color shows ratio regardless of volume — two countries "
                        f"with the same shade have the same import/export ratio with {map_inspect}, "
                        f"even if their volumes differ.{period_note}"
                    )
                else:
                    fig_map = build_bilateral_map(
                        display_baci, year, map_categories, map_inspect, _cat_display_label
                    )
                    st.caption(
                        f"**Green**: partner is net *sender* to {map_inspect}.  "
                        f"**Red**: partner is net *receiver* from {map_inspect}.  "
                        f"**White**: balanced flow. Intensity scales with bilateral volume — pale = small flows.{period_note}"
                    )

            st.plotly_chart(fig_map, use_container_width=True)

            st.divider()
            _df_cats = display_baci[display_baci["category"].isin(map_categories)]

            if _no_country_selected:
                # No country selected — gross totals across all trade
                _exp_t = (
                    _df_cats.groupby("Exporter")["actual_lead"].sum()
                    .sort_values(ascending=False).head(30).reset_index()
                    .rename(columns={"Exporter": "Country", "actual_lead": "Exports (t Pb)"})
                )
                _exp_t.insert(0, "#", range(1, len(_exp_t) + 1))
                _imp_t = (
                    _df_cats.groupby("Importer")["actual_lead"].sum()
                    .sort_values(ascending=False).head(30).reset_index()
                    .rename(columns={"Importer": "Country", "actual_lead": "Imports (t Pb)"})
                )
                _imp_t.insert(0, "#", range(1, len(_imp_t) + 1))

                _tc1, _tc2 = st.columns(2)
                with _tc1:
                    st.caption(f"Top 30 exporters — {_cat_display_label}")
                    st.dataframe(_exp_t, hide_index=True, use_container_width=True,
                                 column_config={"Exports (t Pb)": st.column_config.NumberColumn(format="%d")})
                with _tc2:
                    st.caption(f"Top 30 importers — {_cat_display_label}")
                    st.dataframe(_imp_t, hide_index=True, use_container_width=True,
                                 column_config={"Imports (t Pb)": st.column_config.NumberColumn(format="%d")})
            else:
                # Country selected — bilateral net relative to map_inspect
                _col_exp = f"Exports to {map_inspect} (t Pb)"
                _col_imp = f"Imports from {map_inspect} (t Pb)"
                _sends = _df_cats[_df_cats["Importer"] == map_inspect].groupby("Exporter")["actual_lead"].sum()
                _gets  = _df_cats[_df_cats["Exporter"] == map_inspect].groupby("Importer")["actual_lead"].sum()
                _partners = _sends.index.union(_gets.index)
                _pbil = pd.DataFrame(index=_partners)
                _pbil.index.name = "Country"
                _pbil[_col_exp] = _sends.reindex(_partners, fill_value=0)
                _pbil[_col_imp] = _gets.reindex(_partners, fill_value=0)
                _pbil["_net"] = _pbil[_col_exp] - _pbil[_col_imp]

                _col_net_exp = f"Net Exports to {map_inspect} (t Pb)"
                _tbl_exp = (
                    _pbil[_pbil["_net"] > 0].sort_values("_net", ascending=False).head(30)
                    .reset_index().rename(columns={"_net": _col_net_exp})
                )
                _tbl_exp.insert(0, "#", range(1, len(_tbl_exp) + 1))

                _col_net_imp = f"Net Imports from {map_inspect} (t Pb)"
                _pbil_neg = _pbil[_pbil["_net"] < 0].copy()
                _pbil_neg[_col_net_imp] = -_pbil_neg["_net"]
                _tbl_imp = (
                    _pbil_neg.sort_values(_col_net_imp, ascending=False).head(30).reset_index()
                )
                _tbl_imp.insert(0, "#", range(1, len(_tbl_imp) + 1))

                _tc1, _tc2 = st.columns(2)
                with _tc1:
                    st.caption(f"Top 30 net exporters to {map_inspect} — {_cat_display_label}")
                    st.dataframe(
                        _tbl_exp[["#", "Country", _col_net_exp, _col_exp, _col_imp]],
                        hide_index=True, use_container_width=True,
                        column_config={
                            _col_exp: st.column_config.NumberColumn(format="%d"),
                            _col_imp: st.column_config.NumberColumn(format="%d"),
                            _col_net_exp: st.column_config.NumberColumn(format="%d"),
                        },
                    )
                with _tc2:
                    st.caption(f"Top 30 net importers from {map_inspect} — {_cat_display_label}")
                    st.dataframe(
                        _tbl_imp[["#", "Country", _col_net_imp, _col_imp, _col_exp]],
                        hide_index=True, use_container_width=True,
                        column_config={
                            _col_imp: st.column_config.NumberColumn(format="%d"),
                            _col_exp: st.column_config.NumberColumn(format="%d"),
                            _col_net_imp: st.column_config.NumberColumn(format="%d"),
                        },
                    )

        # ══ Regional Trade View ══════════════════════════════════════════════════════
        else:
            rtv_col1, rtv_col2 = st.columns([2, 3])

            with rtv_col1:
                rtv_cat_labels = st.multiselect(
                    "Product categories",
                    options=_CAT_LABEL_LIST,
                    default=_CAT_LABEL_LIST,
                    key="rtv_category",
                    help=(
                        "New Batteries = HS 850710/850720 | "
                        "Used Batteries = HS 854810 | "
                        "Lead Scrap = HS 780200 | "
                        "Smelted Lead = HS 780110/780191/780199/850790/282410/282490 | "
                        "Ore & Concentrates = HS 260700"
                    ),
                )
                if not rtv_cat_labels:
                    rtv_cat_labels = _CAT_LABEL_LIST
            rtv_categories = [_CAT_LABELS[l] for l in rtv_cat_labels]
            rtv_cat_display = ", ".join(_strip_cat_emoji(l) for l in rtv_cat_labels)

            with rtv_col2:
                rtv_region_options = (
                    ["— Select a region —"]
                    + MAJOR_REGIONS_ORDERED
                    + [_RTV_REGION_SEP]
                    + REGIONS_ORDERED
                )
                rtv_region_sel = st.selectbox(
                    "Select region",
                    options=rtv_region_options,
                    index=0,
                    key="rtv_region",
                    help="Choose a UN region (e.g. 'Africa') or a UN sub-region (e.g. 'Western Africa').",
                )

            _rtv_is_major = rtv_region_sel in MAJOR_REGIONS_ORDERED
            _rtv_region_map = MAJOR_REGION_MAP if _rtv_is_major else REGION_MAP

            rtv_color_mode = st.radio(
                "Color scale mode",
                options=["Global volume scale", "Bilateral trade balance"],
                index=0,
                horizontal=True,
                key="rtv_color_mode",
                help=(
                    "**Global volume scale**: color by net flow direction, intensity by volume. "
                    "Vivid = large flow, pale = small flow.\n\n"
                    "**Bilateral trade balance**: color by the ratio of trade with the selected "
                    "region, regardless of volume. Two countries with the same color have the "
                    "same import/export ratio, even if their volumes differ."
                ),
            )

            if rtv_region_sel.startswith("—") or rtv_region_sel == _RTV_REGION_SEP:
                st.info("Select a UN region or sub-region from the dropdown above to view its trade partners.")
            else:
                period_note = f"  ({period_label})" if len(active_years) > 1 else ""
                if rtv_color_mode == "Bilateral trade balance":
                    fig_rtv = build_region_balance_map(
                        baci_df=baci_df,
                        active_years=active_years,
                        category=rtv_categories,
                        selected_region=rtv_region_sel,
                        region_map=_rtv_region_map,
                        category_label=rtv_cat_display,
                    )
                    st.caption(
                        f"**Orange**: partner is net *sender* to {rtv_region_sel} (partner dominates sending).  "
                        f"**Purple**: partner is net *receiver* from {rtv_region_sel} (partner dominates receiving).  "
                        f"**White**: balanced trade.  "
                        f"<span style='color:#1565C0'>■</span> **Blue**: {rtv_region_sel} countries.{period_note}  "
                        "Color shows ratio regardless of volume — two countries with the same shade have the same "
                        f"import/export ratio with {rtv_region_sel}, even if their volumes differ.",
                        unsafe_allow_html=True,
                    )
                else:
                    fig_rtv = build_region_bilateral_map(
                        baci_df=baci_df,
                        active_years=active_years,
                        category=rtv_categories,
                        selected_region=rtv_region_sel,
                        region_map=_rtv_region_map,
                        category_label=rtv_cat_display,
                    )
                    st.caption(
                        f"**Green**: partner is net *sender* to {rtv_region_sel}.  "
                        f"**Red**: partner is net *receiver* from {rtv_region_sel}.  "
                        f"**White**: balanced flow.  "
                        f"<span style='color:#1565C0'>■</span> **Blue**: {rtv_region_sel} countries.{period_note}  "
                        "Intensity scales with bilateral volume — pale = small flows.",
                        unsafe_allow_html=True,
                    )
                st.plotly_chart(fig_rtv, use_container_width=True)

                st.divider()
                _rtv_df = baci_df[baci_df["Year"].isin(active_years) & baci_df["category"].isin(rtv_categories)]
                _n_yrs = max(len(active_years), 1)
                _rtv_region_ctries = {c for c, r in _rtv_region_map.items() if r == rtv_region_sel}

                # External flows only — cross-region bilateral relative to selected region
                _rtv_col_exp = f"Exports to {rtv_region_sel} (t Pb)"
                _rtv_col_imp = f"Imports from {rtv_region_sel} (t Pb)"
                _rtv_sends = (
                    _rtv_df[_rtv_df["Importer"].isin(_rtv_region_ctries)
                            & ~_rtv_df["Exporter"].isin(_rtv_region_ctries)]
                    .groupby("Exporter")["actual_lead"].sum() / _n_yrs
                )
                _rtv_gets = (
                    _rtv_df[_rtv_df["Exporter"].isin(_rtv_region_ctries)
                            & ~_rtv_df["Importer"].isin(_rtv_region_ctries)]
                    .groupby("Importer")["actual_lead"].sum() / _n_yrs
                )
                _rtv_partners = _rtv_sends.index.union(_rtv_gets.index)
                _rtv_pbil = pd.DataFrame(index=_rtv_partners)
                _rtv_pbil.index.name = "Country"
                _rtv_pbil[_rtv_col_exp] = _rtv_sends.reindex(_rtv_partners, fill_value=0)
                _rtv_pbil[_rtv_col_imp] = _rtv_gets.reindex(_rtv_partners, fill_value=0)
                _rtv_pbil["_net"] = _rtv_pbil[_rtv_col_exp] - _rtv_pbil[_rtv_col_imp]

                _rtv_col_net_exp = f"Net Exports to {rtv_region_sel} (t Pb)"
                _rtv_tbl_exp = (
                    _rtv_pbil[_rtv_pbil["_net"] > 0].sort_values("_net", ascending=False).head(30)
                    .reset_index().rename(columns={"_net": _rtv_col_net_exp})
                )
                _rtv_tbl_exp.insert(0, "#", range(1, len(_rtv_tbl_exp) + 1))

                _rtv_col_net_imp = f"Net Imports from {rtv_region_sel} (t Pb)"
                _rtv_pbil_neg = _rtv_pbil[_rtv_pbil["_net"] < 0].copy()
                _rtv_pbil_neg[_rtv_col_net_imp] = -_rtv_pbil_neg["_net"]
                _rtv_tbl_imp = (
                    _rtv_pbil_neg.sort_values(_rtv_col_net_imp, ascending=False).head(30).reset_index()
                )
                _rtv_tbl_imp.insert(0, "#", range(1, len(_rtv_tbl_imp) + 1))

                _rc1, _rc2 = st.columns(2)
                with _rc1:
                    st.caption(f"Top 30 net exporters to {rtv_region_sel} — {rtv_cat_display}")
                    st.dataframe(
                        _rtv_tbl_exp[["#", "Country", _rtv_col_net_exp, _rtv_col_exp, _rtv_col_imp]],
                        hide_index=True, use_container_width=True,
                        column_config={
                            _rtv_col_exp: st.column_config.NumberColumn(format="%d"),
                            _rtv_col_imp: st.column_config.NumberColumn(format="%d"),
                            _rtv_col_net_exp: st.column_config.NumberColumn(format="%d"),
                        },
                    )
                with _rc2:
                    st.caption(f"Top 30 net importers from {rtv_region_sel} — {rtv_cat_display}")
                    st.dataframe(
                        _rtv_tbl_imp[["#", "Country", _rtv_col_net_imp, _rtv_col_imp, _rtv_col_exp]],
                        hide_index=True, use_container_width=True,
                        column_config={
                            _rtv_col_imp: st.column_config.NumberColumn(format="%d"),
                            _rtv_col_exp: st.column_config.NumberColumn(format="%d"),
                            _rtv_col_net_imp: st.column_config.NumberColumn(format="%d"),
                        },
                    )



    # ── Trade Trends ──────────────────────────────────────────────────────────

    with _t_trends:
        _tt_desc_col, _tt_lm_col = st.columns([8, 1])
        with _tt_desc_col:
            st.write(
                "Plot import and export volumes for up to three countries or regions over time. "
                "Choose from trade series (by product category) or production series "
                "(mined and refined lead from BGS/USGS). Absolute tonnes or year-on-year % change."
            )
        with _tt_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data source:** BACI for trade series; BGS World Mineral Statistics "
                    "or USGS Mineral Yearbooks for production series.\n\n"
                    "**How it's used:** Trade series sum BACI flows by year for the selected "
                    "country or UN sub-region. Production series are drawn directly from the "
                    "mining/refining dataset chosen in the sidebar. The 'Lead Smelted/Refined' "
                    "series combines primary and secondary refined lead."
                )
        _bgs_data       = _load_bgs()
        _tt_usgs_mined  = _load_usgs_mined()
        _tt_usgs_ref    = _load_usgs_refined()
        _tt_year_idx = pd.Index(sorted(baci_df["Year"].unique()), name="Year")

        _tt_opts = sorted(all_baci_countries) + [_TT_SEPARATOR] + list(REGIONS_ORDERED)
        _tt_opt_with_none = ["— None —"] + _tt_opts
        _tt_ghana_idx = _tt_opts.index("Ghana") if "Ghana" in _tt_opts else 0

        tt_c1, tt_c2, tt_c3 = st.columns(3)
        with tt_c1:
            tt_sel1 = st.selectbox(
                "Country / Region 1", _tt_opts,
                index=_tt_ghana_idx, key="tt_sel1",
            )
        with tt_c2:
            tt_sel2 = st.selectbox(
                "Country / Region 2 (optional)", _tt_opt_with_none,
                index=0, key="tt_sel2",
            )
        with tt_c3:
            tt_sel3 = st.selectbox(
                "Country / Region 3 (optional)", _tt_opt_with_none,
                index=0, key="tt_sel3",
            )

        tt_rc1, tt_rc2, tt_rc3 = st.columns([3, 1, 1])
        with tt_rc1:
            tt_series = st.multiselect(
                "Series", _TT_SERIES_NAMES,
                default=["Total Imports", "Total Exports"], key="tt_series",
            )
        with tt_rc2:
            tt_view = st.radio(
                "View",
                ["Absolute (tonnes Pb)", "Relative (% change from prior year)"],
                index=0, key="tt_view",
            )
        with tt_rc3:
            tt_shared_y = st.checkbox("Shared y-axis", value=True, key="tt_shared_y")

        def _tt_valid(s: str) -> bool:
            return bool(s) and s != "— None —" and s != _TT_SEPARATOR

        _tt_slots = [
            (sel, col)
            for sel, col in zip([tt_sel1, tt_sel2, tt_sel3], _TT_SLOT_COLORS)
            if _tt_valid(sel)
        ]

        if not tt_series:
            st.info("Select at least one series above.")
        elif not _tt_slots:
            st.warning("Select a valid country or region in slot 1.")
        else:
            _tt_slot_results = [
                _tt_compute_slot(
                    baci_df, _bgs_data, _tt_usgs_mined, _tt_usgs_ref,
                    sel, tt_series, _tt_year_idx, _mining_pref,
                )
                for sel, _ in _tt_slots
            ]

            yrange = None
            if tt_shared_y:
                _all_y: list[float] = []
                for sd, _ in _tt_slot_results:
                    for s in sd.values():
                        y = (s.pct_change() * 100) if tt_view.startswith("Relative") else s
                        _all_y.extend(y.dropna().tolist())
                if _all_y:
                    _gmin, _gmax = min(_all_y), max(_all_y)
                    _pad = (_gmax - _gmin) * 0.05 if _gmax != _gmin else (abs(_gmax) * 0.1 or 1.0)
                    _lower = (_gmin - _pad) if tt_view.startswith("Relative") else 0.0
                    yrange = [_lower, _gmax + _pad]

            for (sel, color), (sd, notes) in zip(_tt_slots, _tt_slot_results):
                for _note in notes:
                    st.info(_note)
                fig = _tt_build_chart(sd, sel, color, tt_view, yrange)
                st.plotly_chart(fig, use_container_width=True)



    # ── Flow Network ──────────────────────────────────────────────────────────

    with _t_flow:
        _fn_desc_col, _fn_lm_col = st.columns([8, 1])
        with _fn_desc_col:
            st.write(
                "A schematic node-link diagram of bilateral lead product flows between selected countries. "
                "Bubble size reflects total BACI trade volume (imports + exports across all categories). "
                "Arrows are colored by product category and scaled by flow volume. "
                "Choose a layout preset — or switch to the **Interactive** view to drag nodes freely."
            )
        with _fn_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data source:** BACI bilateral trade flows.\n\n"
                    "**How it's used:** Flows below the minimum threshold are hidden to reduce "
                    "clutter. Each arrow represents the net flow between two countries for a "
                    "given product category. Arrow width is scaled to √(volume) so small flows "
                    "remain visible. Use this view to spot dominant corridors — e.g. which "
                    "countries are the main scrap suppliers to a processing hub."
                )

        # ── View mode ─────────────────────────────────────────────────────────────
        fn_view_mode = st.radio(
            "View mode",
            options=["Static (Plotly)", "Interactive (draggable)"],
            index=0,
            horizontal=True,
            key="fn_view_mode",
            help=(
                "**Static**: Plotly figure with a chosen layout preset (circle, grid, "
                "force-directed, or geographic).\n\n"
                "**Interactive**: pyvis / vis.js graph where you can grab and drag "
                "individual nodes. Zoom with the scroll wheel."
            ),
        )

        # ── Controls ──────────────────────────────────────────────────────────────
        fn_col1, fn_col2 = st.columns([3, 1])

        with fn_col1:
            _fn_defaults = [
                c for c in ["Ghana", "Nigeria", "India", "Rep. of Korea", "USA"]
                if c in all_baci_countries
            ]
            fn_countries = st.multiselect(
                "Select countries",
                options=all_baci_countries,
                default=_fn_defaults,
                key="fn_countries",
                help="Select 2–15 countries. Bubbles sized by total BACI trade volume; arrows show bilateral flows.",
            )

        with fn_col2:
            fn_min_flow = st.slider(
                "Minimum flow to display (t Pb)",
                min_value=100, max_value=10000, value=500, step=100,
                key="fn_min_flow",
            )

        # Layout preset + focal country + cluster controls
        _LAYOUT_LABELS = {
            "Circle": "circle",
            "Grid": "grid",
            "Force-directed (by trade weight)": "force",
            "Geographic centroids": "geographic",
        }
        fn_layout_col0, fn_layout_col1, fn_layout_col2 = st.columns([2, 2, 2])
        with fn_layout_col0:
            fn_layout_label = st.selectbox(
                "Initial layout",
                options=list(_LAYOUT_LABELS.keys()),
                index=0,
                key="fn_layout",
                help=(
                    "**Circle**: even spacing around a ring (with focal at center).  "
                    "**Grid**: rows and columns.  "
                    "**Force-directed**: pairs with heavier trade pull closer together.  "
                    "**Geographic centroids**: approximate world position on a schematic canvas."
                ),
            )
        fn_layout_code = _LAYOUT_LABELS[fn_layout_label]

        with fn_layout_col1:
            if fn_countries:
                fn_focal_raw = st.selectbox(
                    "Focal country (placed at center)",
                    options=["— None —"] + fn_countries,
                    index=1 if fn_countries else 0,
                    key="fn_focal",
                    help=(
                        "For the Circle layout the focal country goes at the center. "
                        "For Grid, Force, and Geographic layouts it is placed but not centered."
                    ),
                )
                fn_focal_country = None if fn_focal_raw.startswith("—") else fn_focal_raw
            else:
                fn_focal_country = None

        with fn_layout_col2:
            fn_use_cluster = st.toggle(
                "Treat a region as single node",
                value=False,
                key="fn_use_cluster",
                help="Aggregate all flows to/from countries in the selected region into a single node.",
            )
            if fn_use_cluster:
                _fn_cluster_options = (
                    MAJOR_REGIONS_ORDERED
                    + [_RTV_REGION_SEP]
                    + REGIONS_ORDERED
                )
                fn_cluster_region = st.selectbox(
                    "Region to aggregate",
                    options=_fn_cluster_options,
                    key="fn_cluster_region",
                    help="Choose a UN region (e.g. 'Africa') or a UN sub-region (e.g. 'Western Africa').",
                )
            else:
                fn_cluster_region = None

        # Category checkboxes — horizontal row
        fn_cat_cols = st.columns(4)
        with fn_cat_cols[0]:
            fn_feed  = st.checkbox(
                "Smelted Lead (FEED)", value=True, key="fn_feed",
            )
        with fn_cat_cols[1]:
            fn_batt  = st.checkbox(
                "New Batteries (BATT)", value=True, key="fn_batt",
            )
        with fn_cat_cols[2]:
            fn_used  = st.checkbox(
                "Used Batteries (USED)", value=True, key="fn_used",
            )
        with fn_cat_cols[3]:
            fn_scrap = st.checkbox(
                "Lead Scrap (SCRAP)", value=True, key="fn_scrap",
            )

        fn_categories = (
            (["FEED"]  if fn_feed  else []) +
            (["BATT"]  if fn_batt  else []) +
            (["USED"]  if fn_used  else []) +
            (["SCRAP"] if fn_scrap else [])
        )

        fn_dir = st.radio(
            "Flow direction",
            options=["Both", "Import flows", "Export flows"],
            index=0,
            horizontal=True,
            key="fn_dir",
            disabled=fn_focal_country is None,
            help=(
                "Filters arrows relative to the focal country: **Import flows** "
                "shows arrows into the focal country, **Export flows** shows arrows "
                "out of it. Requires a focal country — with none set, every arrow is "
                "both an import and an export, so only 'Both' applies."
            ),
        )
        # Import/export is only meaningful relative to the focal country; with
        # none set, force "Both" so the (disabled) radio can't hide everything.
        if fn_focal_country is None:
            fn_dir = "Both"
        fn_show_imports = fn_dir in ["Both", "Import flows"]
        fn_show_exports = fn_dir in ["Both", "Export flows"]

        # ── Scope + arrow hiding controls ────────────────────────────────────────
        fn_scope_col1, fn_scope_col2, fn_scope_col3 = st.columns(3)
        with fn_scope_col1:
            fn_all_years = st.checkbox(
                "Use all years in BACI dataset",
                value=False,
                key="fn_all_years",
                help=(
                    "Ignore the sidebar year filter and use every year in the "
                    f"BACI dataset ({min_year}–{max_year}). Bubble sizes and arrow "
                    "widths reflect the full-period annualised trade volume."
                ),
            )
        with fn_scope_col2:
            fn_focal_only = st.checkbox(
                "Only show flows involving focal country",
                value=False,
                key="fn_focal_only",
                disabled=fn_focal_country is None,
                help=(
                    "When a focal country is set, hide arrows between two non-focal "
                    "countries. Useful for keeping the focal country as the visual "
                    "centerpiece (e.g. Ghana with USA and India both present, but no "
                    "USA↔India arrow)."
                ),
            )
        with fn_scope_col3:
            fn_prune_isolated = st.checkbox(
                "Auto-remove countries with no focal connection",
                value=False,
                key="fn_prune_isolated",
                disabled=fn_focal_country is None,
                help=(
                    "After all other filters, drop any selected country that has no "
                    "arrow to the focal country. Useful when you pick 'all countries' "
                    "and only want to see those that actually trade with the focal "
                    "country. Reversible — untick to bring them back."
                ),
            )

        _FN_PAIR_HIDER_CAP = 20
        if len(fn_countries) > _FN_PAIR_HIDER_CAP:
            st.caption(
                f"Pair-hider available with ≤ {_FN_PAIR_HIDER_CAP} countries selected "
                f"({len(fn_countries)} selected). Narrow the selection to hide "
                "specific pairs."
            )
            fn_hidden_pairs: set[frozenset[str]] = set()
        else:
            _fn_all_pairs: list[tuple[str, str]] = []
            _seen: set[frozenset[str]] = set()
            for _a in fn_countries:
                for _b in fn_countries:
                    if _a == _b:
                        continue
                    key = frozenset([_a, _b])
                    if key in _seen:
                        continue
                    _seen.add(key)
                    _fn_all_pairs.append(tuple(sorted([_a, _b])))
            _fn_pair_labels = [f"{a} ↔ {b}" for a, b in _fn_all_pairs]
            _fn_pair_lookup = {f"{a} ↔ {b}": frozenset([a, b]) for a, b in _fn_all_pairs}
            fn_hidden_pair_labels = st.multiselect(
                "Hide specific country pairs",
                options=_fn_pair_labels,
                default=[],
                key="fn_hidden_pairs",
                help=(
                    "Pick country pairs to hide. Both directions are removed. "
                    "Example: select 'India ↔ USA' to drop USA→India and India→USA "
                    "arrows while keeping every other flow."
                ),
            )
            fn_hidden_pairs = {
                _fn_pair_lookup[l] for l in fn_hidden_pair_labels
            }

        st.divider()

        # ── Cluster aggregation ───────────────────────────────────────────────────
        _baci_for_fn = baci_df
        _fn_countries_final = fn_countries
        if fn_use_cluster and fn_cluster_region and fn_cluster_region != _RTV_REGION_SEP and fn_countries:
            _fn_cluster_map = MAJOR_REGION_MAP if fn_cluster_region in MAJOR_REGIONS_ORDERED else REGION_MAP
            _cluster_members = {c for c, r in _fn_cluster_map.items() if r == fn_cluster_region}
            _cluster_label = fn_cluster_region
            _baci_fn_subset = baci_df.copy()
            _baci_fn_subset["Exporter"] = _baci_fn_subset["Exporter"].apply(
                lambda x: _cluster_label if x in _cluster_members else x
            )
            _baci_fn_subset["Importer"] = _baci_fn_subset["Importer"].apply(
                lambda x: _cluster_label if x in _cluster_members else x
            )
            _baci_fn_subset = (
                _baci_fn_subset
                .groupby(["Year", "Exporter", "Importer", "Product", "category"], as_index=False)
                ["actual_lead"].sum()
            )
            _fn_countries_final = [c for c in fn_countries if c not in _cluster_members]
            if any(c in fn_countries for c in _cluster_members):
                _fn_countries_final = [_cluster_label] + _fn_countries_final
            _baci_for_fn = _baci_fn_subset
            if fn_focal_country in _cluster_members:
                fn_focal_country = _cluster_label

        # ── Effective year range for the Flow Network ─────────────────────────────
        _fn_active_years = all_years if fn_all_years else active_years
        _fn_period_label = (
            f"{all_years[0]}–{all_years[-1]} (all years)"
            if fn_all_years
            else period_label
        )

        # ── Validation + render ───────────────────────────────────────────────────
        if len(_fn_countries_final) < 2:
            st.info("Select at least 2 countries (or a region cluster with other countries) to display the flow network.")
        elif not fn_categories:
            st.info("Select at least one product category.")
        elif fn_view_mode == "Interactive (draggable)":
            _fn_hidden_key = frozenset(fn_hidden_pairs)
            try:
                _fn_html = _build_flow_network_html_cached(
                    baci_df=_baci_for_fn,
                    active_years=tuple(_fn_active_years),
                    countries=tuple(_fn_countries_final),
                    categories=tuple(fn_categories),
                    min_flow=fn_min_flow,
                    show_imports=fn_show_imports,
                    show_exports=fn_show_exports,
                    focal_country=fn_focal_country,
                    layout=fn_layout_code,
                    physics_enabled=False,
                    height_px=650,
                    focal_only=fn_focal_only,
                    hidden_pairs=_fn_hidden_key,
                    prune_isolated=fn_prune_isolated,
                )
                components.html(_fn_html, height=680, scrolling=False)
                st.caption(
                    "**Drag** any node to reposition it. **Scroll** to zoom. "
                    "**Drag empty canvas** to pan. Initial positions come from the "
                    f"chosen layout preset (**{fn_layout_label}**). "
                    f"Bubble size ∝ √(total BACI trade volume). "
                    f"Arrow width ∝ √(flow volume). "
                    f"Showing flows ≥ {fn_min_flow:,} t Pb. Period: {_fn_period_label}."
                )
            except ImportError as e:
                st.error(str(e))
                st.info("Falling back to the Static view.")
                fig_fn = _build_flow_network_cached(
                    baci_df=_baci_for_fn,
                    active_years=tuple(_fn_active_years),
                    countries=tuple(_fn_countries_final),
                    categories=tuple(fn_categories),
                    min_flow=fn_min_flow,
                    show_imports=fn_show_imports,
                    show_exports=fn_show_exports,
                    focal_country=fn_focal_country,
                    layout=fn_layout_code,
                    focal_only=fn_focal_only,
                    hidden_pairs=_fn_hidden_key,
                    prune_isolated=fn_prune_isolated,
                )
                st.plotly_chart(fig_fn, use_container_width=True)
        else:
            _fn_hidden_key = frozenset(fn_hidden_pairs)
            fig_fn = _build_flow_network_cached(
                baci_df=_baci_for_fn,
                active_years=tuple(_fn_active_years),
                countries=tuple(_fn_countries_final),
                categories=tuple(fn_categories),
                min_flow=fn_min_flow,
                show_imports=fn_show_imports,
                show_exports=fn_show_exports,
                focal_country=fn_focal_country,
                layout=fn_layout_code,
                focal_only=fn_focal_only,
                hidden_pairs=_fn_hidden_key,
                prune_isolated=fn_prune_isolated,
            )
            st.plotly_chart(fig_fn, use_container_width=True)
            st.caption(
                f"Layout: **{fn_layout_label}**. "
                f"Bubble size ∝ √(total BACI trade volume). "
                f"Arrow width ∝ √(flow volume). "
                f"Showing flows ≥ {fn_min_flow:,} t Pb between selected countries. "
                f"Period: {_fn_period_label}."
            )


    # ── Trade Composition ─────────────────────────────────────────────────────

    with _t_comp:
        _tc_hdr_col, _tc_lm_col = st.columns([8, 1])
        with _tc_hdr_col:
            st.subheader("Lead Trade Composition Map")
            st.caption(
                "Color encodes each country's import or export mix across three lead product categories. "
                "Uses a triangular RYB color model: blue = battery inputs, yellow = new batteries, "
                "red = battery waste/scrap. Mixed colors indicate blended portfolios; white = balanced."
            )
        with _tc_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data source:** BACI trade flows, aggregated into three categories: "
                    "Battery Inputs (refined lead, oxides, battery parts), "
                    "New Batteries (HS 850710/720), and Battery Waste (scrap + used batteries).\n\n"
                    "**How it's used:** Each country's share across the three categories is "
                    "mapped to an RYB color using a circular-mean algorithm. The color at each "
                    "corner is pure — blue, yellow, or red. Mixed economies (e.g. importing both "
                    "batteries and feedstock) appear as intermediate hues. A white country has "
                    "roughly equal shares across all three categories.\n\n"
                    "Use the ternary chart above the map to compare specific countries in detail."
                )
        render_trade_composition_map(sidebar_year=year, dataset=_dataset_key, pb_factors=pb_factors)


    # ── Supply Chain Provenance ───────────────────────────────────────────────

    with _t_prov:
        _pv_desc_col, _pv_lm_col = st.columns([8, 1])
        with _pv_desc_col:
            st.write(
                "Trace all input supply chains into a selected end-market country across three pathways. "
                "Pathway A follows the battery supply chain: who ships new batteries, who "
                "supplies their feedstock, and who supplies scrap to those smelters. "
                "Pathway B traces direct smelted lead suppliers and their scrap inputs. "
                "Pathway C shows direct scrap and waste battery suppliers. "
                "Colors match the product palette: 🟩 green = battery pathway (A), 🟦 blue = smelted lead pathway (B), 🟧 orange = scrap/waste pathway (C). Darker shade = closer to the end market."
            )
        with _pv_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data source:** BACI bilateral trade flows.\n\n"
                    "**How it's used:** Starting from the selected country, the tool walks "
                    "backwards through the BACI network up to three layers deep. Layer 1 = "
                    "direct suppliers; Layer 2 = those suppliers' suppliers; Layer 3 = one "
                    "step further back. Each pathway follows a specific product type so you "
                    "can trace, for example, which countries supply scrap to the smelters "
                    "that supply refined lead to the battery manufacturers that supply your "
                    "end market. Note: this is a trade-flow proxy — it cannot verify actual "
                    "physical supply chain links."
                )

        prov_col1, prov_col2 = st.columns([3, 1])

        with prov_col1:
            prov_country = st.selectbox(
                "Select end-market country",
                options=all_baci_countries,
                index=all_baci_countries.index("USA") if "USA" in all_baci_countries else 0,
                key="prov_country",
                help=(
                    "Country receiving lead-acid battery products. "
                    "The map traces all input supply chains backwards through three pathways."
                ),
            )

        with prov_col2:
            prov_top_n = st.slider(
                "Countries per tier",
                min_value=3,
                max_value=20,
                value=10,
                step=1,
                key="prov_top_n",
                help="Maximum number of countries shown in each supply-chain tier.",
            )

        prov_layer = st.radio(
            "Supply chain depth",
            options=["Layer 1 only", "Layers 1–2", "Layers 1–3"],
            index=2,
            horizontal=True,
            key="prov_layer",
            help=(
                "Layer 1: direct suppliers (A1, B1, C1). "
                "Layer 2: suppliers to direct suppliers (A2, B2). "
                "Layer 3: one step further back (A3)."
            ),
        )
        prov_max_layer = {"Layer 1 only": 1, "Layers 1–2": 2, "Layers 1–3": 3}[prov_layer]

        period_note = f" ({period_label})" if len(active_years) > 1 else ""

        fig_prov = build_provenance_map(
            baci_df=baci_df,
            active_years=active_years,
            selected_country=prov_country,
            top_n=prov_top_n,
            max_layer=prov_max_layer,
        )
        st.plotly_chart(fig_prov, use_container_width=True)

        st.markdown(
            f"**{prov_country}** "
            "<span style='background:#1565C0;color:#fff;padding:1px 6px;border-radius:3px'>"
            "■ Selected</span> &nbsp;"
            "<span style='background:#43A047;color:#fff;padding:1px 6px;border-radius:3px'>"
            "■ A1 — battery suppliers</span> &nbsp;"
            "<span style='background:#A5D6A7;color:#333;padding:1px 6px;border-radius:3px'>"
            "■ A2/A3 — upstream battery chain</span> &nbsp;"
            "<span style='background:#1E88E5;color:#fff;padding:1px 6px;border-radius:3px'>"
            "■ B1 — smelted lead suppliers</span> &nbsp;"
            "<span style='background:#90CAF9;color:#333;padding:1px 6px;border-radius:3px'>"
            "■ B2 — upstream smelter feedstock</span> &nbsp;"
            "<span style='background:#FB8C00;color:#fff;padding:1px 6px;border-radius:3px'>"
            "■ C1 — scrap/waste suppliers</span>  "
            f"Hover over any country for tier details.{period_note}",
            unsafe_allow_html=True,
        )

        a1_df, a2_df, a3_df, b1_df, b2_df, c1_df = get_provenance_tables(
            baci_df=baci_df,
            active_years=active_years,
            selected_country=prov_country,
            top_n=prov_top_n,
            max_layer=prov_max_layer,
        )

        st.divider()
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Pathway A — Battery Supply Chain**")
            st.caption(f"A1 — New battery suppliers to {prov_country}")
            if a1_df.empty:
                st.info(f"No battery imports recorded for **{prov_country}** in {period_label}.")
            else:
                st.dataframe(a1_df, hide_index=True, use_container_width=True,
                             column_config={"t Pb (new batteries)": st.column_config.NumberColumn(format="%d")})
            st.caption("A2 — Feedstock (FEED/SCRAP/USED) suppliers to Tier A1")
            if a2_df.empty:
                st.info("No feedstock flows found for Tier A1 countries.")
            else:
                st.dataframe(a2_df, hide_index=True, use_container_width=True,
                             column_config={"t Pb (feedstock to battery mfrs)": st.column_config.NumberColumn(format="%d")})
            st.caption("A3 — Scrap/waste suppliers to Tier A2")
            if a3_df.empty:
                st.info("No scrap/waste flows found for Tier A2 countries.")
            else:
                st.dataframe(a3_df, hide_index=True, use_container_width=True,
                             column_config={"t Pb (scrap to smelters)": st.column_config.NumberColumn(format="%d")})

        with col_b:
            st.markdown("**Pathway B — Direct Smelted Lead Supply**")
            st.caption(f"B1 — Smelted lead suppliers to {prov_country}")
            if b1_df.empty:
                st.info(f"No smelted lead imports recorded for **{prov_country}** in {period_label}.")
            else:
                st.dataframe(b1_df, hide_index=True, use_container_width=True,
                             column_config={"t Pb (smelted lead)": st.column_config.NumberColumn(format="%d")})
            st.caption("B2 — Scrap/waste suppliers to Tier B1 countries")
            if b2_df.empty:
                st.info("No scrap/waste flows found for Tier B1 countries.")
            else:
                st.dataframe(b2_df, hide_index=True, use_container_width=True,
                             column_config={"t Pb (scrap to lead producers)": st.column_config.NumberColumn(format="%d")})

        with col_c:
            st.markdown("**Pathway C — Direct Scrap/Waste Supply**")
            st.caption(f"C1 — Scrap/waste suppliers directly to {prov_country}")
            if c1_df.empty:
                st.info(f"No scrap or waste battery imports recorded for **{prov_country}** in {period_label}.")
            else:
                st.dataframe(c1_df, hide_index=True, use_container_width=True,
                             column_config={"t Pb (scrap/waste)": st.column_config.NumberColumn(format="%d")})

        st.caption(
            f"Flows annualised over {period_label}. All values in tonnes of lead (t Pb). "
            "A1: HS 850710/850720 (new batteries). "
            "B1: HS 780110/780191/780199/850790/282410/282490 (smelted lead/feed). "
            "C1/A2/A3/B2: HS 780200/854810 (scrap/waste batteries). "
            "Countries excluded from lower tiers if already shown in a higher-priority tier."
        )



# ══════════════════════════════════════════════════════════════════════════════
# TAB PROD — Production & Capacity
# ══════════════════════════════════════════════════════════════════════════════

def _make_choropleth(
    locations: list[str],
    z_values: list[float],
    title: str,
    colorscale: str,
    colorbar_title: str,
    height: int = 420,
    log_scale: bool = False,
    zmin_log: float | None = None,
    zmax_log: float | None = None,
) -> go.Figure:
    """
    Build a choropleth map. With log_scale=True, z_values are log10-transformed
    and the colorbar shows original (non-log) tick labels.
    Countries not in locations appear in light grey (no data).
    """
    import math

    orig_vals = list(z_values)

    if log_scale:
        plot_z = [math.log10(v) if v and v > 0 else float("nan") for v in orig_vals]
        valid_logs = [v for v in plot_z if v == v]  # filter NaN
        _zmin = zmin_log if zmin_log is not None else (math.floor(min(valid_logs)) if valid_logs else 0)
        _zmax = zmax_log if zmax_log is not None else (math.ceil(max(valid_logs)) if valid_logs else 5)
        tickvals = list(range(int(_zmin), int(_zmax) + 1))
        ticktext = [f"{10**v:,.0f}" for v in tickvals]
        colorbar = dict(
            title=colorbar_title, len=0.6, thickness=14,
            tickvals=tickvals, ticktext=ticktext,
        )
        hover = "%{location}: %{customdata:,.0f} t Pb<extra></extra>"
        choropleth = go.Choropleth(
            locations=locations,
            z=plot_z,
            customdata=orig_vals,
            locationmode="country names",
            colorscale=colorscale,
            zmin=_zmin,
            zmax=_zmax,
            colorbar=colorbar,
            hovertemplate=hover,
            marker_line_color="#ffffff",
            marker_line_width=0.4,
        )
    else:
        choropleth = go.Choropleth(
            locations=locations,
            z=orig_vals,
            locationmode="country names",
            colorscale=colorscale,
            zmin=0,
            colorbar=dict(title=colorbar_title, len=0.6, thickness=14),
            hovertemplate="%{location}: %{z:,.0f} t Pb<extra></extra>",
            marker_line_color="#ffffff",
            marker_line_width=0.4,
        )

    fig = go.Figure(choropleth)
    fig.update_layout(
        height=height,
        margin={"r": 0, "t": 36, "l": 0, "b": 0},
        title=dict(text=title, x=0.5, font=dict(size=14)),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#aaaaaa",
            showland=True,
            landcolor="#d8d8d8",
            showocean=True,
            oceancolor="#e3f2fd",
            showlakes=False,
            projection_type="natural earth",
        ),
    )
    return fig




with tab_massbal:
    # Display order matches the labels list. Variable names remain bound to
    # their semantic content (e.g. _mb_snapshot still wraps the snapshot block
    # further down the file); only the tab order shown to the user changes.
    _mb_prod, _mb_snapshot, _mb_accum, _mb_process, _mb_mass_balance = st.tabs([
        "Production & Capacity",
        "Recycling Economy Snapshot",
        "Lead Accumulation",
        "Process Estimates",
        "Mass Balance (Under Construction)",
    ])

    # ── Production & Capacity ─────────────────────────────────────────────────

    with _mb_prod:
        import math as _math

        _mbp_desc_col, _mbp_lm_col = st.columns([8, 1])
        with _mbp_desc_col:
            st.write(
                "Where is lead being mined? Where is it being refined? Where are batteries "
                "manufactured, and where do they enter service? Select a question below to "
                "explore the global picture. All scales are logarithmic — enabling comparison "
                "across small and large producers. Countries in grey have no data for the "
                "selected view. Use the animated toggle for time-lapse playback."
            )
        with _mbp_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Lead Mining / Lead Refining** — drawn directly from BGS or USGS "
                    "national statistics (select source in sidebar). These are reported "
                    "figures, not estimates.\n\n"
                    "**Battery Manufacturing / Battery Use** — derived from the mass-balance "
                    "model using trade data and process parameters. Adjust the parameter "
                    "sliders below the map to test sensitivity.\n\n"
                    "For Lead Mining and Lead Refining, a ranked country table is shown "
                    "below the map."
                )

        _mining_refining_df3 = _load_mining_refining()

        _ESTIMATED_DATASETS = {
            "Battery Manufacturing",
            "Battery Use",
        }
        _COLORSCALES = {
            "Lead Mining":              "Greys",
            "Lead Refining":            "Purples",
            "Battery Manufacturing":    "Greens",
            "Battery Use":              "Reds",
        }
        _METRIC_KEYS = {
            "Battery Manufacturing":    "F4_batt_lead",
            "Battery Use":              "F5_implied",
        }

        # ── Dataset selector + animation toggle ───────────────────────────────────
        _pd_ctrl1, _pd_ctrl2 = st.columns([5, 1])
        with _pd_ctrl1:
            prod_dataset = st.radio(
                "Dataset",
                options=list(_COLORSCALES.keys()),
                index=0,
                horizontal=True,
                key="prod_dataset",
            )
        with _pd_ctrl2:
            prod_animate = st.toggle("Animated", value=False, key="prod_animate")

        _colorscale = _COLORSCALES[prod_dataset]

        # ── Process parameters (estimated metrics only) ───────────────────────────
        _map_eta_secondary = 0.97
        _map_eta_break     = 0.95
        _map_delta_pb      = 0.95
        _map_beta          = 0.85
        _map_eta_mfg       = 0.98
        _map_eta_ore       = 0.95
        _map_gamma         = 0.70

        if prod_dataset in _ESTIMATED_DATASETS:
            st.info(
                f"**{prod_dataset}** is a model estimate derived from the mass-balance "
                "equations using trade data and mining/refining anchors. "
                "Adjust the parameters below to explore sensitivity."
            )
            with st.expander("Process parameters", expanded=False):
                _mp_c1, _mp_c2 = st.columns(2)
                with _mp_c1:
                    _map_eta_secondary = st.slider(
                        "Secondary smelting recovery (η_secondary)",
                        0.80, 1.00, 0.97, 0.01, format="%.2f", key="map_eta_secondary",
                    )
                    _map_eta_break = st.slider(
                        "Breaking recovery (η_break)",
                        0.70, 1.00, 0.95, 0.01, format="%.2f", key="map_eta_break",
                    )
                    _map_delta_pb = st.slider(
                        "Pb retained at end-of-life (δ)",
                        0.80, 1.00, 0.95, 0.01, format="%.2f", key="map_delta_pb",
                    )
                    _map_gamma = st.slider(
                        "Collection rate (γ)",
                        0.30, 1.00, 0.70, 0.01, format="%.2f", key="map_gamma",
                        help="Global default. Country-specific rates are not applied here.",
                    )
                with _mp_c2:
                    _map_beta = st.slider(
                        "Battery share of lead demand (β)",
                        0.50, 1.00, 0.85, 0.01, format="%.2f", key="map_beta",
                    )
                    _map_eta_mfg = st.slider(
                        "Manufacturing efficiency (η_mfg)",
                        0.90, 1.00, 0.98, 0.01, format="%.2f", key="map_eta_mfg",
                    )
                    _map_eta_ore = st.slider(
                        "Primary smelting recovery (η_ore)",
                        0.80, 1.00, 0.95, 0.01, format="%.2f", key="map_eta_ore",
                    )

        # ── Resolve data source and years ─────────────────────────────────────────
        if prod_dataset in _ESTIMATED_DATASETS:
            # Restrict to years present in BOTH BACI trade data and the mining/refining
            # anchor. The anchor only goes to 2023; without it build_sankey returns
            # __no_refining__ for every country, producing an empty map.
            _mining_years_set = set(_mining_refining_df3["year"].unique().tolist())
            _prod_years = sorted(
                y for y in baci_df["Year"].unique() if y in _mining_years_set
            )
            _caption = (
                f"**{prod_dataset}** — model estimate (mass-balance). "
                "Values are in metric tonnes of lead content. "
                "Scale is logarithmic. Countries with no refining anchor shown in grey."
            )
        else:
            # Use country_year_mining_refining.csv
            if prod_dataset == "Lead Mining":
                _mr_col = "mined_bgs_t" if _mining_pref == "BGS" else "mined_usgs_t"
            else:  # Lead Refining
                if _mining_pref == "BGS":
                    _mr_col = "refined_bgs_t"
                else:
                    _mining_refining_df3 = _mining_refining_df3.copy()
                    _idx = _mining_refining_df3.index
                    _p = (_mining_refining_df3["refined_primary_usgs_t"].fillna(0)
                          if "refined_primary_usgs_t" in _mining_refining_df3.columns
                          else pd.Series(0.0, index=_idx))
                    _s = (_mining_refining_df3["refined_secondary_usgs_t"].fillna(0)
                          if "refined_secondary_usgs_t" in _mining_refining_df3.columns
                          else pd.Series(0.0, index=_idx))
                    _mining_refining_df3["_ref_usgs"] = _p + _s
                    _mr_col = "_ref_usgs"

            _mr_sub = (
                _mining_refining_df3[_mining_refining_df3[_mr_col].notna()]
                if _mr_col in _mining_refining_df3.columns
                else pd.DataFrame()
            )
            _prod_years = sorted(_mr_sub["year"].unique().tolist()) if not _mr_sub.empty else []
            _src_name = "BGS" if _mining_pref == "BGS" else "USGS"
            _prod_type = (
                "mine production (t Pb in concentrates)"
                if prod_dataset == "Lead Mining"
                else "primary + secondary refined lead (t Pb)"
            )
            _caption = (
                f"Source: {_src_name} ({min(_prod_years) if _prod_years else '—'}–"
                f"{max(_prod_years) if _prod_years else '—'}). {_prod_type.capitalize()}. "
                "Scale is logarithmic. Countries with no data shown in grey."
            )

        if not _prod_years:
            st.warning("No data available for the selected dataset and source.")
        else:
            # ── Frame data helper ──────────────────────────────────────────────────
            _pb_factors_tuple = tuple(sorted(pb_factors.items()))

            def _get_frame_data(yr: int) -> tuple[list[str], list[float]]:
                if prod_dataset in _ESTIMATED_DATASETS:
                    _od = _compute_model_map_data(
                        baci_df, _mining_refining_df3, yr,
                        _dataset_key, _pb_factors_tuple, _mining_pref,
                        _map_eta_secondary, _map_eta_break, _map_delta_pb,
                        _map_beta, _map_eta_mfg, _map_eta_ore, _map_gamma,
                    )
                    _mk = _METRIC_KEYS[prod_dataset]
                    _ll, _vv = [], []
                    for _c, _o in _od.items():
                        _v = max(0.0, _o.get(_mk, 0.0))
                        if _v > 0:
                            _ll.append(_c)
                            _vv.append(_v)
                    return _ll, _vv
                else:
                    _df = _mining_refining_df3[_mining_refining_df3["year"] == yr].copy()
                    if _mr_col not in _df.columns:
                        return [], []
                    _df = _df[_df[_mr_col].notna() & (_df[_mr_col] > 0)]
                    return _df["country"].tolist(), _df[_mr_col].tolist()

            # ── Compute global log scale bounds across all years ───────────────────
            # Sample two boundary years to find a stable scale; full scan for animation
            _scale_years = _prod_years if prod_animate else [
                _prod_years[-1] if _prod_years else 2022
            ]
            _all_scale_vals: list[float] = []
            for _sy in _scale_years:
                _, _sv = _get_frame_data(_sy)
                _all_scale_vals.extend([v for v in _sv if v and v > 0])

            if not _all_scale_vals:
                st.info(f"No data available for {prod_dataset}.")
            else:
                _g_zmin_log = _math.floor(_math.log10(min(_all_scale_vals)))
                _g_zmax_log = _math.ceil(_math.log10(max(_all_scale_vals)))

                # ── Static map ────────────────────────────────────────────────────
                if not prod_animate:
                    _default_yr = 2022 if 2022 in _prod_years else _prod_years[-1]
                    prod_year = st.slider(
                        "Year",
                        min_value=_prod_years[0], max_value=_prod_years[-1],
                        value=_default_yr, step=1, key="prod_static_year",
                    )
                    _locs, _zvals = _get_frame_data(prod_year)
                    if not _locs:
                        st.info(f"No data available for {prod_dataset} in {prod_year}.")
                    else:
                        fig_prod = _make_choropleth(
                            locations=_locs,
                            z_values=_zvals,
                            title=f"{prod_dataset} — {prod_year}",
                            colorscale=_colorscale,
                            colorbar_title="t Pb",
                            height=520,
                            log_scale=True,
                            zmin_log=_g_zmin_log,
                            zmax_log=_g_zmax_log,
                        )
                        st.plotly_chart(fig_prod, use_container_width=True)

                        # ── Ranked table (reported data only) ────────────────────
                        if prod_dataset not in _ESTIMATED_DATASETS:
                            _ranked_df = (
                                pd.DataFrame({"Country": _locs, "t Pb": _zvals})
                                .sort_values("t Pb", ascending=False)
                                .reset_index(drop=True)
                            )
                            _ranked_df.insert(0, "Rank", range(1, len(_ranked_df) + 1))
                            with st.expander(f"Country ranking — {prod_dataset}, {prod_year}", expanded=False):
                                st.dataframe(
                                    _ranked_df,
                                    hide_index=True,
                                    use_container_width=True,
                                    column_config={
                                        "t Pb": st.column_config.NumberColumn(
                                            format="%,.0f", help="Metric tonnes of lead content"
                                        )
                                    },
                                )

                # ── Animated map ──────────────────────────────────────────────────
                else:
                    _geo_layout = dict(
                        showframe=False,
                        showcoastlines=True,
                        coastlinecolor="#aaaaaa",
                        showland=True,
                        landcolor="#d8d8d8",
                        showocean=True,
                        oceancolor="#e3f2fd",
                        showlakes=False,
                        projection_type="natural earth",
                    )
                    _log_tickvals = list(range(int(_g_zmin_log), int(_g_zmax_log) + 1))
                    _log_ticktext = [f"{10**v:,.0f}" for v in _log_tickvals]

                    def _anim_trace(locs, orig_vals):
                        _lz = [
                            _math.log10(v) if v and v > 0 else float("nan")
                            for v in orig_vals
                        ]
                        return go.Choropleth(
                            locations=locs,
                            z=_lz,
                            customdata=orig_vals,
                            locationmode="country names",
                            colorscale=_colorscale,
                            zmin=_g_zmin_log,
                            zmax=_g_zmax_log,
                            colorbar=dict(
                                title="t Pb", len=0.6, thickness=14,
                                tickvals=_log_tickvals, ticktext=_log_ticktext,
                            ),
                            hovertemplate="%{location}: %{customdata:,.0f} t Pb<extra></extra>",
                            marker_line_color="#ffffff",
                            marker_line_width=0.4,
                        )

                    _init_locs, _init_zvals = _get_frame_data(_prod_years[0])
                    _frames = []
                    _slider_steps = []
                    for _yr in _prod_years:
                        _fl, _fz = _get_frame_data(_yr)
                        _frames.append(go.Frame(data=[_anim_trace(_fl, _fz)], name=str(_yr)))
                        _slider_steps.append(dict(
                            args=[[str(_yr)], dict(
                                frame=dict(duration=750, redraw=True),
                                mode="immediate",
                                transition=dict(duration=200),
                            )],
                            label=str(_yr),
                            method="animate",
                        ))

                    fig_anim = go.Figure(
                        data=[_anim_trace(_init_locs, _init_zvals)],
                        frames=_frames,
                    )
                    fig_anim.update_layout(
                        height=580,
                        margin={"r": 0, "t": 36, "l": 0, "b": 80},
                        title=dict(
                            text=f"{prod_dataset}",
                            x=0.5, font=dict(size=14),
                        ),
                        geo=_geo_layout,
                        updatemenus=[dict(
                            type="buttons",
                            showactive=False,
                            x=0.05, y=-0.08,
                            xanchor="left", yanchor="top",
                            buttons=[
                                dict(
                                    label="▶  Play",
                                    method="animate",
                                    args=[None, dict(
                                        frame=dict(duration=750, redraw=True),
                                        fromcurrent=True,
                                        transition=dict(duration=200),
                                    )],
                                ),
                                dict(
                                    label="⏸  Pause",
                                    method="animate",
                                    args=[[None], dict(
                                        frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0),
                                    )],
                                ),
                            ],
                        )],
                        sliders=[dict(
                            active=0,
                            steps=_slider_steps,
                            x=0.0, y=0,
                            len=1.0,
                            xanchor="left", yanchor="top",
                            pad=dict(b=10, t=55),
                            currentvalue=dict(
                                prefix="Year: ",
                                visible=True,
                                xanchor="center",
                                font=dict(size=13),
                            ),
                            transition=dict(duration=200),
                        )],
                    )
                    st.plotly_chart(fig_anim, use_container_width=True)

            st.caption(_caption)

    # ── Lead Accumulation ─────────────────────────────────────────────────────

    with _mb_accum:
        _la_hdr_col, _la_lm_col = st.columns([8, 1])
        with _la_hdr_col:
            st.subheader("Lead Mass Accumulation")
            st.caption(
                "Annual net lead balance by country or sub-region: "
                "mining production + imports − exports, broken down by product category. "
                "All values in kt Pb (kilotonnes of lead content)."
            )
        with _la_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data sources:** BACI trade flows + BGS/USGS mining production.\n\n"
                    "**How it's used:** Net balance = mined Pb + imported Pb − exported Pb, "
                    "summed across HS codes within each product category. A positive value "
                    "means more lead entered the country than left (net accumulator). A "
                    "negative value means more left than entered (net depleter — typical "
                    "for processing hubs that import raw material and export refined product).\n\n"
                    "Note: this is a trade-based approximation. It does not account for "
                    "domestic consumption, stockpiles, or informal flows."
                )
        render_lead_accumulation_tab(
            REGION_MAP, REGIONS_ORDERED,
            sidebar_year=year, dataset=_dataset_key,
            pb_factors=pb_factors, mining_source=_mining_pref,
        )


    # ── Process Estimates ─────────────────────────────────────────────────────

    with _mb_process:
        _pe_desc_col, _pe_lm_col = st.columns([8, 1])
        with _pe_desc_col:
            st.write(
                "Material-flow Sankey for a selected country. Shows how lead flows "
                "through collection, breaking, secondary smelting, manufacturing, and "
                "installation, with diagnostic flags where the model detects imbalances. "
                "Available for many countries, but **not reconciled against external "
                "observations** — see the **Mass Balance** tab for a fully reconciled "
                "model (India only for now)."
            )
        with _pe_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data sources:** BACI trade flows + BGS/USGS production + "
                    "Eurostat collection data (where available).\n\n"
                    "**How it's used:** The model chains five equations "
                    "(collection → breaking → secondary smelting → manufacturing → "
                    "installation) using BACI as the trade anchor and production data as "
                    "the refining anchor. Process efficiencies (η) scale each stage. "
                    "Diagnostic flags highlight:\n"
                    "- **D2:** Break-to-smelt ratio (>1.3 = excess scrap leaving the country)\n"
                    "- **D3:** Installation gap (model vs. reported installation)\n"
                    "- **D5:** Feedstock coverage ratio (unexplained smelting feedstock)\n\n"
                    "**Process Estimates vs. Mass Balance:** This tab covers many "
                    "countries using a single forward pass with fixed parameters. The "
                    "**Mass Balance** tab additionally reconciles against an external "
                    "stock series and USGS primary/secondary refining splits via a "
                    "constrained optimizer, so its parameters are calibrated rather than "
                    "assumed. Treat Mass Balance results as more reliable for the "
                    "countries it covers."
                )
        _mining_refining_df = _load_mining_refining()
        _eurostat_df = _load_eurostat_collection()
        render_mass_balance_sankey_tab(
            baci_df         = baci_df,
            mining_df       = _mining_refining_df,
            region_map      = REGION_MAP,
            regions_ordered = REGIONS_ORDERED,
            sidebar_year    = year,
            active_years    = active_years,
            dataset         = _dataset_key,
            pb_factors      = pb_factors,
            mining_source   = _mining_pref,
            eurostat_df     = _eurostat_df,
        )


    # ── Economy Snapshot ──────────────────────────────────────────────────────

    with _mb_snapshot:
        _es_desc_col, _es_lm_col = st.columns([8, 1])
        with _es_desc_col:
            st.write(
                "Summary scorecard of a country's lead recycling economy. "
                "Combines trade-derived and model-estimated metrics to characterize "
                "where a country sits in the global recycling system."
            )
        with _es_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data sources:** BACI trade flows + BGS/USGS production + "
                    "mass-balance model outputs.\n\n"
                    "**How it's used:** Key metrics (collection rate, secondary smelting "
                    "share, battery self-sufficiency, etc.) are derived from the same "
                    "mass-balance equations as the Process Estimates tab, but presented "
                    "as a compact summary rather than a full Sankey. "
                    "Use this tab to quickly compare countries or identify which stage "
                    "in the recycling loop is under-performing."
                )
        from visualizations.mass_balance_sankey import render_economy_snapshot_tab
        _mining_refining_df2 = _load_mining_refining()
        render_economy_snapshot_tab(
            baci_df         = baci_df,
            mining_df       = _mining_refining_df2,
            region_map      = REGION_MAP,
            regions_ordered = REGIONS_ORDERED,
            active_years    = active_years,
            dataset         = _dataset_key,
            pb_factors      = pb_factors,
            mining_source   = _mining_pref,
        )


    # ── Mass Balance (India v4 three-stage chain) ───────────────────────────

    with _mb_mass_balance:
        st.warning(
            "🚧 **Under Construction** — The Mass Balance tab is an active "
            "work-in-progress and not ready for general use. The India "
            "calibration is still being refined (smelt-vs-USGS fit, install "
            "equality feasibility, and parameter rail diagnostics). Numbers "
            "below may change."
        )
        render_india_v4_tab()

        with st.expander(
            "v3 calibration explorer (legacy fallback)",
            expanded=False,
        ):
            st.caption(
                "The v3 India calibration explorer is preserved as a fallback. "
                "It uses the previous two-stage chain (no explicit refine stage; "
                "780199 in FEED; γ, β, and the φ's all time-varying / fitted) "
                "and remains the working reference for cross-country scenario "
                "work until v4 is extended beyond India."
            )
            render_india_calibration_explorer_tab(
                baci_df       = baci_df,
                mining_df     = _load_mining_refining(),
                pb_factors    = pb_factors,
                active_years  = active_years,
                time_period   = time_period,
                mining_source = _mining_pref,
                dataset_key   = _dataset_key,
            )


# ── India Lead Model tab (hidden — preserved for later re-activation) ─────────
# To restore: add "India Lead Model" as a 7th entry in st.tabs() above,
# capture it as `tab_india`, wrap the block below in `with tab_india:`,
# and remove the `if False:` line.
if False:  # noqa: dead-code  pylint: disable=using-constant-test
    st.subheader("India Lead-Acid Battery Mass Balance Model")
    st.caption(
        "Estimates formal vs. informal recycling pathways using USGS production data, "
        "BACI trade flows, and a 7-parameter mass balance. τ = 3 yr fixed battery lifespan lag."
    )

    # ── Data loading ──────────────────────────────────────────────────────────
    _india_csvs = _load_india_csvs_cached()
    _india_net_trade, _india_trade_warnings = _build_india_net_trade_cached(baci_df)

    _tau = st.slider("τ — battery lifespan lag (years)", 2, 7, 4, 1, key="ind_tau")

    _india_data, _india_valid_years, _india_missing_lag = india_prepare_inputs(
        _india_csvs, _india_net_trade, tau=_tau
    )

    if _india_trade_warnings:
        with st.expander("BACI data warnings", expanded=False):
            for _w in _india_trade_warnings:
                st.warning(_w)

    if _india_missing_lag:
        for _y, _lag_y in sorted(_india_missing_lag.items()):
            st.error(
                f"Install data missing for year {_lag_y}. "
                f"Required for τ={_tau} lag (model year {_y})."
            )

    if not _india_valid_years:
        st.error(
            "No valid model years available. "
            f"Provide install estimates covering lag years (model year − {_tau})."
        )
    else:
        # Note about install data interpretation
        _install_vals = list(_india_csvs["install"]["value"])
        if _install_vals and max(_install_vals) > 2e6:
            st.info(
                "**Note — install estimates appear to be total installed stock, not annual flow.** "
                "The install CSV values (~{:,.0f}–{:,.0f} t Pb) are consistent with India's "
                "total lead-acid battery stock, not annual new installations (~500k–1,000k t/yr). "
                "Large residuals in Section 1 reflect this mismatch. "
                "To fix: provide annual installation flow in the CSV, or divide stock values "
                "by average battery lifetime (~4 years).".format(
                    min(_install_vals), max(_install_vals)
                )
            )

        # ── Parameters ────────────────────────────────────────────────────────
        st.markdown(
            "**Model Parameters** — Adjust sliders then click **Solve** to re-fit γ_F(t)."
        )
        _pc1, _pc2, _pc3, _pc4 = st.columns(4)
        with _pc1:
            _p_gamma_disp = st.slider(
                "γ_disposal (disposal rate)", 0.00, 0.15, 0.05, 0.005, format="%.3f",
                key="ind_gamma_disposal",
            )
            _p_eta_smelt_I = st.slider(
                "η_smelt_I (informal smelting)", 0.40, 0.95, 0.70, 0.01,
                key="ind_eta_smelt_i",
            )
        with _pc2:
            _p_eta_break_F = st.slider(
                "η_break_F (formal breaking)", 0.80, 1.00, 0.95, 0.01,
                key="ind_eta_break_f",
            )
            _p_eta_refine = st.slider(
                "η_refine (secondary refining)", 0.95, 1.00, 0.99, 0.005, format="%.3f",
                key="ind_eta_refine",
            )
        with _pc3:
            _p_eta_break_I = st.slider(
                "η_break_I (informal breaking)", 0.40, 0.95, 0.70, 0.01,
                key="ind_eta_break_i",
            )
            _p_beta = st.slider(
                "β (battery share of Pb demand)", 0.60, 0.95, 0.75, 0.01,
                key="ind_beta",
            )
        with _pc4:
            _p_eta_smelt_F = st.slider(
                "η_smelt_F (formal smelting)", 0.85, 1.00, 0.97, 0.01,
                key="ind_eta_smelt_f",
            )
            _p_lead_loss = st.slider(
                "lead_loss_rate", 0.00, 0.05, 0.02, 0.005, format="%.3f",
                key="ind_lead_loss",
            )

        _pgam_col, _pphi1, _pphi2 = st.columns(3)
        with _pgam_col:
            _p_gamma_F = st.slider(
                "γ_F (formal collection rate)", 0.00, 1.00, 0.50, 0.01,
                key="ind_gamma_f",
            )
        with _pphi1:
            _p_phi_B = st.slider(
                "φ_B (informal → formal breaking)", 0.00, 1.00, 0.70, 0.01,
                key="ind_phi_b",
            )
        with _pphi2:
            _p_phi_S = st.slider(
                "φ_S (informal → formal smelting)", 0.00, 1.00, 0.70, 0.01,
                key="ind_phi_s",
            )

        _india_fixed_params = {
            "gamma_disposal": _p_gamma_disp,
            "gamma_F": _p_gamma_F,
            "eta_break_F": _p_eta_break_F,
            "eta_break_I": _p_eta_break_I,
            "eta_smelt_F": _p_eta_smelt_F,
            "eta_smelt_I": _p_eta_smelt_I,
            "eta_refine": _p_eta_refine,
            "beta": _p_beta,
            "lead_loss_rate": _p_lead_loss,
            "phi_B": _p_phi_B,
            "phi_S": _p_phi_S,
        }

        # ── Forward model (runs on every slider change) ────────────────────────
        _india_df = india_forward_model(
            _india_data, _india_valid_years, _india_fixed_params
        )

        st.divider()

        # ── Section 1: Model fit summary ──────────────────────────────────────
        st.markdown("#### 1 — Model Fit Summary")

        _s1_cols = [
            "year", "gamma_F", "gamma_I", "phi_B", "phi_S",
            "residual_sec_refine", "residual_install",
        ]
        _s1_df = _india_df[_s1_cols].copy()

        def _residual_row_style(row):
            styles = [""] * len(row)
            _pairs = [
                ("residual_sec_refine",
                 _india_data[int(row["year"])]["secondary_refined_obs"]),
                ("residual_install",
                 _india_data[int(row["year"])]["install_obs"]),
            ]
            for _col, _obs in _pairs:
                if _col in row.index and _obs != 0:
                    _pct = abs(row[_col] / _obs)
                    _ci = list(row.index).index(_col)
                    if _pct < 0.05:
                        styles[_ci] = "background-color: #c6efce; color: #276221"
                    elif _pct < 0.15:
                        styles[_ci] = "background-color: #ffeb9c; color: #7d6008"
                    else:
                        styles[_ci] = "background-color: #ffc7ce; color: #9c0006"
            return styles

        _s1_fmt = {
            "gamma_F": "{:.4f}", "gamma_I": "{:.4f}",
            "phi_B": "{:.4f}", "phi_S": "{:.4f}",
            "residual_sec_refine": "{:,.0f}",
            "residual_install": "{:,.0f}",
        }
        st.dataframe(
            _s1_df.style.apply(_residual_row_style, axis=1).format(_s1_fmt),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"φ_B = {_p_phi_B:.4f}  |  φ_S = {_p_phi_S:.4f}  "
            "(year-invariant — set via sliders)"
        )
        st.caption(
            "Residual colours: green < 5% of observed | amber < 15% | red ≥ 15%"
        )

        st.divider()

        # ── Section 2: Stage-by-stage mass flow ───────────────────────────────
        st.markdown("#### 2 — Stage-by-Stage Mass Flow (tonnes Pb)")

        _s2_cols = [
            # Use / collection stage
            "year", "USE_eol", "net_trade_use",
            "COLL_F", "COLL_I", "DISPOSED", "net_trade_collection",
            # Breaking stage
            "BREAK_F_out", "BREAK_I_out",
            # Smelting stage
            "net_trade_scrap", "SMELT_F_out", "SMELT_I_out",
            # Secondary refining
            "net_trade_smelted", "SEC_REFINED",
            # Refined pool and manufacturing
            "net_trade_refined", "POOL", "BATTERY_MFG", "NON_BATTERY", "implied_install",
        ]
        _s2_df = _india_df[_s2_cols].copy()
        _s2_fmt = {c: "{:+,.0f}" if c.startswith("net_trade") else "{:,.0f}"
                   for c in _s2_cols if c != "year"}
        st.dataframe(
            _s2_df.style.format(_s2_fmt),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "net_trade columns: positive = net imports, negative = net exports. "
            "net_trade_use = batteries (850710/20); collection = waste batteries (854810); "
            "scrap = 780200; smelted = 780199; refined = 780110/91 + 282410/90."
        )

        st.divider()

        # ── Section 3: Charts ──────────────────────────────────────────────────
        st.markdown("#### 3 — Charts")

        # Chart A — Collection split
        _fig_a = go.Figure()
        _fig_a.add_trace(go.Bar(
            name="Formal collected",
            x=_india_df["year"], y=_india_df["COLL_F"],
            marker_color="#1E88E5",
        ))
        _fig_a.add_trace(go.Bar(
            name="Informal collected",
            x=_india_df["year"], y=_india_df["COLL_I"],
            marker_color="#FB8C00",
        ))
        _fig_a.add_trace(go.Bar(
            name="Disposed (no recycling)",
            x=_india_df["year"], y=_india_df["DISPOSED"],
            marker_color="#E53935",
        ))
        _fig_a.update_layout(
            barmode="stack",
            title="A — Collection Split",
            xaxis_title="Year",
            yaxis_title="tonnes Pb",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=380,
        )
        st.plotly_chart(_fig_a, use_container_width=True)

        # Chart B — Lead losses by stage
        _fig_b = go.Figure()
        _fig_b.add_trace(go.Bar(
            name="Degradation loss",
            x=_india_df["year"], y=_india_df["degradation_loss"],
            marker_color="#9E9E9E",
        ))
        _fig_b.add_trace(go.Bar(
            name="Disposal",
            x=_india_df["year"], y=_india_df["DISPOSED"],
            marker_color="#E53935",
        ))
        _fig_b.add_trace(go.Bar(
            name="Breaking losses",
            x=_india_df["year"], y=_india_df["BREAK_loss"],
            marker_color="#FB8C00",
        ))
        _fig_b.add_trace(go.Bar(
            name="Smelting losses",
            x=_india_df["year"], y=_india_df["SMELT_loss"],
            marker_color="#F9A825",
        ))
        _fig_b.add_trace(go.Bar(
            name="Refining losses",
            x=_india_df["year"], y=_india_df["SEC_REFINE_loss"],
            marker_color="#FDD835",
        ))
        _fig_b.add_trace(go.Bar(
            name="Non-battery sink",
            x=_india_df["year"], y=_india_df["NON_BATTERY"],
            marker_color="#43A047",
        ))
        _fig_b.update_layout(
            barmode="stack",
            title="B — Lead Losses by Stage",
            xaxis_title="Year",
            yaxis_title="tonnes Pb",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=380,
        )
        st.plotly_chart(_fig_b, use_container_width=True)

        # Chart C — Model vs observed
        _fig_c = go.Figure()
        _fig_c.add_trace(go.Scatter(
            name="Implied install (model)",
            x=_india_df["year"], y=_india_df["implied_install"],
            mode="lines+markers",
            line=dict(color="#1E88E5", width=2),
        ))
        _fig_c.add_trace(go.Scatter(
            name="Install estimate (CSV)",
            x=_india_df["year"], y=_india_df["install_obs"],
            mode="lines+markers",
            line=dict(color="#1E88E5", width=2, dash="dash"),
        ))
        _fig_c.add_trace(go.Scatter(
            name="Sec. refined (model)",
            x=_india_df["year"], y=_india_df["SEC_REFINED"],
            mode="lines+markers",
            line=dict(color="#43A047", width=2),
        ))
        _fig_c.add_trace(go.Scatter(
            name="Sec. refined (observed)",
            x=_india_df["year"], y=_india_df["secondary_refined_obs"],
            mode="lines+markers",
            line=dict(color="#43A047", width=2, dash="dash"),
        ))
        _fig_c.update_layout(
            title="C — Model vs Observed",
            xaxis_title="Year",
            yaxis_title="tonnes Pb",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=380,
        )
        st.plotly_chart(_fig_c, use_container_width=True)

        st.divider()

        # ── Section 4: Informal economy diagnostic ─────────────────────────────
        st.markdown("#### 4 — Informal Economy Diagnostic")

        _avg_gamma_I = float(_india_df["gamma_I"].mean())
        _avg_gamma_total = float(_india_df["gamma_total"].mean())
        _avg_coll_I = float(_india_df["COLL_I"].mean())
        _avg_break_I = float(_india_df["BREAK_I_in"].mean())
        _avg_smelt_I = float(_india_df["SMELT_I_in"].mean())
        _avg_informal = _avg_coll_I + _avg_break_I + _avg_smelt_I

        _all_res = np.concatenate([
            _india_df["residual_sec_refine"].values,
            _india_df["residual_install"].values,
        ])
        _rmse = float(np.sqrt(np.mean(_all_res ** 2)))

        _infml_share = (
            f"{100 * _avg_gamma_I / _avg_gamma_total:.1f}%"
            if _avg_gamma_total > 0 else "—"
        )

        _d1, _d2 = st.columns(2)
        with _d1:
            st.metric("Informal share of collection (avg)", _infml_share)
            st.metric(
                "Informal batteries → formal breaking (φ_B)",
                f"{100 * _p_phi_B:.1f}%",
            )
            st.metric(
                "Informal scrap → formal smelting (φ_S)",
                f"{100 * _p_phi_S:.1f}%",
            )
        with _d2:
            st.metric(
                "Avg. lead through informal pathway",
                f"{_avg_informal / 1000:.1f} kt Pb",
            )
            st.metric("RMSE (both residuals)", f"{_rmse:,.0f} t Pb")

    st.caption("All values in metric tonnes of lead (t Pb).")
