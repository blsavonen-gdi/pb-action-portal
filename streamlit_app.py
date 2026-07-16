"""
streamlit_app.py — Pb Action Portal (public)

The public, no-password Portal: a simplified subset of the Pb Action Toolkit.
One flat level of tabs (no high-level grouping). A global Easy/Advanced mode
toggle (sidebar) controls how much is exposed: Easy mode fixes the data
assumptions and hides the detailed controls; Advanced mode exposes every input.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from model.data_loader import load_baci, load_smelter_capacity, CATEGORY_MAP
from visualizations.trade_map import (
    build_total_volume_map, build_bilateral_map, build_bilateral_balance_map,
    build_region_bilateral_map, build_region_balance_map,
)
from visualizations.flow_network import build_flow_network
from visualizations.flow_network_interactive import build_flow_network_html
import streamlit.components.v1 as components
from model.regions import REGIONS_ORDERED, REGION_MAP, MAJOR_REGIONS_ORDERED, MAJOR_REGION_MAP
from visualizations.lead_accumulation import (
    render_lead_accumulation_tab, HS_META, CATEGORIES_ORDERED, _DEFAULT_OFF,
)
from visualizations.mass_balance_sankey import (
    render_mass_balance_sankey_tab,
    EASY_MODEL_EQUATIONS_MD as _EASY_MODEL_EQUATIONS_MD,
)

st.set_page_config(
    page_title="Pb Action Data Portal",
    page_icon="🔋",
    layout="wide",
)

# ── Unified-app page imports ──────────────────────────
from literature.app import render as render_literature


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

# ── Advanced (6-category) grouping ────────────────────────────────────────────
# The Advanced-mode product grouping mirrors the sidebar "slide-out" (Pb content
# factors) grouping: every lead HS code bucketed into one of the six material-
# flow categories from HS_META. Colors reuse the shared 5-cat values where they
# overlap (grey/blue/green/yellow) and add brown/purple for the two advanced-
# only categories. Slag and Other Lead Products are excluded by default on every
# tab (their codes exist only in the HS22 dataset).
ADV_CAT_COLORS = {
    "Mining Outputs":      "#9E9E9E",  # grey   (= Ore & Concentrates)
    "Battery Inputs":      "#1E88E5",  # blue   (= Smelted Lead)
    "New Batteries":       "#43A047",  # green
    "Battery Waste":       "#FDD835",  # yellow (= Used Batteries)
    "Slag":                "#8D6E63",  # brown
    "Other Lead Products": "#7E57C2",  # purple
}
ADV_CATS_DEFAULT_OFF = frozenset({"Slag", "Other Lead Products"})
# HS codes off by default (Advanced): every code in an off-by-default category.
_ADV_DEFAULT_OFF_HS = {
    hs for hs, m in HS_META.items() if m["cat"] in ADV_CATS_DEFAULT_OFF
}


def _adv_cat_of(hs) -> str | None:
    """Return the 6-category (material-flow) label for an HS code, or None."""
    return HS_META.get(int(hs), {}).get("cat")


def _advanced_hs_picker(present_products: list[int], key_prefix: str) -> list[int]:
    """Render per-HS tick boxes grouped by the six slide-out categories and
    return the selected HS codes. Slag and Other Lead Products default to
    unchecked. Only codes present in the active dataset are shown."""
    st.caption("Products to include (by HS code)")
    selected: list[int] = []
    for _cat in CATEGORIES_ORDERED:
        _codes = [p for p in present_products if _adv_cat_of(p) == _cat]
        if not _codes:
            continue
        st.markdown(f"**{_cat}**")
        _default_on = _cat not in ADV_CATS_DEFAULT_OFF
        _cols = st.columns(3)
        for _i, _hs in enumerate(_codes):
            with _cols[_i % 3]:
                if st.checkbox(
                    f"{_hs} — {HS_META[_hs]['name']}",
                    value=_default_on,
                    key=f"{key_prefix}_hs_{_hs}",
                ):
                    selected.append(_hs)
    return selected

st.markdown("""
<style>
/* ── Tighten the top of the page so the header hugs the top ───────────── */
.block-container { padding-top: 3.2rem; }

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
    grouping: str = "trade",
    products: tuple[int, ...] | None = None,
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
        grouping=grouping,
        products=products,
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
    grouping: str = "trade",
    products: tuple[int, ...] | None = None,
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
        grouping=grouping,
        products=products,
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
def _load_mining_refining() -> pd.DataFrame:
    """Country x year mining/refining panel, built by outer-joining the SEPARATE
    USGS and BGS extracts. USGS and BGS are NEVER reconciled - each keeps its own
    column (mined_usgs_t vs mined_bgs_t, etc.). Rows are kept only where some
    source reports a positive value, for years >= 2012. Replaces the former
    pre-joined data/country_year_mining_refining.csv."""
    ref = "data/Reference"
    vc = ["mined_usgs_t", "mined_bgs_t", "refined_bgs_t",
          "refined_primary_usgs_t", "refined_secondary_usgs_t"]
    try:
        mine = pd.read_csv(f"{ref}/USGS mined.csv", usecols=["country", "year", "value_metric_t"]).rename(columns={"value_metric_t": "mined_usgs_t"})
        rp = pd.read_csv(f"{ref}/USGS refined_primary.csv", usecols=["country", "year", "value_metric_t"]).rename(columns={"value_metric_t": "refined_primary_usgs_t"})
        rs = pd.read_csv(f"{ref}/USGS refined_secondary.csv", usecols=["country", "year", "value_metric_t"]).rename(columns={"value_metric_t": "refined_secondary_usgs_t"})
        bgs = pd.read_csv(f"{ref}/BGS Refined and Smelted.csv", encoding="utf-8-sig")
    except FileNotFoundError:
        return pd.DataFrame(columns=["country", "year"] + vc)
    bm = bgs[bgs["bgs_commodity_trans"] == "lead, mine"][["country_trans", "year", "Mass of Pb"]].rename(columns={"country_trans": "country", "Mass of Pb": "mined_bgs_t"})
    br = bgs[bgs["bgs_commodity_trans"] == "lead, refined"][["country_trans", "year", "Mass of Pb"]].rename(columns={"country_trans": "country", "Mass of Pb": "refined_bgs_t"})
    df = mine
    for other in (bm, br, rp, rs):
        df = df.merge(other, on=["country", "year"], how="outer")
    df = df[((df[vc] > 0).any(axis=1)) & (df["year"] >= 2012)]
    return df[["country", "year"] + vc].sort_values(["country", "year"]).reset_index(drop=True)


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


# ── UI mode (Easy vs Advanced) ────────────────────────────────────────────────
# Resolved here (before the data load) using the same deferred-read pattern as
# the dataset selector: the sidebar toggle writes st.session_state["ui_mode"];
# this read picks it up on the next rerun. Easy mode fixes a set of data
# assumptions (HS12 / BGS / 3-year average / default Pb factors) and hides the
# advanced controls; Advanced mode exposes every input.
_ui_mode = st.session_state.get("ui_mode", "Easy")
ADVANCED = _ui_mode == "Advanced"

_dataset = st.session_state.get("baci_dataset", "HS12 (2012–2024)")
# Easy mode is always HS12; only Advanced mode honours an HS22 selection.
_dataset_key = "hs22" if (ADVANCED and _dataset.startswith("HS22")) else "hs12"
baci_df = _load_baci_cached(_dataset_key)


def _sig2(v: float) -> float:
    """Round a model-derived value to 2 significant figures (UI display)."""
    import math
    if v is None or v == 0 or (isinstance(v, float) and math.isnan(v)):
        return 0.0
    exp = math.floor(math.log10(abs(v)))
    factor = 10 ** (exp - 1)
    return round(v / factor) * factor


def _easy_assumptions_footer():
    """In Easy mode, render the fixed data assumptions in light grey at the
    bottom of a data tab. No-op in Advanced mode (there the user set these)."""
    if ADVANCED:
        return
    st.markdown(
        "<p style='color:#9aa0a6;font-size:12px;margin-top:1.6rem;'>"
        "Data based on: BACI HS12 trade data, BGS mining/refining data, "
        "3-year averages centered on the target year to reduce noise, and "
        "standard assumptions on lead content per product."
        "</p>",
        unsafe_allow_html=True,
    )

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
        st.markdown("## Pb Action Data Portal")
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
    st.markdown("### 🧭 Easy mode vs. Advanced mode")
    st.write(
        "Use the **View mode** toggle in the sidebar to switch between two ways of working:\n\n"
        "- **Easy mode** makes sensible data assumptions for you and hides the detailed "
        "controls — the fastest way to get a clear, reliable picture.\n"
        "- **Advanced mode** exposes every input (dataset, data sources, time period, and "
        "lead-content factors), so you can adjust the model and challenge its underlying "
        "assumptions.\n\n"
        "It's a tradeoff between **usability** and **control**: start in Easy mode to get "
        "oriented, then switch to Advanced when you want to test how the numbers respond."
    )
    st.divider()
    if st.button("I understand — enter the app", type="primary"):
        st.session_state.disclaimer_accepted = True
        st.rerun()
    st.stop()

# ── Compact app header (logo + title on one tight row; shown on every tab) ───
_hdr_logo, _hdr_txt = st.columns([1, 13], vertical_alignment="center")
with _hdr_logo:
    st.image("Pb Action Logo - Primary.png", width=64)
with _hdr_txt:
    st.markdown(
        "<div style='line-height:1.2;margin:0;'>"
        "<span style='font-size:1.5rem;font-weight:700;'>Pb Action Data Portal</span><br>"
        "<span style='color:#888;font-size:0.85rem;'>Partnership for Battery Action "
        "(Pb Action / GDI) &middot; all values in metric tonnes of lead content</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Top navigation (one flat level of tabs) ───────────────────────────────
_nav_options = ["Trade Map", "Trade Trends", "Trade Relationships", "Lead Accumulation", "Production & Capacity", "Recycling Economy Snapshot (Beta 🧪)", "Material Flow (Beta 🧪)", "Literature Stats"]

# The seven data tabs share the same sidebar controls + data prep; Literature
# Stats is standalone.
_DATA_TABS = ("Trade Map", "Trade Trends", "Trade Relationships", "Production & Capacity", "Lead Accumulation", "Recycling Economy Snapshot (Beta 🧪)", "Material Flow (Beta 🧪)")

# If a persisted selection is no longer valid, fall back to the first tab.
if st.session_state.get("toolkit_page") not in _nav_options:
    st.session_state["toolkit_page"] = _nav_options[0]

_page = st.segmented_control(
    "Section",
    _nav_options,
    label_visibility="collapsed",
    key="toolkit_page",
)
if not _page:
    _page = _nav_options[0]

# ── Sidebar: mode toggle first (shown on every tab), then advanced controls ──
with st.sidebar:
    st.radio(
        "View mode",
        options=["Easy", "Advanced"],
        key="ui_mode",
        help=(
            "Easy mode makes sensible data assumptions for you and hides the "
            "detailed controls. Advanced mode lets you choose the dataset, data "
            "sources, time period, and lead-content factors yourself."
        ),
    )
    if ADVANCED:
        st.caption(
            "**Advanced mode** — you control the dataset, data sources, time "
            "period, and lead-content factors below."
        )
    else:
        st.caption(
            "**Easy mode** makes a number of data assumptions for you (listed in "
            "grey at the bottom of each tab). Switch to **Advanced** to control "
            "every input."
        )
    st.divider()

# ── Data parameters for the trade / material-flow tabs ───────────────────────
# Advanced mode renders the full control set in the sidebar. Easy mode shows
# only the year slider (always a 3-year average) and fixes every other input to
# the recommended assumptions. The Easy-mode pb_factors MUST match what Advanced
# produces at its default widget states: codes in _DEFAULT_OFF are excluded
# (factor 0.0), everything else uses its default factor. Any mismatch here makes
# Easy and Advanced disagree on identical settings across every tab.
if _page in _DATA_TABS and not ADVANCED:
    with st.sidebar:
        _default_year = min(2022, max_year)
        year = st.slider(
            "Center year",
            min_value=min_year, max_value=max_year, value=_default_year, step=1,
            help="Easy mode always uses a 3-year average centered on this year to reduce noise.",
        )
        st.caption(
            "For some variables we use a 3-year average (centered on this year) to cut "
            "down on noise in the data. Switch to Advanced mode to hone in on a single year."
        )
    _mining_pref = "BGS"
    time_period = "3-year average (recommended)"
    active_years = [y for y in [year - 1, year, year + 1] if min_year <= y <= max_year]
    period_label = f"{active_years[0]}–{active_years[-1]} average"
    st.sidebar.caption(f"Window: {period_label}")
    pb_factors = {
        hs: (0.0 if hs in _ADV_DEFAULT_OFF_HS else meta["default"])
        for hs, meta in HS_META.items()
    }

if _page in _DATA_TABS and ADVANCED:
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
                    _default_on = _hs not in _ADV_DEFAULT_OFF_HS
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

# Waste-battery HS code differs between the HS12 and HS22 datasets.
_TT_UBATT_CODE = 854911 if _dataset_key == "hs22" else 854810
_TT_HS_ALL = {260700, 780110, 780191, 780199, 780200, 850710, 850720,
              850790, _TT_UBATT_CODE, 282410, 282490}

# Base series name (without the "Imports"/"Exports" suffix) → set of HS codes.
# Covers both the five trade categories (Easy) and the six material-flow
# categories (Advanced). "Smelted Lead" (Easy) and "Battery Inputs" (Advanced)
# are the same codes; "New Batteries" is shared by both groupings.
_TT_BASE_HS: dict[str, set[int]] = {
    "Total Lead-Related":  _TT_HS_ALL,
    # ── five trade categories (Easy) ──
    "New Batteries":       {850710, 850720},
    "Used Batteries":      {_TT_UBATT_CODE},
    "Lead Scrap":          {780200},
    "Smelted Lead":        {282410, 282490, 780110, 780191, 780199, 850790},
    "Ore & Concentrates":  {260700},
    # ── six material-flow categories (Advanced) ──
    "Mining Outputs":      {260700},
    "Battery Inputs":      {282410, 282490, 780110, 780191, 780199, 850790},
    "Battery Waste":       {780200, _TT_UBATT_CODE},
    "Slag":                {262021, 262029},
    "Other Lead Products": {780411, 780419, 780420, 780600},
}

# Mode-specific Series options.
_TT_SERIES_EASY: list[str] = [
    "Total Lead-Related Imports", "Total Lead-Related Exports",
    "New Batteries Imports", "New Batteries Exports",
    "Used Batteries Imports", "Used Batteries Exports",
    "Lead Scrap Imports", "Lead Scrap Exports",
    "Smelted Lead Imports", "Smelted Lead Exports",
    "Ore & Concentrates Imports", "Ore & Concentrates Exports",
    "Lead Mined", "Lead Refined",
]
_TT_SERIES_ADV: list[str] = [
    "Total Lead-Related Imports", "Total Lead-Related Exports",
    "Mining Outputs Imports", "Mining Outputs Exports",
    "Battery Inputs Imports", "Battery Inputs Exports",
    "New Batteries Imports", "New Batteries Exports",
    "Battery Waste Imports", "Battery Waste Exports",
    "Slag Imports", "Slag Exports",
    "Other Lead Products Imports", "Other Lead Products Exports",
    "Lead Mined", "Lead Refined/Smelted",
]

# Production series (BGS/USGS) — not derived from BACI trade flows.
_TT_PROD_SERIES: frozenset[str] = frozenset({
    "Lead Mined", "Lead Refined", "Lead Refined/Smelted", "Lead Smelted/Refined",
})

_TT_SERIES_COLORS: dict[str, str] = {
    "Total Lead-Related Imports":  "#1f77b4",
    "Total Lead-Related Exports":  "#ff7f0e",
    # five trade categories
    "New Batteries Imports":       "#43A047",
    "New Batteries Exports":       "#1B5E20",
    "Used Batteries Imports":      "#FDD835",
    "Used Batteries Exports":      "#F9A825",
    "Lead Scrap Imports":          "#FB8C00",
    "Lead Scrap Exports":          "#E65100",
    "Smelted Lead Imports":        "#1E88E5",
    "Smelted Lead Exports":        "#0D47A1",
    "Ore & Concentrates Imports":  "#9E9E9E",
    "Ore & Concentrates Exports":  "#616161",
    # six material-flow categories
    "Mining Outputs Imports":      "#9E9E9E",
    "Mining Outputs Exports":      "#616161",
    "Battery Inputs Imports":      "#1E88E5",
    "Battery Inputs Exports":      "#0D47A1",
    "Battery Waste Imports":       "#FDD835",
    "Battery Waste Exports":       "#F9A825",
    "Slag Imports":                "#8D6E63",
    "Slag Exports":                "#5D4037",
    "Other Lead Products Imports": "#7E57C2",
    "Other Lead Products Exports": "#4527A0",
    # production series
    "Lead Mined":                  "#6a3d9a",
    "Lead Refined":                "#b15928",
    "Lead Refined/Smelted":        "#b15928",
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
    # Strip the trailing "Imports"/"Exports" word to get the category base name.
    base = series.rsplit(" ", 1)[0]
    hs = _TT_BASE_HS.get(base)
    if not hs:
        return pd.Series(dtype=float)
    return baci[side & baci["Product"].isin(hs)].groupby("Year")["actual_lead"].sum()


def _tt_bgs_series(
    bgs: pd.DataFrame, countries: list[str], series: str
) -> pd.Series:
    commodity = "lead, mine" if "Mined" in series else "lead, refined"
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
    usgs_df = usgs_mined if "Mined" in series else usgs_refined
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
        if s in _TT_PROD_SERIES:
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



if _page in ("Trade Map", "Trade Trends", "Trade Relationships"):

    # ── Trade Analysis ────────────────────────────────────────────────────────

    if _page == "Trade Map":
        # ── Description ───────────────────────────────────────────────────────
        _ta_desc_col, _ta_lm_col = st.columns([8, 1])
        with _ta_desc_col:
            if ADVANCED:
                st.write(
                    "Explore how lead products move between countries and regions. "
                    "Pick a country or region to see its bilateral trade partners on the map, "
                    "and choose exactly which HS product codes to include."
                )
            else:
                st.write("Explore the lead-related trade relationships between countries.")
        with _ta_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data source:** BACI (Base pour l'Analyse du Commerce International), "
                    "published by CEPII. Harmonizes UN Comtrade import and export reports to "
                    "resolve bilateral discrepancies. Values are converted to tonnes of lead "
                    "content using standard Pb fraction factors.\n\n"
                    "**How it's used:** Each row in BACI is a bilateral flow "
                    "(exporter to importer, product, year). This tab aggregates those flows "
                    "and displays them on a world map.\n\n"
                    "**Selecting a place:** Choose *All countries* for a total-volume view, or "
                    "pick a single country or a UN region (regions are listed first, then "
                    "countries A-Z) to see who trades with it. When a region is selected, only "
                    "cross-region (external) flows are shown."
                )

        # ── Primary map slot: filled after all the controls below are read, so
        # the map sits directly under the intro while still reflecting their
        # state (place, color mode, product ticks). ──────────────────────────
        _map_slot = st.container()

        # ── Place selector: countries + regions in one dropdown ───────────────
        # Regions (continents, then UN sub-regions) are listed first, then all
        # countries alphabetically. Selection is dispatched three ways below:
        # All countries -> total-volume map; a region -> regional maps/tables;
        # a country -> bilateral maps/tables.
        _ALL_PLACES = "— All countries (total volume) —"
        _regions_all = list(MAJOR_REGIONS_ORDERED) + list(REGIONS_ORDERED)
        _place_options = [_ALL_PLACES] + _regions_all + list(all_baci_countries)
        place_sel = st.selectbox(
            "Select a country or region",
            options=_place_options,
            index=0,
            key="ta_place",
            help="Regions (continents and UN sub-regions) are listed first, then countries A-Z.",
        )

        _is_all = (place_sel == _ALL_PLACES)
        _is_region = place_sel in _regions_all
        _is_major = place_sel in MAJOR_REGIONS_ORDERED
        _region_map = MAJOR_REGION_MAP if _is_major else REGION_MAP

        # Color-scale mode is an Advanced-only control; Easy mode is always the
        # default global volume scale.
        if ADVANCED and not _is_all:
            map_color_mode = st.radio(
                "Color scale mode",
                options=["Global volume scale", "Bilateral trade balance"],
                index=0,
                horizontal=True,
                key="map_color_mode",
                help=(
                    "**Global volume scale**: color by net flow direction, intensity by volume.\n\n"
                    "**Bilateral trade balance**: color by the trade ratio with the selected "
                    "place, regardless of volume."
                ),
            )
        else:
            map_color_mode = "Global volume scale"

        # ── Product category / HS-code tick boxes (below the map) ─────────────
        _code_to_name = {code: _strip_cat_emoji(lbl) for lbl, code in _CAT_LABELS.items()}
        _present_products = sorted(int(p) for p in baci_df["Product"].dropna().unique())

        if ADVANCED:
            # Per-HS tick boxes grouped by the six slide-out categories;
            # Slag + Other Lead Products default off.
            _selected_hs = _advanced_hs_picker(_present_products, "ta")
            if not _selected_hs:
                _selected_hs = [p for p in _present_products
                                if _adv_cat_of(p) not in ADV_CATS_DEFAULT_OFF]
        else:
            st.caption("Product categories to include")
            _cat_cols = st.columns(len(_CAT_LABEL_LIST))
            _selected_cats_easy: list[str] = []
            for _i, _lbl in enumerate(_CAT_LABEL_LIST):
                with _cat_cols[_i]:
                    if st.checkbox(_lbl, value=True, key=f"ta_cat_{_CAT_LABELS[_lbl]}"):
                        _selected_cats_easy.append(_CAT_LABELS[_lbl])
            if not _selected_cats_easy:
                _selected_cats_easy = [_CAT_LABELS[l] for l in _CAT_LABEL_LIST]
            _selected_hs = [p for p in _present_products
                            if CATEGORY_MAP.get(p) in _selected_cats_easy]

        # Categories present among the selected HS codes (drives the category
        # filter passed to the map builders + the display label). Advanced uses
        # the six material-flow categories; Easy uses the five trade categories.
        if ADVANCED:
            _cats_present = [c for c in CATEGORIES_ORDERED
                             if any(_adv_cat_of(p) == c for p in _selected_hs)]
            _cat_display_label = ", ".join(_cats_present) or "no products"
        else:
            _cats_present = [c for c in ["BATT", "USED", "SCRAP", "FEED", "ORE"]
                             if any(CATEGORY_MAP.get(p) == c for p in _selected_hs)]
            _cat_display_label = ", ".join(_code_to_name[c] for c in _cats_present) or "no products"

        # ── Build map + tables, filtered to the selected HS codes ─────────────
        # In Advanced mode the category column is re-bucketed to the six
        # material-flow groups so codes outside the five trade categories
        # (Slag, Other Lead Products) survive the builders' category filter.
        display_baci = _build_display_baci(baci_df, active_years, year)
        _disp_f = display_baci[display_baci["Product"].isin(_selected_hs)].copy()
        _baci_f = baci_df[baci_df["Product"].isin(_selected_hs)].copy()
        if ADVANCED:
            _disp_f["category"] = _disp_f["Product"].map(_adv_cat_of)
            _baci_f["category"] = _baci_f["Product"].map(_adv_cat_of)
        _period_note = f"  ({period_label})" if len(active_years) > 1 else ""

        with _map_slot:
            if _is_all:
                st.caption(
                    f"Showing annualised trade volume (imports + exports) per country{_period_note}. "
                    "Select a country or region below to see its trade partners."
                )
                _fig = build_total_volume_map(_disp_f, year, _cats_present, _cat_display_label)
            elif _is_region:
                if map_color_mode == "Bilateral trade balance":
                    _fig = build_region_balance_map(
                        baci_df=_baci_f, active_years=active_years, category=_cats_present,
                        selected_region=place_sel, region_map=_region_map,
                        category_label=_cat_display_label,
                    )
                    st.caption(
                        f"**Orange**: partner is net *sender* to {place_sel}.  "
                        f"**Purple**: partner is net *receiver* from {place_sel}.  "
                        f"**White**: balanced.  <span style='color:#1565C0'>&#9632;</span> **Blue**: "
                        f"{place_sel} countries.{_period_note}",
                        unsafe_allow_html=True,
                    )
                else:
                    _fig = build_region_bilateral_map(
                        baci_df=_baci_f, active_years=active_years, category=_cats_present,
                        selected_region=place_sel, region_map=_region_map,
                        category_label=_cat_display_label,
                    )
                    st.caption(
                        f"**Green**: partner is net *sender* to {place_sel}.  "
                        f"**Red**: partner is net *receiver* from {place_sel}.  "
                        f"**White**: balanced.  <span style='color:#1565C0'>&#9632;</span> **Blue**: "
                        f"{place_sel} countries.{_period_note}",
                        unsafe_allow_html=True,
                    )
            else:
                if map_color_mode == "Bilateral trade balance":
                    _fig = build_bilateral_balance_map(_disp_f, year, _cats_present, place_sel, _cat_display_label)
                    st.caption(
                        f"**Orange**: {place_sel} is net *receiver* from partner.  "
                        f"**Purple**: {place_sel} is net *sender* to partner.  "
                        f"**White**: balanced. Color shows ratio regardless of volume.{_period_note}"
                    )
                else:
                    _fig = build_bilateral_map(_disp_f, year, _cats_present, place_sel, _cat_display_label)
                    st.caption(
                        f"**Green**: partner is net *sender* to {place_sel}.  "
                        f"**Red**: partner is net *receiver* from {place_sel}.  "
                        f"**White**: balanced. Intensity scales with bilateral volume.{_period_note}"
                    )
            st.plotly_chart(_fig, use_container_width=True)

        # ── Partner tables ────────────────────────────────────────────────────
        st.divider()
        if _is_all:
            _exp_t = (
                _disp_f.groupby("Exporter")["actual_lead"].sum()
                .sort_values(ascending=False).head(30).reset_index()
                .rename(columns={"Exporter": "Country", "actual_lead": "Exports (t Pb)"})
            )
            _exp_t.insert(0, "#", range(1, len(_exp_t) + 1))
            _imp_t = (
                _disp_f.groupby("Importer")["actual_lead"].sum()
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
        elif _is_region:
            _n_yrs = max(len(active_years), 1)
            _region_ctries = {c for c, r in _region_map.items() if r == place_sel}
            _col_exp = f"Exports to {place_sel} (t Pb)"
            _col_imp = f"Imports from {place_sel} (t Pb)"
            _yr_f = _baci_f[_baci_f["Year"].isin(active_years)]
            _sends = (
                _yr_f[_yr_f["Importer"].isin(_region_ctries)
                      & ~_yr_f["Exporter"].isin(_region_ctries)]
                .groupby("Exporter")["actual_lead"].sum() / _n_yrs
            )
            _gets = (
                _yr_f[_yr_f["Exporter"].isin(_region_ctries)
                      & ~_yr_f["Importer"].isin(_region_ctries)]
                .groupby("Importer")["actual_lead"].sum() / _n_yrs
            )
            _partners = _sends.index.union(_gets.index)
            _pbil = pd.DataFrame(index=_partners); _pbil.index.name = "Country"
            _pbil[_col_exp] = _sends.reindex(_partners, fill_value=0)
            _pbil[_col_imp] = _gets.reindex(_partners, fill_value=0)
            _pbil["_net"] = _pbil[_col_exp] - _pbil[_col_imp]
            _col_net_exp = f"Net Exports to {place_sel} (t Pb)"
            _tbl_exp = (_pbil[_pbil["_net"] > 0].sort_values("_net", ascending=False).head(30)
                        .reset_index().rename(columns={"_net": _col_net_exp}))
            _tbl_exp.insert(0, "#", range(1, len(_tbl_exp) + 1))
            _col_net_imp = f"Net Imports from {place_sel} (t Pb)"
            _pn = _pbil[_pbil["_net"] < 0].copy(); _pn[_col_net_imp] = -_pn["_net"]
            _tbl_imp = _pn.sort_values(_col_net_imp, ascending=False).head(30).reset_index()
            _tbl_imp.insert(0, "#", range(1, len(_tbl_imp) + 1))
            _rc1, _rc2 = st.columns(2)
            with _rc1:
                st.caption(f"Top 30 net exporters to {place_sel} — {_cat_display_label}")
                st.dataframe(_tbl_exp[["#", "Country", _col_net_exp, _col_exp, _col_imp]],
                             hide_index=True, use_container_width=True,
                             column_config={c: st.column_config.NumberColumn(format="%d")
                                            for c in (_col_exp, _col_imp, _col_net_exp)})
            with _rc2:
                st.caption(f"Top 30 net importers from {place_sel} — {_cat_display_label}")
                st.dataframe(_tbl_imp[["#", "Country", _col_net_imp, _col_imp, _col_exp]],
                             hide_index=True, use_container_width=True,
                             column_config={c: st.column_config.NumberColumn(format="%d")
                                            for c in (_col_imp, _col_exp, _col_net_imp)})
        else:
            _col_exp = f"Exports to {place_sel} (t Pb)"
            _col_imp = f"Imports from {place_sel} (t Pb)"
            _sends = _disp_f[_disp_f["Importer"] == place_sel].groupby("Exporter")["actual_lead"].sum()
            _gets = _disp_f[_disp_f["Exporter"] == place_sel].groupby("Importer")["actual_lead"].sum()
            _partners = _sends.index.union(_gets.index)
            _pbil = pd.DataFrame(index=_partners); _pbil.index.name = "Country"
            _pbil[_col_exp] = _sends.reindex(_partners, fill_value=0)
            _pbil[_col_imp] = _gets.reindex(_partners, fill_value=0)
            _pbil["_net"] = _pbil[_col_exp] - _pbil[_col_imp]
            _col_net_exp = f"Net Exports to {place_sel} (t Pb)"
            _tbl_exp = (_pbil[_pbil["_net"] > 0].sort_values("_net", ascending=False).head(30)
                        .reset_index().rename(columns={"_net": _col_net_exp}))
            _tbl_exp.insert(0, "#", range(1, len(_tbl_exp) + 1))
            _col_net_imp = f"Net Imports from {place_sel} (t Pb)"
            _pn = _pbil[_pbil["_net"] < 0].copy(); _pn[_col_net_imp] = -_pn["_net"]
            _tbl_imp = _pn.sort_values(_col_net_imp, ascending=False).head(30).reset_index()
            _tbl_imp.insert(0, "#", range(1, len(_tbl_imp) + 1))
            _tc1, _tc2 = st.columns(2)
            with _tc1:
                st.caption(f"Top 30 net exporters to {place_sel} — {_cat_display_label}")
                st.dataframe(_tbl_exp[["#", "Country", _col_net_exp, _col_exp, _col_imp]],
                             hide_index=True, use_container_width=True,
                             column_config={c: st.column_config.NumberColumn(format="%d")
                                            for c in (_col_exp, _col_imp, _col_net_exp)})
            with _tc2:
                st.caption(f"Top 30 net importers from {place_sel} — {_cat_display_label}")
                st.dataframe(_tbl_imp[["#", "Country", _col_net_imp, _col_imp, _col_exp]],
                             hide_index=True, use_container_width=True,
                             column_config={c: st.column_config.NumberColumn(format="%d")
                                            for c in (_col_imp, _col_exp, _col_net_imp)})


    # ── Trade Trends ──────────────────────────────────────────────────────────

    if _page == "Trade Trends":
        _tt_desc_col, _tt_lm_col = st.columns([8, 1])
        with _tt_desc_col:
            if ADVANCED:
                st.write(
                    "Plot import and export volumes for up to three countries or regions over time. "
                    "Choose from trade series (by product category) or production series "
                    "(mined and refined lead from BGS/USGS). Absolute tonnes or year-on-year % change."
                )
            else:
                st.write("Plot trade and related data over time to see trends.")
        with _tt_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data source:** BACI for trade series; BGS World Mineral Statistics "
                    "or USGS Mineral Yearbooks for production series.\n\n"
                    "**How it's used:** Plot import and export volumes for one or more countries "
                    "or UN sub-regions over time. Choose trade series (by product category) or "
                    "production series (mined and refined lead). Trade series sum BACI flows by "
                    "year; production series come from the mining/refining dataset. The 'Lead "
                    "Smelted/Refined' series combines primary and secondary refined lead. "
                    "Advanced mode adds up to three places, a year-on-year % change view, and a "
                    "shared y-axis toggle."
                )

        # ── Primary chart slot: filled after the controls below are read, so the
        # trend charts sit directly under the intro. ─────────────────────────
        _tt_chart_slot = st.container()

        _bgs_data       = _load_bgs()
        _tt_usgs_mined  = _load_usgs_mined()
        _tt_usgs_ref    = _load_usgs_refined()
        _tt_year_idx = pd.Index(sorted(baci_df["Year"].unique()), name="Year")

        # No pre-selected country: slot 1 opens on a placeholder in both modes.
        _TT_PLACEHOLDER = "— Select a country or region —"
        _tt_opts = sorted(all_baci_countries) + [_TT_SEPARATOR] + list(REGIONS_ORDERED)
        _tt_opt_slot1 = [_TT_PLACEHOLDER] + _tt_opts
        _tt_opt_with_none = ["— None —"] + _tt_opts

        if ADVANCED:
            tt_c1, tt_c2, tt_c3 = st.columns(3)
            with tt_c1:
                tt_sel1 = st.selectbox(
                    "Country / Region 1", _tt_opt_slot1, index=0, key="tt_sel1",
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
            _tt_selections = [tt_sel1, tt_sel2, tt_sel3]
        else:
            tt_sel1 = st.selectbox(
                "Country / Region", _tt_opt_slot1, index=0, key="tt_sel1",
            )
            _tt_selections = [tt_sel1]

        if ADVANCED:
            tt_rc1, tt_rc2, tt_rc3 = st.columns([3, 1, 1])
            with tt_rc1:
                tt_series = st.multiselect(
                    "Series", _TT_SERIES_ADV,
                    default=["Total Lead-Related Imports", "Total Lead-Related Exports"],
                    key="tt_series_adv",
                )
            with tt_rc2:
                tt_view = st.radio(
                    "View",
                    ["Absolute (tonnes Pb)", "Relative (% change from prior year)"],
                    index=0, key="tt_view",
                )
            with tt_rc3:
                tt_shared_y = st.checkbox("Shared y-axis", value=True, key="tt_shared_y")
        else:
            tt_series = st.multiselect(
                "Series", _TT_SERIES_EASY,
                default=["Total Lead-Related Imports", "Total Lead-Related Exports"],
                key="tt_series_easy",
            )
            tt_view = "Absolute (tonnes Pb)"
            tt_shared_y = True

        def _tt_valid(s: str) -> bool:
            return bool(s) and s not in ("— None —", _TT_PLACEHOLDER, _TT_SEPARATOR)

        _tt_slots = [
            (sel, col)
            for sel, col in zip(_tt_selections, _TT_SLOT_COLORS)
            if _tt_valid(sel)
        ]

        with _tt_chart_slot:
            if not tt_series:
                st.info("Select at least one series below.")
            elif not _tt_slots:
                st.info("Select a country or region below to plot its trends.")
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

    if _page == "Trade Relationships":
        _fn_desc_col, _fn_lm_col = st.columns([8, 1])
        with _fn_desc_col:
            if ADVANCED:
                st.write(
                    "A schematic node-link diagram of bilateral lead product flows between selected countries. "
                    "Bubble size reflects total BACI trade volume (imports + exports across all categories). "
                    "Arrows are colored by product category and scaled by flow volume. "
                    "Drag nodes freely in the interactive view."
                )
            else:
                st.write(
                    "A schematic node-link diagram of bilateral lead product flows between selected countries."
                )
        with _fn_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data source:** BACI bilateral trade flows.\n\n"
                    "**How it's used:** Bubble size reflects total BACI trade volume; arrows are "
                    "colored by product category and scaled by flow volume (√-scaled so small "
                    "flows stay visible). Flows below the minimum threshold are hidden to reduce "
                    "clutter. Choose a focal country to show only the flows that involve it — "
                    "countries with no connection to it are dropped automatically. Use this view "
                    "to spot dominant corridors, e.g. the main scrap suppliers to a processing hub."
                )

        # ── Primary graph slot: filled after the controls below are read, so the
        # network diagram sits directly under the intro. ────────────────────
        _fn_graph_slot = st.container()

        # ── View mode: interactive (draggable) is the default in both modes;
        # Advanced can switch back to the static Plotly view.
        if ADVANCED:
            fn_view_mode = st.radio(
                "View mode",
                options=["Interactive (draggable)", "Static (Plotly)"],
                index=0,
                horizontal=True,
                key="fn_view_mode",
                help=(
                    "**Interactive**: pyvis / vis.js graph where you can grab and drag "
                    "individual nodes. Zoom with the scroll wheel.\n\n"
                    "**Static**: Plotly figure with a chosen layout preset (circle, grid, "
                    "force-directed, or geographic)."
                ),
            )
        else:
            fn_view_mode = "Interactive (draggable)"

        # ── Countries (+ minimum flow in Advanced) ────────────────────────────
        _fn_defaults = [c for c in ["India", "China", "USA"] if c in all_baci_countries]
        if ADVANCED:
            fn_col1, fn_col2 = st.columns([3, 1])
            with fn_col1:
                fn_countries = st.multiselect(
                    "Select countries",
                    options=all_baci_countries,
                    default=_fn_defaults,
                    key="fn_countries",
                    help="Select 2-15 countries. Bubbles sized by total BACI trade volume; arrows show bilateral flows.",
                )
            with fn_col2:
                fn_min_flow = st.slider(
                    "Minimum flow to display (t Pb)",
                    min_value=100, max_value=10000, value=500, step=100,
                    key="fn_min_flow",
                )
        else:
            fn_countries = st.multiselect(
                "Select countries",
                options=all_baci_countries,
                default=_fn_defaults,
                key="fn_countries",
                help="Select 2-15 countries. Bubbles sized by total BACI trade volume; arrows show bilateral flows.",
            )
            fn_min_flow = 100

        # ── Layout + focal country (+ region cluster in Advanced) ─────────────
        _LAYOUT_LABELS = {
            "Circle": "circle",
            "Grid": "grid",
            "Force-directed (by trade weight)": "force",
            "Geographic centroids": "geographic",
        }

        def _fn_focal_selectbox():
            # Focal country: optional, no default center. Choosing one shows only
            # the flows that involve it and drops unconnected countries.
            if fn_countries:
                _raw = st.selectbox(
                    "Focal country (optional)",
                    options=["— None —"] + fn_countries,
                    index=0,
                    key="fn_focal",
                    help=(
                        "Optional. Choose a focal country to show only the flows that involve "
                        "it; countries with no connection to it are removed automatically."
                    ),
                )
                return None if _raw.startswith("—") else _raw
            return None

        if ADVANCED:
            fn_lc0, fn_lc1, fn_lc2 = st.columns([2, 2, 2])
            with fn_lc0:
                fn_layout_label = st.selectbox(
                    "Initial layout",
                    options=list(_LAYOUT_LABELS.keys()),
                    index=list(_LAYOUT_LABELS.keys()).index("Geographic centroids"),
                    key="fn_layout",
                    help=(
                        "**Circle**: even spacing around a ring (with focal at center).  "
                        "**Grid**: rows and columns.  "
                        "**Force-directed**: pairs with heavier trade pull closer together.  "
                        "**Geographic centroids**: approximate world position on a schematic canvas."
                    ),
                )
            with fn_lc1:
                fn_focal_country = _fn_focal_selectbox()
            with fn_lc2:
                fn_use_cluster = st.toggle(
                    "Treat a region as single node",
                    value=False,
                    key="fn_use_cluster",
                    help="Aggregate all flows to/from countries in the selected region into a single node.",
                )
                if fn_use_cluster:
                    _fn_cluster_options = (
                        MAJOR_REGIONS_ORDERED + [_RTV_REGION_SEP] + REGIONS_ORDERED
                    )
                    fn_cluster_region = st.selectbox(
                        "Region to aggregate",
                        options=_fn_cluster_options,
                        key="fn_cluster_region",
                        help="Choose a UN region (e.g. 'Africa') or a UN sub-region (e.g. 'Western Africa').",
                    )
                else:
                    fn_cluster_region = None
        else:
            _fn_easy_layouts = {
                "Circle": "circle",
                "Geographic centroids": "geographic",
            }
            fn_lc0, fn_lc1 = st.columns(2)
            with fn_lc0:
                fn_layout_label = st.selectbox(
                    "Layout",
                    options=list(_fn_easy_layouts.keys()),
                    index=list(_fn_easy_layouts.keys()).index("Geographic centroids"),
                    key="fn_layout_easy",
                )
            with fn_lc1:
                fn_focal_country = _fn_focal_selectbox()
            fn_use_cluster = False
            fn_cluster_region = None
        fn_layout_code = _LAYOUT_LABELS[fn_layout_label]

        # ── Product selection ─────────────────────────────────────────────────
        # Easy: the five trade categories (arrows coloured by 5-cat palette).
        # Advanced: per-HS tick boxes grouped by the six slide-out categories
        # (arrows coloured by 6-cat palette; Slag + Other Lead Products off).
        _fn_present = sorted(int(p) for p in baci_df["Product"].dropna().unique())
        if ADVANCED:
            fn_selected_hs = _advanced_hs_picker(_fn_present, "fn")
            if not fn_selected_hs:
                fn_selected_hs = [p for p in _fn_present
                                  if _adv_cat_of(p) not in ADV_CATS_DEFAULT_OFF]
            fn_categories = [c for c in CATEGORIES_ORDERED
                             if any(_adv_cat_of(p) == c for p in fn_selected_hs)]
            fn_grouping = "advanced"
            fn_products: tuple[int, ...] | None = tuple(sorted(fn_selected_hs))
        else:
            st.caption("Product categories to include")
            fn_cat_cols = st.columns(len(_CAT_LABEL_LIST))
            fn_categories = []
            for _i, _lbl in enumerate(_CAT_LABEL_LIST):
                with fn_cat_cols[_i]:
                    if st.checkbox(_lbl, value=True, key=f"fn_cat_{_CAT_LABELS[_lbl]}"):
                        fn_categories.append(_CAT_LABELS[_lbl])
            fn_grouping = "trade"
            fn_products = None

        # ── Flow direction (Advanced only; Easy always shows both) ────────────
        if ADVANCED:
            fn_dir = st.radio(
                "Flow direction",
                options=["Both", "Import flows", "Export flows"],
                index=0,
                horizontal=True,
                key="fn_dir",
                disabled=fn_focal_country is None,
                help=(
                    "Filters arrows relative to the focal country: **Import flows** shows "
                    "arrows into the focal country, **Export flows** shows arrows out of it. "
                    "Requires a focal country."
                ),
            )
            if fn_focal_country is None:
                fn_dir = "Both"
        else:
            fn_dir = "Both"
        fn_show_imports = fn_dir in ["Both", "Import flows"]
        fn_show_exports = fn_dir in ["Both", "Export flows"]

        # ── Year scope (Advanced only) ────────────────────────────────────────
        if ADVANCED:
            fn_all_years = st.checkbox(
                "Use all years in BACI dataset",
                value=False,
                key="fn_all_years",
                help=(
                    "Ignore the sidebar year filter and use every year in the "
                    f"BACI dataset ({min_year}-{max_year}). Bubble sizes and arrow "
                    "widths reflect the full-period annualised trade volume."
                ),
            )
        else:
            fn_all_years = False

        # ── Focal-country behaviours are bundled into the focal choice ────────
        # Choosing a focal country automatically limits the view to flows that
        # involve it and drops countries with no connection to it (both modes).
        fn_focal_only = fn_focal_country is not None
        fn_prune_isolated = fn_focal_country is not None

        # ── Pair hider (Advanced only) ────────────────────────────────────────
        if ADVANCED:
            _FN_PAIR_HIDER_CAP = 20
            if len(fn_countries) > _FN_PAIR_HIDER_CAP:
                st.caption(
                    f"Pair-hider available with <= {_FN_PAIR_HIDER_CAP} countries selected "
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
                _fn_pair_labels = [f"{a} <-> {b}" for a, b in _fn_all_pairs]
                _fn_pair_lookup = {f"{a} <-> {b}": frozenset([a, b]) for a, b in _fn_all_pairs}
                fn_hidden_pair_labels = st.multiselect(
                    "Hide specific country pairs",
                    options=_fn_pair_labels,
                    default=[],
                    key="fn_hidden_pairs",
                    help=(
                        "Pick country pairs to hide. Both directions are removed. "
                        "Example: select 'India <-> USA' to drop USA->India and India->USA "
                        "arrows while keeping every other flow."
                    ),
                )
                fn_hidden_pairs = {_fn_pair_lookup[l] for l in fn_hidden_pair_labels}
        else:
            fn_hidden_pairs = set()

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

        with _fn_graph_slot:
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
                        grouping=fn_grouping,
                        products=fn_products,
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
                        grouping=fn_grouping,
                        products=fn_products,
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
                    grouping=fn_grouping,
                    products=fn_products,
                )
                st.plotly_chart(fig_fn, use_container_width=True)
                st.caption(
                    f"Layout: **{fn_layout_label}**. "
                    f"Bubble size ∝ √(total BACI trade volume). "
                    f"Arrow width ∝ √(flow volume). "
                    f"Showing flows ≥ {fn_min_flow:,} t Pb between selected countries. "
                    f"Period: {_fn_period_label}."
                )


    _easy_assumptions_footer()


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




if _page in ("Production & Capacity", "Lead Accumulation", "Recycling Economy Snapshot (Beta 🧪)", "Material Flow (Beta 🧪)"):

    # ── Production & Capacity ─────────────────────────────────────────────────

    if _page == "Production & Capacity":
        import math as _math

        _mbp_desc_col, _mbp_lm_col = st.columns([8, 1])
        with _mbp_desc_col:
            if ADVANCED:
                st.write(
                    "Where is lead being mined? Where is it being refined? Where are batteries "
                    "manufactured, and where do they enter service? Select a question below to "
                    "explore the global picture. All scales are logarithmic — enabling comparison "
                    "across small and large producers. Countries in grey have no data for the "
                    "selected view. Use the animated toggle for time-lapse playback."
                )
            else:
                st.write("Explore where lead is mined, refined, manufactured, and used.")
        with _mbp_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Lead Mining / Lead Refining** — drawn directly from BGS or USGS "
                    "national statistics (select source in sidebar). These are reported "
                    "figures, not estimates.\n\n"
                    "**Battery Manufacturing (BOTEC) / Battery Consumption (BOTEC)** — "
                    "back-of-the-envelope estimates from the material-flow model using trade "
                    "data and process parameters. Advanced mode exposes parameter sliders to "
                    "test sensitivity.\n\n"
                    "All scales are logarithmic. For Lead Mining and Lead Refining, a ranked "
                    "country table is shown below the map."
                )

        # ── Primary map slot: filled after the controls below are read, so the
        # map sits directly under the intro. ─────────────────────────────────
        _prod_map_slot = st.container()

        _mining_refining_df3 = _load_mining_refining()

        _ESTIMATED_DATASETS = {
            "Battery Manufacturing (BOTEC)",
            "Battery Consumption (BOTEC)",
        }
        # Custom yellow (amber) sequential scale for Battery Consumption — Plotly
        # has no yellow-dominant named sequential that stays legible on white.
        _YELLOW_SCALE = [[0.0, "#FFF9C4"], [0.55, "#FDD835"], [1.0, "#F57F17"]]
        _COLORSCALES = {
            "Lead Mining":                   "Purples",
            "Lead Refining":                 "Blues",
            "Battery Manufacturing (BOTEC)": "Greens",
            "Battery Consumption (BOTEC)":   _YELLOW_SCALE,
        }
        _METRIC_KEYS = {
            "Battery Manufacturing (BOTEC)": "F4_batt_lead",
            "Battery Consumption (BOTEC)":   "F5_implied",
        }

        # ── Dataset selector (+ animation toggle in Advanced) ─────────────────────
        if ADVANCED:
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
        else:
            prod_dataset = st.radio(
                "Dataset",
                options=list(_COLORSCALES.keys()),
                index=0,
                horizontal=True,
                key="prod_dataset",
            )
            prod_animate = False

        _colorscale = _COLORSCALES[prod_dataset]

        # Mining/refining source for THIS tab only. Advanced follows the global
        # sidebar; Easy exposes a page-local BGS/USGS toggle rendered below the
        # map (read here via the deferred-read pattern, applied on next rerun).
        if ADVANCED:
            _prod_mining_pref = _mining_pref
        else:
            _prod_mining_pref = st.session_state.get("prod_mining_pref_easy", "BGS")

        # ── Process parameters (estimated metrics only) ───────────────────────────
        _map_eta_secondary = 0.97
        _map_eta_break     = 0.95
        _map_delta_pb      = 0.95
        _map_beta          = 0.85
        _map_eta_mfg       = 0.98
        _map_eta_ore       = 0.95
        _map_gamma         = 0.95

        if prod_dataset in _ESTIMATED_DATASETS:
            if ADVANCED:
                st.info(
                    f"**{prod_dataset}** is a model estimate derived from the material-flow "
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
                            0.30, 1.00, 0.95, 0.01, format="%.2f", key="map_gamma",
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
            else:
                st.info(
                    f"**{prod_dataset}** is a back-of-the-envelope model estimate from the "
                    "material-flow equations using trade data and mining/refining anchors."
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
                f"**{prod_dataset}** — model estimate (material flow). "
                "Values are in metric tonnes of lead content. "
                "Scale is logarithmic. Countries with no refining anchor shown in grey."
            )
        else:
            # Use country_year_mining_refining.csv
            if prod_dataset == "Lead Mining":
                _mr_col = "mined_bgs_t" if _prod_mining_pref == "BGS" else "mined_usgs_t"
            else:  # Lead Refining
                if _prod_mining_pref == "BGS":
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
            _src_name = "BGS" if _prod_mining_pref == "BGS" else "USGS"
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
                        _dataset_key, _pb_factors_tuple, _prod_mining_pref,
                        _map_eta_secondary, _map_eta_break, _map_delta_pb,
                        _map_beta, _map_eta_mfg, _map_eta_ore, _map_gamma,
                    )
                    _mk = _METRIC_KEYS[prod_dataset]
                    _ll, _vv = [], []
                    for _c, _o in _od.items():
                        _v = max(0.0, _o.get(_mk, 0.0))
                        if _v > 0:
                            _ll.append(_c)
                            # Model-derived (BOTEC) → confine to 2 significant figures.
                            _vv.append(_sig2(_v))
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
                    if ADVANCED:
                        prod_year = st.slider(
                            "Year",
                            min_value=_prod_years[0], max_value=_prod_years[-1],
                            value=_default_yr, step=1, key="prod_static_year",
                        )
                    else:
                        prod_year = _default_yr
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
                        with _prod_map_slot:
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
                    with _prod_map_slot:
                        st.plotly_chart(fig_anim, use_container_width=True)

            with _prod_map_slot:
                st.caption(_caption)

        # Easy-mode-only page-local data-source toggle, below the map. Applies
        # to this tab only; the global sidebar source is unchanged.
        if not ADVANCED:
            st.radio(
                "Mining & refining data source (this map only)",
                options=["BGS", "USGS"],
                key="prod_mining_pref_easy",
                horizontal=True,
                help=(
                    "Switch the production data source for this map only. "
                    "BGS covers 1971-2023; USGS separates primary and secondary "
                    "refined lead (2015-2023). Other tabs are unaffected."
                ),
            )

    # ── Lead Accumulation ─────────────────────────────────────────────────────

    if _page == "Lead Accumulation":
        _la_hdr_col, _la_lm_col = st.columns([8, 1])
        with _la_hdr_col:
            st.subheader("Lead Mass Accumulation")
            if ADVANCED:
                st.caption(
                    "Annual net lead balance by country or sub-region: "
                    "mining production + imports − exports, broken down by product category. "
                    "All values in kt Pb (kilotonnes of lead content)."
                )
            else:
                st.caption(
                    "Shows how a country's internal lead stock has changed over the years "
                    "due to production and trade."
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
            major_region_map=MAJOR_REGION_MAP,
            major_regions_ordered=MAJOR_REGIONS_ORDERED,
            advanced=ADVANCED,
        )


    # ── Process Estimates ─────────────────────────────────────────────────────

    if _page == "Material Flow (Beta 🧪)":
        st.warning(
            "This is only based on BOTEC-type calculations, and has not been reconciled "
            "against external data sources. These numbers also do not account for "
            "informal vs. formal dynamics, which can greatly affect the efficiency values "
            "at every step.",
            icon="⚠️",
        )
        _pe_desc_col, _pe_lm_col = st.columns([8, 1])
        with _pe_desc_col:
            if ADVANCED:
                st.write(
                    "Material-flow Sankey for a selected country. Shows how lead flows "
                    "through collection, breaking, secondary smelting, manufacturing, and "
                    "installation."
                )
            else:
                st.write(
                    "Shows how lead flows through collection, breaking, secondary smelting, "
                    "manufacturing, and installation."
                )
        with _pe_lm_col:
            with st.popover("ℹ Learn more"):
                st.markdown(
                    "**Data sources:** BACI trade flows + BGS/USGS production + "
                    "Eurostat collection data (where available).\n\n"
                    "**Experimental model.** Values are estimates derived from trade and "
                    "production data (BOTEC-type calculations) and are not reconciled against "
                    "external observations. Treat them as indicative, not precise.\n\n"
                    + _EASY_MODEL_EQUATIONS_MD +
                    "\n\n**Which production source is used:** The model is anchored on each "
                    "country's measured refined-lead production. It defaults to BGS, but when "
                    "BGS and USGS disagree by more than 25% it switches to USGS — which measures "
                    "recycled (secondary) lead directly — and notes the switch under the chart."
                )
        _mining_refining_df = _load_mining_refining()
        _eurostat_df = _load_eurostat_collection()
        # Easy mode auto-selects the more plausible refined-lead source per
        # country (BGS default, USGS when the two diverge); Advanced respects
        # the explicit sidebar choice.
        _recycle_src = "AUTO" if not ADVANCED else _mining_pref
        render_mass_balance_sankey_tab(
            baci_df         = baci_df,
            mining_df       = _mining_refining_df,
            region_map      = REGION_MAP,
            regions_ordered = REGIONS_ORDERED,
            sidebar_year    = year,
            active_years    = active_years,
            dataset         = _dataset_key,
            pb_factors      = pb_factors,
            mining_source   = _recycle_src,
            eurostat_df     = _eurostat_df,
            advanced        = ADVANCED,
        )


    # ── Economy Snapshot ──────────────────────────────────────────────────────

    if _page == "Recycling Economy Snapshot (Beta 🧪)":
        from visualizations.mass_balance_sankey import render_economy_snapshot_tab
        _mining_refining_df2 = _load_mining_refining()
        _recycle_src2 = "AUTO" if not ADVANCED else _mining_pref
        render_economy_snapshot_tab(
            baci_df         = baci_df,
            mining_df       = _mining_refining_df2,
            region_map      = REGION_MAP,
            regions_ordered = REGIONS_ORDERED,
            active_years    = active_years,
            dataset         = _dataset_key,
            pb_factors      = pb_factors,
            mining_source   = _recycle_src2,
            advanced        = ADVANCED,
        )

    _easy_assumptions_footer()


# ── Literature Stats page ───────────────────────────────────────
if _page == "Literature Stats":
    render_literature()
