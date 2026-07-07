"""
mass_balance_sankey.py — Multi-step Sankey diagram for the Pb Action lead mass balance.

Entry point:
    render_mass_balance_sankey_tab(
        baci_df, mining_df, region_map, regions_ordered,
        sidebar_year, active_years, dataset, pb_factors, mining_source
    )

The Sankey traces lead through a country's battery recycling economy:
    Mine / Ore Imports → Primary Refining ↘
                                             Feedstock Pool → Battery Mfg → Battery Service
    ULAB Collection → Breaking → Scrap → Secondary Refining ↗
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model.data_loader import EUROSTAT_ELIGIBLE, REFINING_HUBS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"
_MASTER_BACI_PATH = _DATA_DIR / "BACI_lead_trade_2012_2024_modified_vHS_4_master.csv"

# ---------------------------------------------------------------------------
# Metal-product HS codes (HS12 only)
# ---------------------------------------------------------------------------
_METAL_HS = {780411, 780419, 780420, 780600}

# Default Pb factor for metal products (lead wrought/tubes/other)
_METAL_DEFAULT_FACTOR = 0.97

# Used battery HS codes (HS12 + HS22 equivalent)
_USED_HS = {854810, 854911}

# Feed HS codes for imp_feed / exp_feed
_FEED_HS = {780110, 780191}


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
# Node fill colors (hex)
_C = {
    "mine":       "#78909C",   # blue-grey    — mine output
    "ore_pool":   "#90A4AE",   # lighter blue-grey — ore pool
    "ore_import": "#B0BEC5",   # pale blue-grey — ore import
    "ulab_import":"#FFCC80",   # pale orange  — ULAB net import
    "collection": "#66BB6A",   # green        — battery collection
    "breaking":   "#FF9800",   # deeper orange— breaking
    "scrap_pool": "#EF5350",   # red          — scrap pool
    "scrap_import":"#FFCDD2",  # pale red     — scrap import
    "feed_import": "#81D4FA",  # pale blue    — feed import
    "primary":    "#5C6BC0",   # indigo       — primary refining
    "secondary":  "#7986CB",   # lighter indigo— secondary refining
    "refining":   "#5C6BC0",   # indigo       — BGS unified refining
    "feedstock":  "#29B6F6",   # light blue   — feedstock pool
    "nonbatt":    "#78909C",   # blue-grey    — non-battery lead
    "battery_mfg":"#66BB6A",   # green        — battery manufacturing
    "batt_service":"#388E3C",  # dark green   — battery service entry
    "batt_import": "#A5D6A7",  # pale green   — battery net import
    "metal_import":"#CFD8DC",  # light grey   — metal products import
    "lead_products":"#607D8B", # slate        — lead products terminal
    "disposal":   "#B71C1C",   # dark red     — permanent disposal
    "export_sink":"#CFD8DC",   # light grey   — all export sinks
}


def _rgba(hex_color: str, alpha: float = 0.35) -> str:
    """Convert #rrggbb to rgba(r,g,b,alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_master_baci() -> pd.DataFrame:
    """Load the master BACI file which includes metal-product HS codes."""
    if not _MASTER_BACI_PATH.exists():
        return pd.DataFrame(columns=["Year", "Exporter", "Importer", "Product", "Quantity"])
    df = pd.read_csv(_MASTER_BACI_PATH, encoding="utf-8-sig",
                     usecols=["Year", "Exporter", "Importer", "Product", "Quantity"])
    df["Product"] = pd.to_numeric(df["Product"], errors="coerce").astype("Int64")
    df = df[df["Product"].isin(_METAL_HS)].copy()
    return df


# ---------------------------------------------------------------------------
# Trade aggregation helpers
# ---------------------------------------------------------------------------

def _sum_trade(
    df: pd.DataFrame,
    country: str,
    years: list[int],
    hs_codes: set[int],
    pb_factors: dict[int, float],
    direction: str,  # "import" or "export"
) -> float:
    """
    Return total Pb content (t) averaged over years for given HS codes
    using Quantity × pb_factors[hs].
    """
    col = "Importer" if direction == "import" else "Exporter"
    mask = (
        df[col].eq(country) &
        df["Year"].isin(years) &
        df["Product"].isin(hs_codes)
    )
    sub = df[mask]
    if sub.empty:
        return 0.0
    total = (
        sub.apply(lambda r: r["Quantity"] * pb_factors.get(int(r["Product"]), 0.0), axis=1)
        .sum()
    )
    return total / max(1, len(years))


def _sum_trade_vectorised(
    df: pd.DataFrame,
    country: str,
    years: list[int],
    hs_codes: set[int],
    pb_factors: dict[int, float],
    direction: str,
) -> float:
    """Vectorised version of _sum_trade for the main baci_df."""
    col = "Importer" if direction == "import" else "Exporter"
    mask = df[col].eq(country) & df["Year"].isin(years) & df["Product"].isin(hs_codes)
    sub = df[mask].copy()
    if sub.empty:
        return 0.0
    factor_series = sub["Product"].map(pb_factors).fillna(0.0)
    total = (sub["Quantity"] * factor_series).sum()
    return float(total) / max(1, len(years))


# ---------------------------------------------------------------------------
# Mining/refining extraction
# ---------------------------------------------------------------------------

def _get_mining_row(
    mining_df: pd.DataFrame,
    country: str,
    years: list[int],
    mining_source: str,
) -> dict[str, float]:
    """
    Return averaged mining/refining values for country+years.
    All values default to 0.0 if missing.
    """
    sub = mining_df[
        mining_df["country"].eq(country) & mining_df["year"].isin(years)
    ]

    def _mean_col(col: str) -> float:
        if col not in sub.columns or sub.empty:
            return 0.0
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else 0.0

    return {
        "mined_bgs":              _mean_col("mined_bgs_t"),
        "mined_usgs":             _mean_col("mined_usgs_t"),
        "refined_bgs":            _mean_col("refined_bgs_t"),
        "refined_primary_usgs":   _mean_col("refined_primary_usgs_t"),
        "refined_secondary_usgs": _mean_col("refined_secondary_usgs_t"),
    }


# ---------------------------------------------------------------------------
# Sankey builder
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    """Format a tonnage value with commas."""
    return f"{v:,.0f}"


def build_sankey(
    baci_df: pd.DataFrame,
    master_df: pd.DataFrame,
    mining_df: pd.DataFrame,
    country: str,
    active_years: list[int],
    dataset: str,
    pb_factors: dict[int, float],
    mining_source: str,
    # process parameters
    eta_secondary: float = 0.97,
    eta_break: float = 0.95,
    delta_pb: float = 0.95,
    beta: float = 0.85,
    eta_mfg: float = 0.98,
    eta_ore: float = 0.95,
    gamma: float = 0.70,
    min_flow: float = 10.0,
    # Eurostat anchor parameters
    anchor_mode: str = "mining",
    eurostat_input_t: float = 0.0,
    battery_lead_content_fraction: float = 0.65,
) -> tuple[go.Figure | None, dict, str]:
    """
    Build the mass-balance Sankey for the given country and parameters.

    Returns (fig, model_outputs_dict, warning_message).
    fig is None if rendering is impossible.
    warning_message is "" when no warning.
    """
    n = max(1, len(active_years))
    if len(active_years) > 1:
        period_label = f"{min(active_years)}–{max(active_years)} avg"
    else:
        period_label = str(active_years[0])

    warning_msg = ""

    # ------------------------------------------------------------------
    # 1. BACI trade flows (Quantity × pb_factors)
    # ------------------------------------------------------------------
    def _imp(hs_set: set[int]) -> float:
        return _sum_trade_vectorised(baci_df, country, active_years, hs_set, pb_factors, "import")

    def _exp(hs_set: set[int]) -> float:
        return _sum_trade_vectorised(baci_df, country, active_years, hs_set, pb_factors, "export")

    imp_ore   = _imp({260700})
    exp_ore   = _exp({260700})
    imp_scrap = _imp({780200})
    exp_scrap = _exp({780200})
    imp_used  = _imp(_USED_HS)
    exp_used  = _exp(_USED_HS)
    imp_feed  = _imp(_FEED_HS)
    exp_feed  = _exp(_FEED_HS)
    imp_bull  = _imp({780199})
    exp_bull  = _exp({780199})
    imp_batt  = _imp({850710, 850720, 850790})
    exp_batt  = _exp({850710, 850720, 850790})

    # Metal products (HS12 only, from master BACI)
    imp_metal = 0.0
    exp_metal = 0.0
    if dataset == "hs12" and not master_df.empty:
        def _metal(direction: str) -> float:
            col = "Importer" if direction == "import" else "Exporter"
            mask = (
                master_df[col].eq(country) &
                master_df["Year"].isin(active_years) &
                master_df["Product"].isin(_METAL_HS)
            )
            sub = master_df[mask].copy()
            if sub.empty:
                return 0.0
            factor_series = sub["Product"].apply(
                lambda hs: pb_factors.get(int(hs), _METAL_DEFAULT_FACTOR)
            )
            return float((sub["Quantity"] * factor_series).sum()) / n

        imp_metal = _metal("import")
        exp_metal = _metal("export")

    # ------------------------------------------------------------------
    # 2. Mining / refining anchor
    # ------------------------------------------------------------------
    m = _get_mining_row(mining_df, country, active_years, mining_source)

    mined_bgs            = m["mined_bgs"]
    mined_usgs           = m["mined_usgs"]
    refined_bgs          = m["refined_bgs"]
    ref_primary_usgs     = m["refined_primary_usgs"]
    ref_secondary_usgs   = m["refined_secondary_usgs"]

    has_bgs  = refined_bgs > 0
    has_usgs = (ref_primary_usgs > 0 or ref_secondary_usgs > 0)

    # Recommend switching source if active source has no data but the other does.
    # In Eurostat anchor mode, absence of mining/refining data is not fatal
    # (eff_primary will be 0, which is acceptable for secondary-smelting-only countries).
    _eurostat_active = anchor_mode == "eurostat" and eurostat_input_t > 0
    if not _eurostat_active:
        if mining_source == "BGS" and not has_bgs and has_usgs:
            warning_msg = (
                f"No BGS refining data found for **{country}** in the selected years. "
                "USGS data is available — switch to USGS in the sidebar."
            )
        elif mining_source == "USGS" and not has_usgs and has_bgs:
            warning_msg = (
                f"No USGS refining data found for **{country}** in the selected years. "
                "BGS data is available — switch to BGS in the sidebar."
            )
        elif not has_bgs and not has_usgs:
            warning_msg = "__no_refining__"

    # ------------------------------------------------------------------
    # 3. Primary smelting (always from mining data, both anchor modes)
    # ------------------------------------------------------------------
    if mining_source == "BGS":
        mined = mined_bgs
        ore_dom = mined + imp_ore - exp_ore
        est_primary = max(0.0, ore_dom) * eta_ore
        pct_primary = est_primary / refined_bgs if refined_bgs > 0 else 0.0
        if pct_primary < 0.05:
            eff_primary = 0.0
            mining_eff_secondary = refined_bgs
        else:
            eff_primary = est_primary
            mining_eff_secondary = max(0.0, refined_bgs - est_primary)
    else:  # USGS
        mined = mined_usgs
        bullion_net = exp_bull - imp_bull
        eff_primary = ref_primary_usgs
        mining_eff_secondary = max(0.0, ref_secondary_usgs + bullion_net)

    # ------------------------------------------------------------------
    # 4. Backward chain — Eurostat anchor or mining anchor
    # ------------------------------------------------------------------
    _use_eurostat = anchor_mode == "eurostat" and eurostat_input_t > 0

    if _use_eurostat:
        # Anchor: Eurostat COLLECT (t LAB/yr) → lead-content USED_DOM
        B3_used_dom   = eurostat_input_t * battery_lead_content_fraction * delta_pb
        B2_break_out  = B3_used_dom * eta_break
        B1_scrap_dom  = B2_break_out - imp_scrap + exp_scrap
        eff_secondary = max(0.0, B1_scrap_dom) * eta_secondary
    else:
        # Anchor: secondary smelting output from BGS/USGS
        eff_secondary = mining_eff_secondary
        B1_scrap_dom  = eff_secondary / eta_secondary if eff_secondary > 0 else 0.0
        B2_break_out  = max(0.0, B1_scrap_dom - imp_scrap + exp_scrap)
        B3_used_dom   = B2_break_out / (delta_pb * eta_break) if B2_break_out > 0 else 0.0

    B4_collected  = max(0.0, B3_used_dom - imp_used + exp_used)
    B5_total_ulab = B4_collected / gamma if gamma > 0 else 0.0
    B6_disposed   = B5_total_ulab * (1.0 - gamma)

    eff_total = eff_primary + eff_secondary

    # ------------------------------------------------------------------
    # 5. Forward chain (from total smelting)
    # ------------------------------------------------------------------
    # 780199 (bullion) intentionally excluded from feedstock pool (cancels algebraically)
    F1_feedstock = eff_total + imp_feed - exp_feed
    F2_nonbatt   = F1_feedstock * (1.0 - beta)
    F3_feed_batt = F1_feedstock * beta
    F4_batt_lead = max(0.0, F3_feed_batt) * eta_mfg
    F5_implied   = F4_batt_lead + imp_batt - exp_batt

    # ------------------------------------------------------------------
    # Collect model outputs for summary table
    # ------------------------------------------------------------------
    model_outputs = {
        "eff_primary":   eff_primary,
        "eff_secondary": eff_secondary,
        "B1_scrap_dom":  B1_scrap_dom,
        "B2_break_out":  B2_break_out,
        "B3_used_dom":   B3_used_dom,
        "B4_collected":  B4_collected,
        "B5_total_ulab": B5_total_ulab,
        "B6_disposed":   B6_disposed,
        "F1_feedstock":  F1_feedstock,
        "F2_nonbatt":    F2_nonbatt,
        "F3_feed_batt":  F3_feed_batt,
        "F4_batt_lead":  F4_batt_lead,
        "F5_implied":    F5_implied,
        # raw trade
        "imp_ore":   imp_ore,   "exp_ore":   exp_ore,
        "imp_scrap": imp_scrap, "exp_scrap": exp_scrap,
        "imp_used":  imp_used,  "exp_used":  exp_used,
        "imp_feed":  imp_feed,  "exp_feed":  exp_feed,
        "imp_bull":  imp_bull,  "exp_bull":  exp_bull,
        "imp_batt":  imp_batt,  "exp_batt":  exp_batt,
        "imp_metal": imp_metal, "exp_metal": exp_metal,
        "mined":     mined,
    }

    if warning_msg == "__no_refining__":
        return None, model_outputs, warning_msg

    # ------------------------------------------------------------------
    # 6. Node/link construction
    # ------------------------------------------------------------------
    node_labels  : list[str]   = []
    node_colors  : list[str]   = []
    node_x       : list[float] = []
    node_y       : list[float] = []
    node_hover   : list[str]   = []

    links_src    : list[int]   = []
    links_tgt    : list[int]   = []
    links_val    : list[float] = []
    links_color  : list[str]   = []
    links_hover  : list[str]   = []

    def _add_node(label: str, color: str, x: float, y: float, value: float) -> int:
        idx = len(node_labels)
        node_labels.append(label)
        node_colors.append(color)
        node_x.append(x)
        node_y.append(y)
        node_hover.append(f"{label}: {_fmt(value)} t Pb")
        return idx

    def _add_link(src: int, tgt: int, val: float, src_color: str, label: str) -> None:
        if val < min_flow:
            return
        links_src.append(src)
        links_tgt.append(tgt)
        links_val.append(round(val, 1))
        links_color.append(_rgba(src_color))
        links_hover.append(f"{label}: {_fmt(val)} t Pb")

    # ------------------------------------------------------------------
    # Fixed-position nodes (mid-chain)
    # ------------------------------------------------------------------

    # Ore Available (x=0.18) — only if there is primary smelting or any ore flow
    ore_flow_total = mined + max(0.0, imp_ore - exp_ore) + max(0.0, exp_ore - imp_ore)
    show_ore_pool  = (eff_primary > 0) or (ore_flow_total > min_flow)

    if show_ore_pool:
        ore_dom_val = max(0.0, mined + imp_ore - exp_ore)
        ore_pool_idx = _add_node("Ore Available", _C["ore_pool"], 0.18, 0.15, ore_dom_val)
    else:
        ore_pool_idx = None

    # Battery Collection (x=0.28)
    show_collection = B4_collected > min_flow or imp_used > min_flow
    if show_collection:
        collection_idx = _add_node("ULAB Collection", _C["collection"], 0.28, 0.72, B4_collected)
    else:
        collection_idx = None

    # Battery Breaking (x=0.38)
    show_breaking = B2_break_out > min_flow
    if show_breaking:
        breaking_idx = _add_node("Battery Breaking", _C["breaking"], 0.38, 0.72, B2_break_out)
    else:
        breaking_idx = None

    # Scrap Pool (x=0.46)
    show_scrap_pool = eff_secondary > min_flow
    if show_scrap_pool:
        scrap_pool_idx = _add_node("Scrap Pool", _C["scrap_pool"], 0.46, 0.72, B1_scrap_dom)
    else:
        scrap_pool_idx = None

    # Refining node(s) (x=0.50)
    use_split_refining = (
        mining_source == "USGS" and eff_primary > min_flow and eff_secondary > min_flow
    )
    use_secondary_only = (
        mining_source == "USGS" and eff_primary <= min_flow and eff_secondary > min_flow
    )
    use_primary_only   = (
        mining_source == "USGS" and eff_secondary <= min_flow and eff_primary > min_flow
    )

    primary_ref_idx   = None
    secondary_ref_idx = None
    bgs_ref_idx       = None

    if mining_source == "USGS":
        if use_split_refining:
            primary_ref_idx   = _add_node("Primary Refining",   _C["primary"],   0.50, 0.20, eff_primary)
            secondary_ref_idx = _add_node("Secondary Refining", _C["secondary"], 0.50, 0.70, eff_secondary)
        elif use_secondary_only or (eff_secondary > min_flow):
            secondary_ref_idx = _add_node("Secondary Refining", _C["secondary"], 0.50, 0.70, eff_secondary)
        elif use_primary_only or (eff_primary > min_flow):
            primary_ref_idx   = _add_node("Primary Refining",   _C["primary"],   0.50, 0.20, eff_primary)
    else:
        # BGS: single "Refining" node
        if eff_total > min_flow:
            bgs_ref_idx = _add_node("Refining", _C["refining"], 0.50, 0.50, eff_total)

    # Feedstock Pool (x=0.62) — always shown if any refining
    show_feedstock = (eff_total > 0) or (F1_feedstock > min_flow)
    if show_feedstock:
        feedstock_idx = _add_node("Feedstock Pool", _C["feedstock"], 0.62, 0.50, max(0.0, F1_feedstock))
    else:
        feedstock_idx = None

    # Battery Manufacturing (x=0.75)
    show_mfg = F4_batt_lead > min_flow
    if show_mfg:
        mfg_idx = _add_node("Battery Manufacturing", _C["battery_mfg"], 0.75, 0.65, F4_batt_lead)
    else:
        mfg_idx = None

    # Non-Battery Lead (x=0.75)
    show_nonbatt = F2_nonbatt > min_flow
    if show_nonbatt:
        nonbatt_idx = _add_node("Non-Battery Lead", _C["nonbatt"], 0.75, 0.30, F2_nonbatt)
    else:
        nonbatt_idx = None

    # ------------------------------------------------------------------
    # Source input nodes (x=0.01) — only if net positive
    # ------------------------------------------------------------------
    # We distribute y positions evenly among visible source nodes later.
    source_nodes: list[int] = []

    def _source(label: str, color: str, value: float) -> int:
        idx = _add_node(label, color, 0.01, 0.5, value)
        source_nodes.append(idx)
        return idx

    mine_idx         = None
    ore_imp_idx      = None
    ulab_imp_idx     = None
    scrap_imp_idx    = None
    feed_imp_idx     = None
    batt_imp_idx     = None
    metal_imp_idx    = None

    if mined > min_flow:
        mine_idx = _source("Mine Output", _C["mine"], mined)

    if imp_ore - exp_ore > min_flow:
        ore_imp_idx = _source("Ore Net Import", _C["ore_import"], imp_ore - exp_ore)

    if imp_used - exp_used > min_flow:
        ulab_imp_idx = _source("ULAB Net Import", _C["ulab_import"], imp_used - exp_used)

    if imp_scrap - exp_scrap > min_flow:
        scrap_imp_idx = _source("Scrap Net Import", _C["scrap_import"], imp_scrap - exp_scrap)

    if imp_feed - exp_feed > min_flow:
        feed_imp_idx = _source("Feed Net Import", _C["feed_import"], imp_feed - exp_feed)

    if imp_batt - exp_batt > min_flow:
        batt_imp_idx = _source("Battery Net Import", _C["batt_import"], imp_batt - exp_batt)

    if dataset == "hs12" and (imp_metal - exp_metal) > min_flow:
        metal_imp_idx = _source("Metal Products Import", _C["metal_import"], imp_metal - exp_metal)

    # ------------------------------------------------------------------
    # Sink output nodes (x=0.90)
    # ------------------------------------------------------------------
    sink_nodes: list[int] = []

    def _sink(label: str, color: str, value: float) -> int:
        idx = _add_node(label, color, 0.90, 0.5, value)
        sink_nodes.append(idx)
        return idx

    batt_service_idx      = None
    lead_products_idx     = None
    disposal_idx          = None
    ore_exp_idx           = None
    ulab_exp_idx          = None
    scrap_exp_idx         = None
    feed_exp_idx          = None
    batt_exp_idx          = None
    metal_exp_idx         = None

    if F5_implied > min_flow:
        batt_service_idx = _sink("Battery Service", _C["batt_service"], F5_implied)

    # Lead products terminal (non-battery lead + metal products net import)
    lead_products_val = F2_nonbatt + max(0.0, imp_metal - exp_metal)
    if lead_products_val > min_flow:
        lead_products_idx = _sink("Lead Products", _C["lead_products"], lead_products_val)

    if B6_disposed > min_flow:
        disposal_idx = _sink("Permanent Disposal", _C["disposal"], B6_disposed)

    if exp_ore - imp_ore > min_flow:
        ore_exp_idx = _sink("Ore Exports", _C["export_sink"], exp_ore - imp_ore)

    if exp_used - imp_used > min_flow:
        ulab_exp_idx = _sink("ULAB Exports", _C["export_sink"], exp_used - imp_used)

    if exp_scrap - imp_scrap > min_flow:
        scrap_exp_idx = _sink("Scrap Exports", _C["export_sink"], exp_scrap - imp_scrap)

    if exp_feed - imp_feed > min_flow:
        feed_exp_idx = _sink("Feed Exports", _C["export_sink"], exp_feed - imp_feed)

    if exp_batt - imp_batt > min_flow:
        batt_exp_idx = _sink("Battery Exports", _C["export_sink"], exp_batt - imp_batt)

    if dataset == "hs12" and (exp_metal - imp_metal) > min_flow:
        metal_exp_idx = _sink("Metal Products Exports", _C["export_sink"], exp_metal - imp_metal)

    # ------------------------------------------------------------------
    # Links — backward chain
    # ------------------------------------------------------------------

    # Mine Output → Ore Available
    if mine_idx is not None and ore_pool_idx is not None:
        _add_link(mine_idx, ore_pool_idx, mined, _C["mine"],
                  "Mine Output → Ore Available")

    # Ore Net Import → Ore Available
    if ore_imp_idx is not None and ore_pool_idx is not None:
        _add_link(ore_imp_idx, ore_pool_idx, imp_ore - exp_ore, _C["ore_import"],
                  "Ore Net Import → Ore Available")

    # Ore Available → Ore Exports (net exporter of ore)
    if ore_pool_idx is not None and ore_exp_idx is not None:
        _add_link(ore_pool_idx, ore_exp_idx, exp_ore - imp_ore, _C["ore_pool"],
                  "Ore Available → Ore Exports")

    # Ore Available → Primary Refining
    if ore_pool_idx is not None and eff_primary > min_flow:
        ref_tgt = primary_ref_idx if primary_ref_idx is not None else bgs_ref_idx
        if ref_tgt is not None:
            _add_link(ore_pool_idx, ref_tgt, eff_primary, _C["ore_pool"],
                      "Ore Available → Primary Refining")

    # ULAB Net Import → Battery Collection
    if ulab_imp_idx is not None and collection_idx is not None:
        _add_link(ulab_imp_idx, collection_idx, imp_used - exp_used, _C["ulab_import"],
                  "ULAB Net Import → ULAB Collection")

    # Battery Collection → ULAB Exports
    if collection_idx is not None and ulab_exp_idx is not None:
        _add_link(collection_idx, ulab_exp_idx, exp_used - imp_used, _C["collection"],
                  "ULAB Collection → ULAB Exports")

    # Battery Collection → Battery Breaking
    if collection_idx is not None and breaking_idx is not None:
        _add_link(collection_idx, breaking_idx, B4_collected, _C["collection"],
                  "ULAB Collection → Battery Breaking")

    # Battery Collection → Permanent Disposal
    # B6 = B4 * (1/gamma - 1)
    if collection_idx is not None and disposal_idx is not None:
        _add_link(collection_idx, disposal_idx, B6_disposed, _C["collection"],
                  "ULAB Collection → Permanent Disposal")

    # Battery Breaking → Scrap Pool
    if breaking_idx is not None and scrap_pool_idx is not None:
        _add_link(breaking_idx, scrap_pool_idx, B2_break_out, _C["breaking"],
                  "Battery Breaking → Scrap Pool")

    # Scrap Net Import → Scrap Pool
    if scrap_imp_idx is not None and scrap_pool_idx is not None:
        _add_link(scrap_imp_idx, scrap_pool_idx, imp_scrap - exp_scrap, _C["scrap_import"],
                  "Scrap Net Import → Scrap Pool")

    # Scrap Pool → Scrap Exports
    if scrap_pool_idx is not None and scrap_exp_idx is not None:
        _add_link(scrap_pool_idx, scrap_exp_idx, exp_scrap - imp_scrap, _C["scrap_pool"],
                  "Scrap Pool → Scrap Exports")

    # Scrap Pool → Secondary Refining (or BGS Refining)
    if scrap_pool_idx is not None and eff_secondary > min_flow:
        sec_tgt = secondary_ref_idx if secondary_ref_idx is not None else bgs_ref_idx
        if sec_tgt is not None:
            _add_link(scrap_pool_idx, sec_tgt, eff_secondary, _C["scrap_pool"],
                      "Scrap Pool → Secondary Refining")

    # ------------------------------------------------------------------
    # Links — forward chain
    # ------------------------------------------------------------------

    # Primary Refining → Feedstock Pool
    if primary_ref_idx is not None and feedstock_idx is not None:
        _add_link(primary_ref_idx, feedstock_idx, eff_primary, _C["primary"],
                  "Primary Refining → Feedstock Pool")

    # Secondary Refining → Feedstock Pool
    if secondary_ref_idx is not None and feedstock_idx is not None:
        _add_link(secondary_ref_idx, feedstock_idx, eff_secondary, _C["secondary"],
                  "Secondary Refining → Feedstock Pool")

    # BGS Refining → Feedstock Pool
    if bgs_ref_idx is not None and feedstock_idx is not None:
        _add_link(bgs_ref_idx, feedstock_idx, eff_total, _C["refining"],
                  "Refining → Feedstock Pool")

    # Feed Net Import → Feedstock Pool
    if feed_imp_idx is not None and feedstock_idx is not None:
        _add_link(feed_imp_idx, feedstock_idx, imp_feed - exp_feed, _C["feed_import"],
                  "Feed Net Import → Feedstock Pool")

    # Feedstock Pool → Feed Exports
    if feedstock_idx is not None and feed_exp_idx is not None:
        _add_link(feedstock_idx, feed_exp_idx, exp_feed - imp_feed, _C["feedstock"],
                  "Feedstock Pool → Feed Exports")

    # Feedstock Pool → Non-Battery Lead
    if feedstock_idx is not None and nonbatt_idx is not None:
        _add_link(feedstock_idx, nonbatt_idx, F2_nonbatt, _C["feedstock"],
                  "Feedstock Pool → Non-Battery Lead")

    # Feedstock Pool → Battery Manufacturing
    if feedstock_idx is not None and mfg_idx is not None:
        _add_link(feedstock_idx, mfg_idx, F3_feed_batt, _C["feedstock"],
                  "Feedstock Pool → Battery Manufacturing")

    # Battery Manufacturing → Battery Service
    if mfg_idx is not None and batt_service_idx is not None:
        _add_link(mfg_idx, batt_service_idx, F4_batt_lead, _C["battery_mfg"],
                  "Battery Manufacturing → Battery Service")

    # Battery Net Import → Battery Service
    if batt_imp_idx is not None and batt_service_idx is not None:
        _add_link(batt_imp_idx, batt_service_idx, imp_batt - exp_batt, _C["batt_import"],
                  "Battery Net Import → Battery Service")

    # Battery Service → Battery Exports
    if batt_service_idx is not None and batt_exp_idx is not None:
        _add_link(batt_service_idx, batt_exp_idx, exp_batt - imp_batt, _C["batt_service"],
                  "Battery Service → Battery Exports")

    # Non-Battery Lead → Lead Products
    if nonbatt_idx is not None and lead_products_idx is not None:
        _add_link(nonbatt_idx, lead_products_idx, F2_nonbatt, _C["nonbatt"],
                  "Non-Battery Lead → Lead Products")

    # Metal Products Import → Lead Products
    if metal_imp_idx is not None and lead_products_idx is not None:
        _add_link(metal_imp_idx, lead_products_idx, imp_metal - exp_metal, _C["metal_import"],
                  "Metal Products Import → Lead Products")

    # Lead Products → Metal Products Exports
    if lead_products_idx is not None and metal_exp_idx is not None:
        _add_link(lead_products_idx, metal_exp_idx, exp_metal - imp_metal, _C["lead_products"],
                  "Lead Products → Metal Products Exports")

    # ------------------------------------------------------------------
    # Fallback: if no links were produced, return None
    # ------------------------------------------------------------------
    if not links_src:
        return None, model_outputs, "No flows above the minimum threshold to display."

    # ------------------------------------------------------------------
    # Y-position distribution for source (x=0.01) and sink (x=0.90) nodes
    # ------------------------------------------------------------------
    def _spread_y_uniform(idxs: list[int], lo: float = 0.04, hi: float = 0.96) -> None:
        n_ = len(idxs)
        if n_ == 0:
            return
        if n_ == 1:
            node_y[idxs[0]] = (lo + hi) / 2.0
            return
        step = (hi - lo) / n_
        for k, idx in enumerate(idxs):
            node_y[idx] = round(lo + k * step + step / 2.0, 4)

    _spread_y_uniform(source_nodes)
    _spread_y_uniform(sink_nodes)

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label         = node_labels,
                color         = node_colors,
                x             = node_x,
                y             = node_y,
                pad           = 18,
                thickness     = 20,
                customdata    = node_hover,
                hovertemplate = "%{customdata}<extra></extra>",
            ),
            link=dict(
                source        = links_src,
                target        = links_tgt,
                value         = links_val,
                color         = links_color,
                customdata    = links_hover,
                hovertemplate = "%{customdata}<extra></extra>",
            ),
        )
    )

    fig.update_layout(
        height        = 650,
        font          = dict(family="Arial", size=13, color="#222222"),
        paper_bgcolor = "white",
        margin        = dict(l=10, r=10, t=60, b=10),
        title         = dict(
            text=(
                f"<b>{country} — Lead Mass Balance</b>  "
                f"({period_label}) &nbsp;&nbsp;"
                f"<span style='font-size:12px;font-weight:normal;color:#777'>"
                f"values in t Pb &nbsp;·&nbsp; "
                f"{'Anchor: Eurostat LAB collection' if _use_eurostat else f'Anchor: {mining_source} smelting'}"
                f"</span>"
            ),
            x    = 0.0,
            font = dict(size=15, family="Arial", color="#111111"),
        ),
    )

    return fig, model_outputs, warning_msg


# ---------------------------------------------------------------------------
# Summary table helper
# ---------------------------------------------------------------------------

def _build_summary_table(outputs: dict) -> pd.DataFrame:
    """Return a two-column DataFrame suitable for st.dataframe display."""
    rows = [
        ("Effective Primary Refining (t Pb)",   outputs["eff_primary"]),
        ("Effective Secondary Refining (t Pb)",  outputs["eff_secondary"]),
        ("", ""),
        ("B1 — Scrap Domestic Pool (t Pb)",      outputs["B1_scrap_dom"]),
        ("B2 — Breaking Output (t Pb)",          outputs["B2_break_out"]),
        ("B3 — Used Battery Pool (t Pb)",        outputs["B3_used_dom"]),
        ("B4 — Collected ULABs (t Pb)",          outputs["B4_collected"]),
        ("B5 — Total ULAB Generated (t Pb)",     outputs["B5_total_ulab"]),
        ("B6 — Permanent Disposal (t Pb)",       outputs["B6_disposed"]),
        ("", ""),
        ("F1 — Feedstock Pool (t Pb)",           outputs["F1_feedstock"]),
        ("F2 — Non-Battery Lead (t Pb)",         outputs["F2_nonbatt"]),
        ("F3 — Feed to Battery Mfg (t Pb)",      outputs["F3_feed_batt"]),
        ("F4 — Battery Lead Produced (t Pb)",    outputs["F4_batt_lead"]),
        ("F5 — Implied Installation (t Pb)",     outputs["F5_implied"]),
        ("", ""),
        ("Import — Ore (t Pb)",   outputs["imp_ore"]),
        ("Export — Ore (t Pb)",   outputs["exp_ore"]),
        ("Import — ULAB (t Pb)",  outputs["imp_used"]),
        ("Export — ULAB (t Pb)",  outputs["exp_used"]),
        ("Import — Scrap (t Pb)", outputs["imp_scrap"]),
        ("Export — Scrap (t Pb)", outputs["exp_scrap"]),
        ("Import — Feed (t Pb)",  outputs["imp_feed"]),
        ("Export — Feed (t Pb)",  outputs["exp_feed"]),
        ("Import — Battery (t Pb)", outputs["imp_batt"]),
        ("Export — Battery (t Pb)", outputs["exp_batt"]),
    ]
    formatted = []
    for label, val in rows:
        if label == "":
            formatted.append({"Variable": "", "Value (t Pb)": ""})
        else:
            formatted.append({
                "Variable":    label,
                "Value (t Pb)": f"{val:,.0f}" if isinstance(val, (int, float)) and val != "" else str(val),
            })
    return pd.DataFrame(formatted)


# ---------------------------------------------------------------------------
# Radar chart helper
# ---------------------------------------------------------------------------

def _build_radar_chart(
    outputs: dict,
    country: str,
    period_label: str,
) -> go.Figure | None:
    """
    4-axis radar showing the battery recycling activity profile for a country.
    Axes clockwise from top: ULAB Collection → Breaking → Secondary Smelting → Manufacturing.
    All values normalised to the peak activity (= 1.0); hover shows absolute t Pb.
    Returns None if all four values are zero.
    """
    collecting = max(0.0, outputs.get("B4_collected",  0.0))
    breaking   = max(0.0, outputs.get("B2_break_out",  0.0))
    smelting   = max(0.0, outputs.get("eff_secondary", 0.0))
    mfg        = max(0.0, outputs.get("F4_batt_lead",  0.0))

    peak = max(collecting, breaking, smelting, mfg)
    if peak == 0:
        return None

    cats = ["ULAB Collection", "Breaking", "Sec. Smelting", "Manufacturing"]
    vals = [collecting, breaking, smelting, mfg]
    r    = [v / peak for v in vals]
    hover = [f"<b>{c}</b><br>{v:,.0f} t Pb" for c, v in zip(cats, vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r             = r + [r[0]],
        theta         = cats + [cats[0]],
        fill          = "toself",
        fillcolor     = "rgba(41, 182, 246, 0.18)",
        line          = dict(color="#0288D1", width=2.5),
        text          = hover + [hover[0]],
        hovertemplate = "%{text}<extra></extra>",
        showlegend    = False,
    ))
    fig.update_layout(
        height        = 340,
        margin        = dict(l=50, r=50, t=55, b=30),
        font          = dict(family="Arial", size=12, color="#222222"),
        paper_bgcolor = "white",
        title         = dict(
            text = (
                f"<b>Recycling Activity Profile — {country}</b>  "
                f"<span style='font-size:11px;font-weight:normal;color:#777'>"
                f"({period_label}, normalised to peak)</span>"
            ),
            x    = 0.0,
            font = dict(size=13, family="Arial", color="#111111"),
        ),
        polar = dict(
            bgcolor    = "white",
            radialaxis = dict(
                visible   = True,
                range     = [0, 1.05],
                tickvals  = [0.25, 0.5, 0.75, 1.0],
                ticktext  = ["25%", "50%", "75%", "100%"],
                tickfont  = dict(size=9, color="#999"),
                gridcolor = "#E8E8E8",
                linecolor = "#CCCCCC",
                angle     = 90,
            ),
            angularaxis = dict(
                rotation  = 90,
                direction = "clockwise",
                tickfont  = dict(size=12, family="Arial", color="#333333"),
                gridcolor = "#E8E8E8",
                linecolor = "#CCCCCC",
            ),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Main tab entry point
# ---------------------------------------------------------------------------

def render_mass_balance_sankey_tab(
    baci_df: pd.DataFrame,
    mining_df: pd.DataFrame,
    region_map: dict,
    regions_ordered: list[str],
    sidebar_year: int,
    active_years: list[int],
    dataset: str,
    pb_factors: dict[int, float],
    mining_source: str,
    eurostat_df: dict | None = None,
) -> None:
    """
    Render the Mass Balance Sankey tab inside a Streamlit app.

    Parameters
    ----------
    baci_df         : BACI trade data with columns Year, Exporter, Importer,
                      Product, Value, Quantity, actual_lead, category.
    mining_df       : country_year_mining_refining.csv as a DataFrame.
                      Columns: country, year, mined_usgs_t, mined_bgs_t,
                      refined_bgs_t, refined_primary_usgs_t, refined_secondary_usgs_t.
    region_map      : dict mapping region name → set/list of BACI country names.
    regions_ordered : ordered list of region names.
    sidebar_year    : currently selected center year (int).
    active_years    : list of years to average over (1 or 3 elements).
    dataset         : "hs12" or "hs22".
    pb_factors      : dict mapping HS code (int) → Pb content fraction.
    mining_source   : "BGS" or "USGS".
    """
    # ------------------------------------------------------------------
    # Filter to countries with at least some refining data (BGS or USGS)
    # ------------------------------------------------------------------
    _refining_cols = ["refined_bgs_t", "refined_primary_usgs_t", "refined_secondary_usgs_t"]
    _has_refining = mining_df[
        [c for c in _refining_cols if c in mining_df.columns]
    ].notna().any(axis=1)
    countries_with_refining: set[str] = set(mining_df.loc[_has_refining, "country"].unique())

    # ------------------------------------------------------------------
    # All BACI countries from region_map (preserves region grouping order)
    # ------------------------------------------------------------------
    all_countries: list[str] = []
    seen: set[str] = set()
    for region in regions_ordered:
        for c in sorted(region_map.get(region, [])):
            if c not in seen and c in countries_with_refining:
                all_countries.append(c)
                seen.add(c)
    # Fallback: any country in baci_df not already included
    for c in sorted(set(baci_df["Exporter"].unique()) | set(baci_df["Importer"].unique())):
        if c not in seen and c in countries_with_refining:
            all_countries.append(c)
            seen.add(c)

    if not all_countries:
        st.warning("No countries found in the trade data.")
        return

    # ------------------------------------------------------------------
    # Disclaimer
    # ------------------------------------------------------------------
    st.info(
        "**Experimental model.** Values are estimates derived from trade and production "
        "data and will change as parameters are adjusted. Results should be treated as "
        "indicative, not precise."
    )

    # ------------------------------------------------------------------
    # Layout: narrow left column (controls) + wide right column (chart)
    # ------------------------------------------------------------------
    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.markdown("#### Country")
        country = st.selectbox(
            "Select country",
            options=all_countries,
            index=0,
            key="sankey_country",
            label_visibility="collapsed",
        )

        if len(active_years) > 1:
            st.caption(
                f"Averaging {min(active_years)}–{max(active_years)} "
                f"({len(active_years)} years)"
            )
        else:
            st.caption(f"Single year: {active_years[0]}")

        # --- Anchor selection (Eurostat toggle disabled — preserved for future use) ---
        _is_eu_eligible = country in EUROSTAT_ELIGIBLE
        anchor_choice = "USGS / BGS smelting output"

        st.markdown("#### Display")
        min_flow = st.number_input(
            "Min flow threshold (t Pb)",
            min_value=0,
            value=10,
            step=10,
            help=(
                "Links smaller than this value (in tonnes of lead content) "
                "are hidden to reduce visual clutter."
            ),
            key="sankey_min_flow",
        )

        st.markdown("#### Process parameters")
        with st.expander("Process parameters", expanded=False):
            eta_secondary = st.slider(
                "Net secondary smelting recovery (η_secondary)",
                min_value=0.80, max_value=1.00, value=0.97, step=0.01, format="%.2f",
                key="sankey_eta_secondary",
            )
            eta_break = st.slider(
                "Breaking recovery (η_break)",
                min_value=0.70, max_value=1.00, value=0.95, step=0.01, format="%.2f",
                key="sankey_eta_break",
            )
            delta_pb = st.slider(
                "Pb retained at end-of-life (δ)",
                min_value=0.80, max_value=1.00, value=0.95, step=0.01, format="%.2f",
                key="sankey_delta_pb",
            )
            beta = st.slider(
                "Battery share of lead demand (β)",
                min_value=0.50, max_value=1.00, value=0.85, step=0.01, format="%.2f",
                key="sankey_beta",
            )
            eta_mfg = st.slider(
                "Manufacturing efficiency (η_mfg)",
                min_value=0.90, max_value=1.00, value=0.98, step=0.01, format="%.2f",
                key="sankey_eta_mfg",
            )
            eta_ore = st.slider(
                "Primary smelting recovery (η_ore)",
                min_value=0.80, max_value=1.00, value=0.95, step=0.01, format="%.2f",
                key="sankey_eta_ore",
            )
            gamma = st.slider(
                "Collection rate (γ) — overrides country default",
                min_value=0.30, max_value=1.00, value=0.90, step=0.01, format="%.2f",
                key="sankey_gamma",
                help=(
                    "Default is 0.90 (10% disposal). Adjust this for the specific "
                    "country you are modelling — e.g. ~0.99 for USA/Japan, ~0.70 "
                    "for India, ~0.60 for Nigeria."
                ),
            )
            battery_lead_content_fraction = 0.65
            if _is_eu_eligible and anchor_choice == "Eurostat collection data":
                battery_lead_content_fraction = st.slider(
                    "Battery lead content fraction (Eurostat conversion)",
                    min_value=0.55, max_value=0.75, value=0.65, step=0.01, format="%.2f",
                    key="sankey_batt_lc_fraction",
                    help=(
                        "Converts Eurostat total battery weight (t LAB/yr) to lead "
                        "content. SLI-dominated streams: ~0.65. Industrial-heavy: "
                        "~0.70. Applied before the δ (end-of-life retention) factor."
                    ),
                )

    # ------------------------------------------------------------------
    # Eurostat anchor: resolve availability and effective mode
    # ------------------------------------------------------------------
    _eurostat_available = (
        _is_eu_eligible
        and eurostat_df is not None
        and any(eurostat_df.get((country, y), 0.0) > 0 for y in active_years)
    )
    _want_eurostat = anchor_choice == "Eurostat collection data"
    _anchor_mode = "eurostat" if (_want_eurostat and _eurostat_available) else "mining"

    # Average Eurostat values over the selected years (missing years excluded)
    if _anchor_mode == "eurostat" and eurostat_df is not None:
        _evals = [
            eurostat_df[(country, y)]
            for y in active_years
            if eurostat_df.get((country, y), 0.0) > 0
        ]
        _eurostat_input_t = sum(_evals) / len(_evals) if _evals else 0.0
    else:
        _eurostat_input_t = 0.0

    # ------------------------------------------------------------------
    # Load master BACI (cached)
    # ------------------------------------------------------------------
    master_df = _load_master_baci() if dataset == "hs12" else pd.DataFrame(
        columns=["Year", "Exporter", "Importer", "Product", "Quantity"]
    )

    # ------------------------------------------------------------------
    # Build Sankey
    # ------------------------------------------------------------------
    fig, model_outputs, warning_msg = build_sankey(
        baci_df                    = baci_df,
        master_df                  = master_df,
        mining_df                  = mining_df,
        country                    = country,
        active_years               = active_years,
        dataset                    = dataset,
        pb_factors                 = pb_factors,
        mining_source              = mining_source,
        eta_secondary              = eta_secondary,
        eta_break                  = eta_break,
        delta_pb                   = delta_pb,
        beta                       = beta,
        eta_mfg                    = eta_mfg,
        eta_ore                    = eta_ore,
        gamma                      = gamma,
        min_flow                   = float(min_flow),
        anchor_mode                = _anchor_mode,
        eurostat_input_t           = _eurostat_input_t,
        battery_lead_content_fraction = battery_lead_content_fraction,
    )

    with right_col:
        # Eurostat fallback warning
        if _want_eurostat and not _eurostat_available:
            st.warning(
                f"Eurostat collection data not available for **{country}** "
                f"in {min(active_years)}–{max(active_years) if len(active_years) > 1 else active_years[0]}. "
                f"Eurostat covers EU/EFTA countries for years 2009–2023. "
                f"Falling back to {mining_source} smelting anchor."
            )

        # Refining-hub note (only when Eurostat anchor is active)
        if _anchor_mode == "eurostat" and country in REFINING_HUBS:
            st.info(
                f"**{country}** is a major lead refining hub that processes batteries "
                f"and bullion imported from outside the EU. The Eurostat anchor reflects "
                f"domestically collected batteries only. Smelting output implied by this "
                f"anchor will be lower than {mining_source} figures — the gap represents "
                f"non-EU feedstock flows visible in BACI HS 854810 (ULAB) and "
                f"780199 (bullion) imports."
            )

        if warning_msg == "__no_refining__":
            st.warning(
                f"No refining data found for **{country}** in the selected years "
                f"from either BGS or USGS. The model cannot anchor secondary smelting "
                f"without a refining estimate. Try switching the mining & refining "
                f"source in the sidebar, or selecting different years."
            )
            # Still show whatever BACI trade flows exist in the caption
            st.caption(
                "Trade flows from BACI are available but cannot be connected to "
                "the refining anchor. Adjust parameters or switch data source."
            )
        elif warning_msg:
            st.info(warning_msg)

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        elif warning_msg != "__no_refining__":
            st.info(
                f"No flows above the {min_flow:,.0f} t threshold for **{country}** "
                f"in the selected period. Try lowering the minimum flow threshold "
                f"or selecting a different country or year range."
            )

    # ------------------------------------------------------------------
    # Summary table (full width, below the chart)
    # ------------------------------------------------------------------
    st.divider()
    with st.expander("Model output details", expanded=False):
        summary_df = _build_summary_table(model_outputs)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Variable":    st.column_config.TextColumn("Variable", width="large"),
                "Value (t Pb)": st.column_config.TextColumn("Value (t Pb)", width="medium"),
            },
        )
        st.caption(
            "All values are metric tonnes of lead content, averaged over the selected "
            f"period ({', '.join(str(y) for y in active_years)}). "
            "Backward chain (B1–B6) is anchored to the refining estimate. "
            "Forward chain (F1–F5) derives implied installation from refining output."
        )


# ---------------------------------------------------------------------------
# Economy Snapshot — multi-country radar
# ---------------------------------------------------------------------------

def _fmt_tick(v: float) -> str:
    """Format a tonnage value for radar tick labels."""
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}k"
    return f"{v:.0f}"


def _build_snapshot_radar(
    outputs: dict,
    country: str,
    period_label: str,
    show_mining: bool = False,
    show_collection_baseline: bool = False,
    norm_max: float | None = None,
) -> go.Figure | None:
    """
    Radar for Economy Snapshot tab.

    Optional trace:
      • "Collection baseline" (dashed grey): ULAB Collection value shown at every
        recycling stage — a perfect loop would fill to this line at all steps.

    Axes clockwise from top:
      ULAB Collection → Breaking → Sec. Smelting → Manufacturing [→ Mining]
    """
    ulab_dom  = max(0.0, outputs.get("B4_collected",  0.0))
    break_dom = max(0.0, outputs.get("B2_break_out",  0.0))
    smelt     = max(0.0, outputs.get("eff_secondary", 0.0))
    mfg_dom   = max(0.0, outputs.get("F4_batt_lead",  0.0))
    mine_dom  = max(0.0, outputs.get("mined",          0.0))

    cats     = ["ULAB Collection", "Breaking", "Sec. Smelting", "Manufacturing"]
    dom_vals = [ulab_dom, break_dom, smelt, mfg_dom]

    if show_mining:
        cats.append("Mining")
        dom_vals.append(mine_dom)

    peak = norm_max if norm_max is not None else (max(dom_vals) if dom_vals else 0.0)
    if peak == 0:
        return None

    r_dom     = [v / peak for v in dom_vals]
    hover_dom = [f"<b>{c}</b><br>{v:,.0f} t Pb" for c, v in zip(cats, dom_vals)]

    tick_text = (
        [_fmt_tick(f * peak) for f in [0.25, 0.5, 0.75, 1.0]]
        if norm_max is not None
        else ["25 %", "50 %", "75 %", "100 %"]
    )

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r             = r_dom + [r_dom[0]],
        theta         = cats  + [cats[0]],
        fill          = "toself",
        fillcolor     = "rgba(41, 182, 246, 0.18)",
        line          = dict(color="#0288D1", width=2.5),
        text          = hover_dom + [hover_dom[0]],
        hovertemplate = "%{text}<extra></extra>",
        name          = "Recycling activity",
        showlegend    = show_collection_baseline,
    ))

    if show_collection_baseline and ulab_dom > 0:
        # Dashed square at ULAB Collection level across all 4 recycling axes.
        # Shows what each downstream stage would look like if it fully used
        # everything that was collected — the "perfect loop" reference.
        _bl_cats = ["ULAB Collection", "Breaking", "Sec. Smelting", "Manufacturing"]
        _bl_r    = [ulab_dom / peak] * 4
        _bl_hover = [
            f"<b>{c}</b><br>Collection baseline: {ulab_dom:,.0f} t Pb"
            for c in _bl_cats
        ]
        fig.add_trace(go.Scatterpolar(
            r             = _bl_r + [_bl_r[0]],
            theta         = _bl_cats + [_bl_cats[0]],
            fill          = "none",
            line          = dict(color="#555555", width=1.5, dash="dash"),
            text          = _bl_hover + [_bl_hover[0]],
            hovertemplate = "%{text}<extra></extra>",
            name          = "Collection baseline",
            showlegend    = True,
        ))

    fig.update_layout(
        height        = 340,
        margin        = dict(l=50, r=50, t=60, b=40),
        font          = dict(family="Arial", size=11, color="#222222"),
        paper_bgcolor = "white",
        title         = dict(
            text      = (
                f"<b>{country}</b><br>"
                f"<span style='font-size:10px;font-weight:normal;color:#777'>"
                f"{period_label}</span>"
            ),
            x         = 0.5,
            xanchor   = "center",
            font      = dict(size=13, family="Arial", color="#111111"),
        ),
        legend        = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = -0.18,
            xanchor     = "center",
            x           = 0.5,
            font        = dict(size=10),
        ) if show_collection_baseline else dict(visible=False),
        polar         = dict(
            bgcolor    = "white",
            radialaxis = dict(
                visible   = True,
                range     = [0, 1.05],
                tickvals  = [0.25, 0.5, 0.75, 1.0],
                ticktext  = tick_text,
                tickfont  = dict(size=9, color="#999"),
                gridcolor = "#E8E8E8",
                linecolor = "#CCCCCC",
                angle     = 90,
            ),
            angularaxis = dict(
                rotation  = 90,
                direction = "clockwise",
                tickfont  = dict(size=10, family="Arial", color="#333333"),
                gridcolor = "#E8E8E8",
                linecolor = "#CCCCCC",
            ),
        ),
    )
    return fig


def render_economy_snapshot_tab(
    baci_df: pd.DataFrame,
    mining_df: pd.DataFrame,
    region_map: dict,
    regions_ordered: list[str],
    active_years: list[int],
    dataset: str,
    pb_factors: dict[int, float],
    mining_source: str,
) -> None:
    """
    Recycling Economy Snapshot tab — side-by-side radar comparison of up to 3 countries.
    """
    st.markdown("### Recycling Economy Snapshot")
    st.write(
        "Each country occupies a different position in the lead-acid battery lifecycle. "
        "The radar shows activity at four stages of the recycling loop: "
        "**ULAB Collection** → **Breaking** → **Secondary Smelting** → **Manufacturing**. "
        "Each chart is normalised to the country's own peak stage. "
        "A country with a full, closed recycling economy fills all four axes equally. "
        "Countries that only collect and export scrap show high Collection with low Smelting "
        "and near-zero Manufacturing; import-dependent assemblers show the reverse."
    )
    # Eligible countries (same filter as Process Estimates)
    _refining_cols = ["refined_bgs_t", "refined_primary_usgs_t", "refined_secondary_usgs_t"]
    _has_refining = mining_df[
        [c for c in _refining_cols if c in mining_df.columns]
    ].notna().any(axis=1)
    _eligible: set[str] = set(mining_df.loc[_has_refining, "country"].unique())

    all_countries: list[str] = []
    seen: set[str] = set()
    for region in regions_ordered:
        for c in sorted(region_map.get(region, [])):
            if c not in seen and c in _eligible:
                all_countries.append(c)
                seen.add(c)
    for c in sorted(set(baci_df["Exporter"].unique()) | set(baci_df["Importer"].unique())):
        if c not in seen and c in _eligible:
            all_countries.append(c)
            seen.add(c)

    period_label = (
        f"{min(active_years)}–{max(active_years)} avg"
        if len(active_years) > 1
        else str(active_years[0])
    )
    NONE_OPT = "— none —"
    opts = [NONE_OPT] + all_countries

    # ── Country selectors ──────────────────────────────────────────────
    sel_col1, sel_col2, sel_col3 = st.columns(3)
    defaults = ["Germany", "France", "Poland"]
    with sel_col1:
        c1 = st.selectbox(
            "Country 1", opts,
            index=opts.index(defaults[0]) if defaults[0] in opts else 0,
            key="snap_c1",
        )
    with sel_col2:
        c2 = st.selectbox(
            "Country 2", opts,
            index=opts.index(defaults[1]) if defaults[1] in opts else 0,
            key="snap_c2",
        )
    with sel_col3:
        c3 = st.selectbox(
            "Country 3", opts,
            index=opts.index(defaults[2]) if defaults[2] in opts else 0,
            key="snap_c3",
        )

    selected = [c for c in [c1, c2, c3] if c and c != NONE_OPT]
    if not selected:
        st.info("Select at least one country above to display the radar.")
        return

    # ── Toggles ────────────────────────────────────────────────────────
    tog1, tog2, tog3, _ = st.columns([1, 1, 1, 2])
    with tog1:
        show_mining = st.toggle("Add Mining axis", value=False, key="snap_mining")
    with tog2:
        show_collection_baseline = st.toggle(
            "Show Collection Baseline",
            value=False,
            key="snap_collection_baseline",
            help=(
                "Adds a dashed line at the ULAB Collection level across all recycling "
                "stages. When a stage reaches this line, it is fully using everything "
                "that was collected. Stages below indicate losses or exports at that step."
            ),
        )
    with tog3:
        global_norm = st.toggle("Same scale for all", value=False, key="snap_norm",
                                 help=(
                                     "Normalize all three charts to the same peak so "
                                     "absolute sizes are comparable. Default: each chart "
                                     "is scaled to its own peak."
                                 ))

    # ── Compute model outputs for each country ─────────────────────────
    master_df = _load_master_baci() if dataset == "hs12" else pd.DataFrame(
        columns=["Year", "Exporter", "Importer", "Product", "Quantity"]
    )

    all_outputs: dict[str, dict] = {}
    for country in selected:
        _, outputs, _ = build_sankey(
            baci_df       = baci_df,
            master_df     = master_df,
            mining_df     = mining_df,
            country       = country,
            active_years  = active_years,
            dataset       = dataset,
            pb_factors    = pb_factors,
            mining_source = mining_source,
            min_flow      = 0.0,
        )
        all_outputs[country] = outputs

    # ── Global normalisation max ───────────────────────────────────────
    norm_max: float | None = None
    if global_norm:
        _all_vals: list[float] = []
        for outputs in all_outputs.values():
            vals = [
                outputs.get("B4_collected",  0.0),
                outputs.get("B2_break_out",  0.0),
                outputs.get("eff_secondary", 0.0),
                outputs.get("F4_batt_lead",  0.0),
            ]
            if show_mining:
                vals.append(outputs.get("mined", 0.0))
            _all_vals.extend(vals)
        norm_max = max(_all_vals) if _all_vals else None

    # ── Render radars ──────────────────────────────────────────────────
    cols = st.columns(len(selected))
    for i, country in enumerate(selected):
        with cols[i]:
            fig = _build_snapshot_radar(
                all_outputs[country], country, period_label,
                show_mining              = show_mining,
                show_collection_baseline = show_collection_baseline,
                norm_max                 = norm_max,
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No data for {country}.")

    if global_norm and norm_max:
        st.caption(
            f"All charts share a common scale — 100 % = {norm_max:,.0f} t Pb."
        )
    if show_collection_baseline:
        st.caption(
            "**Dashed grey line** = ULAB Collection baseline. "
            "When a stage reaches this line, it is processing at the same volume as "
            "what was collected — a closed loop. "
            "Stages below the baseline represent losses or exports at that step."
        )
