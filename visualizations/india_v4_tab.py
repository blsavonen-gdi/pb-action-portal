"""Streamlit tab: India v4 Mass Balance calculator (Phase-D dashboard).

Replaces the previous Mass Balance tab. The calculator surfaces the v4
three-stage chain at a country's inputs and *foregrounds the disagreement*
between the two anchors:

  - install fit  : res_install = (INSTALL_implied - INSTALL_target) / INSTALL_target
  - refine fit   : res_refine  = (REFINE_SEC     - USGS_sec)        / USGS_sec

Both are shown side-by-side and never blended into a single fit score.

Three indicators that report different things and can co-occur:
  (b) Infeasibility flag (hard, binary, install side): the stock-derived
      INSTALL_target exceeds the USGS+trade INSTALL_ceiling at every k.
      Statement: "an input is impossible."
  (c) Unrecorded-feedstock estimate (refine side, tonnage): when the chain
      over-produces refined secondary vs USGS, the surplus implies
      secondary refining beyond what USGS+trade records.
      Statement: "there is an unexplained surplus."

A k/tau ridge curve and two efficiency-phi curves let the user explore the
two structural ambiguities (the k/tau ridge and the formal/informal split).
Monte Carlo for parametric uncertainty is opt-in and cached.

Build conventions:
  - Thin UI over `india_model.model_v4`; no chain re-implementation here.
  - All session state is keyed under "v4_" so the tab is isolated.
  - The country dropdown is data-gated; for now only India is enabled.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from india_model.model_v4 import (
    load_segments, tau_eff, fit_growth_rate, retire_rate,
    load_inputs,
    FEED_HS, PARTS_HS, CRUDE_HS,
)
from india_model.model_v5_parallel import (
    forward_parallel_chain,
    REF_PHI, PHI_FLOORS, ETA_DEFAULTS as V5_ETA_DEFAULTS,
    BETA_DEFAULT, GAMMA_TOTAL,
    phi_is_ordered, crossovers_nonneg,
)


# ---------------------------------------------------------------------------
# Country registry (data-gated dropdown)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent


@dataclass(frozen=True)
class CountryPreset:
    """Per-country defaults loaded into the editable inputs panel."""
    label: str
    beta: float
    gamma: float
    inputs_csv: str              # india_model/india_mass_balance_2018_2023.csv
    segments_csv: str            # india_model/segment_lifetimes.csv
    fit_window: tuple[int, int]  # (first, last) smoothed window for residuals


COUNTRIES = {
    "India": CountryPreset(
        label="India",
        beta=0.86,                               # India working default (was BETA_DEFAULT=0.75 ILZSG)
        gamma=GAMMA_TOTAL,                       # 0.98 (Dalberg/USAID)
        inputs_csv=str(ROOT / "india_model" / "india_mass_balance_2018_2023.csv"),
        segments_csv=str(ROOT / "india_model" / "segment_lifetimes.csv"),
        fit_window=(2019, 2022),
    ),
}


# ---------------------------------------------------------------------------
# Efficiency defaults (Phase-D D.2 panel) — parallel chain (v5)
# ---------------------------------------------------------------------------
# Formal and informal η for every stage. Pulled from V5_ETA_DEFAULTS so the
# dashboard and the underlying model stay in lockstep.

EFF_DEFAULTS = dict(V5_ETA_DEFAULTS)   # all 9 keys: delta + 4*(F,I)

# Reference φ-vector — the starting point in §1 inputs.
# Per-stage REF_PHI values are pulled directly from REF_PHI["phi_*_f"] where
# needed (e.g. the φ-inputs panel). No mirror constants at module scope.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt2sig(x: float) -> str:
    """Format to 2 significant figures."""
    if pd.isna(x):
        return "nan"
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    exp = int(math.floor(math.log10(ax)))
    factor = 10 ** (exp - 1)
    rounded = round(ax / factor) * factor
    if rounded >= 1:
        return f"{sign}{int(round(rounded)):,}"
    return f"{sign}{rounded:.2g}"


def _fmt_kt(x: float) -> str:
    """Format a tonnage as kt with one decimal place."""
    if pd.isna(x):
        return "nan"
    return f"{x / 1000:,.1f} kt"


@st.cache_data(show_spinner=False)
def _load_raw_csv(path: str) -> pd.DataFrame:
    """Read the per-country mass-balance CSV (cached on path)."""
    return pd.read_csv(path).sort_values("year").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_segments_df(path: str) -> pd.DataFrame:
    """Read the per-country segment lifetimes CSV (cached on path)."""
    return load_segments(path)


# ---------------------------------------------------------------------------
# Inputs assembly
# ---------------------------------------------------------------------------

FIT_BACI_COLS_INDIA = [
    ("ORE",   "260700"),
    ("FEED_refined (780110/780191/282410/282490)",
                     "feed_refined"),
    ("PARTS (850790)", "850790"),
    ("CRUDE (780199)", "780199"),
    ("USED (854810)",  "854810"),
    ("SCRAP (780200)", "780200"),
    ("BATT (850710+850720)", "batt"),
]


def _build_arr_from_user_inputs(
    df_csv: pd.DataFrame,
    user_stock: pd.DataFrame,
    fit_window: tuple[int, int],
) -> dict:
    """Build the v4 `arr` dict on the fit window using the user's stock values.

    Smoothing convention matches `load_inputs(..., smooth_window=3, trim_years=...)`:
    a 3-yr centered rolling mean over the full source series, trimmed to the fit
    window. Year `t = first - 1` is the raw `stock_pre` anchor for dStock.

    The user's edited stock values overwrite df_csv["stock_total_t_pb"] BEFORE
    smoothing, so user edits propagate into all downstream flows.
    """
    df = df_csv.copy().sort_values("year").reset_index(drop=True)

    # Overwrite stock column with user values, year-by-year.
    user_stock = user_stock.copy()
    user_stock["year"] = user_stock["year"].astype(int)
    user_lookup = dict(zip(user_stock["year"].tolist(),
                           user_stock["stock_total_t_pb"].astype(float).tolist()))
    df["stock_total_t_pb"] = df["year"].astype(int).map(
        lambda y: user_lookup.get(int(y), float("nan"))
    )
    # Fall back to original CSV for any year the user didn't provide.
    df_csv_lookup = dict(zip(df_csv["year"].astype(int),
                             df_csv["stock_total_t_pb"].astype(float)))
    mask = df["stock_total_t_pb"].isna()
    if mask.any():
        df.loc[mask, "stock_total_t_pb"] = df.loc[mask, "year"].astype(int).map(
            lambda y: df_csv_lookup.get(int(y), float("nan"))
        )

    first, last = fit_window
    trim_years = list(range(first, last + 1))

    numeric = [c for c in df.columns if c != "year"]
    rolled = df[numeric].rolling(window=3, center=True, min_periods=3).mean()
    df_s = pd.concat([df["year"], rolled], axis=1)

    pre_year = first - 1
    pre_row = df.loc[df["year"] == pre_year, "stock_total_t_pb"]
    if pre_row.empty or pd.isna(pre_row.iloc[0]):
        raise ValueError(
            f"Stock value for {pre_year} (one year before first fit year) is "
            f"required as the dStock anchor."
        )
    stock_pre = float(pre_row.iloc[0])

    df_s = df_s[df_s["year"].isin(trim_years)].reset_index(drop=True)
    if df_s.empty:
        raise ValueError(f"No rows in fit window {first}-{last} after smoothing.")

    feed_imp = sum(df_s[f"imp_{hs}_t_pb"] for hs in FEED_HS)
    feed_exp = sum(df_s[f"exp_{hs}_t_pb"] for hs in FEED_HS)

    arr = {
        "year":      df_s["year"].astype(int).to_numpy(),
        "stock":     df_s["stock_total_t_pb"].to_numpy(dtype=float),
        "stock_pre": stock_pre,
        "mine_usgs": df_s["mine_pb_t_usgs"].to_numpy(dtype=float),
        "prim_usgs": df_s["primary_pb_t_usgs"].to_numpy(dtype=float),
        "sec_usgs":  df_s["secondary_pb_t_usgs"].to_numpy(dtype=float),
        "imp_ore":   df_s["imp_260700_t_pb"].to_numpy(dtype=float),
        "exp_ore":   df_s["exp_260700_t_pb"].to_numpy(dtype=float),
        "imp_feed":  feed_imp.to_numpy(dtype=float),
        "exp_feed":  feed_exp.to_numpy(dtype=float),
        "imp_parts": df_s[f"imp_{PARTS_HS}_t_pb"].to_numpy(dtype=float),
        "exp_parts": df_s[f"exp_{PARTS_HS}_t_pb"].to_numpy(dtype=float),
        "imp_crude": df_s[f"imp_{CRUDE_HS}_t_pb"].to_numpy(dtype=float),
        "exp_crude": df_s[f"exp_{CRUDE_HS}_t_pb"].to_numpy(dtype=float),
        "imp_batt":  (df_s["imp_850710_t_pb"] + df_s["imp_850720_t_pb"]).to_numpy(dtype=float),
        "exp_batt":  (df_s["exp_850710_t_pb"] + df_s["exp_850720_t_pb"]).to_numpy(dtype=float),
        "imp_used":  df_s["imp_854810_t_pb"].to_numpy(dtype=float),
        "exp_used":  df_s["exp_854810_t_pb"].to_numpy(dtype=float),
        "imp_scrap": df_s["imp_780200_t_pb"].to_numpy(dtype=float),
        "exp_scrap": df_s["exp_780200_t_pb"].to_numpy(dtype=float),
    }
    return arr


def _compute_g_from_stock(user_stock: pd.DataFrame) -> float:
    """Fit log-linear growth on the user's raw stock series."""
    yrs = user_stock["year"].astype(float).to_numpy()
    stk = user_stock["stock_total_t_pb"].astype(float).to_numpy()
    return fit_growth_rate(stk, yrs)


def _compute_tau_eff(segments_df: pd.DataFrame) -> float:
    return tau_eff(segments_df)


# ---------------------------------------------------------------------------
# Indicators (parallel-chain)
# ---------------------------------------------------------------------------

PHI_MAX_FORMAL = {
    "phi_break_f":  0.9987,
    "phi_smelt_f":  0.9988,
    "phi_refine_f": 0.9989,
    "phi_mfg_f":    0.9990,
}


def _install_target_at_k(arr: dict, k: float, g: float, tau: float) -> np.ndarray:
    """INSTALL_target = k * ((stock - stock_prev) + stock * retire_rate)."""
    r = retire_rate(g, tau)
    stock = arr["stock"]
    stock_prev = np.concatenate([[arr["stock_pre"]], stock[:-1]])
    return k * ((stock - stock_prev) + stock * r)


def _max_install_implied(arr: dict, g: float, tau: float, k: float,
                         beta: float, gamma: float, etas: dict) -> np.ndarray:
    """INSTALL_implied at φ → all-formal (the chain's supply ceiling at (k, τ)).

    Under the v5 floor framing, USGS secondary is no longer a hard cap, so the
    install ceiling is the chain itself running at maximum-formal recovery.
    """
    out = forward_parallel_chain(
        arr, k_stock=k, phi=PHI_MAX_FORMAL,
        g=g, tau=tau, beta=beta, gamma=gamma,
        **{k_: v for k_, v in etas.items() if k_ in V5_ETA_DEFAULTS},
    )
    return out["INSTALL_implied"]


def _max_refine_sec_F(arr: dict, g: float, tau: float, k: float,
                      beta: float, gamma: float, etas: dict) -> np.ndarray:
    """REFINE_SEC_F at φ → all-formal — the chain's refine supply ceiling."""
    out = forward_parallel_chain(
        arr, k_stock=k, phi=PHI_MAX_FORMAL,
        g=g, tau=tau, beta=beta, gamma=gamma,
        **{k_: v for k_, v in etas.items() if k_ in V5_ETA_DEFAULTS},
    )
    return out["REFINE_SEC_F"]


def _k_install_feasibility_crossing(arr: dict, g: float, tau: float,
                                    beta: float, gamma: float, etas: dict) -> float:
    """Window-sum k where INSTALL_target = INSTALL_implied(all-formal).

    Below this k the chain (at max formal) can supply the install target; above
    it the stock-derived demand exceeds even the all-formal chain's output.
    Computed by bisection on k since INSTALL_implied is (mostly) linear in k.
    """
    T = _install_target_at_k(arr, 1.0, g, tau)
    T_sum = float(np.sum(T))
    if T_sum <= 0:
        return float("nan")
    # Compute the chain at two k's, infer the linear-in-k coefficient.
    with np.errstate(divide="ignore", invalid="ignore"):
        I0 = float(np.sum(_max_install_implied(arr, g, tau, 1.0, beta, gamma, etas)))
        Ip = float(np.sum(_max_install_implied(arr, g, tau, 0.0, beta, gamma, etas)))
    # Solve: Ip + k·(I0 − Ip) = k·T_sum  =>  k = Ip / (T_sum − (I0 − Ip)).
    denom = T_sum - (I0 - Ip)
    if abs(denom) < 1e-12:
        return float("nan")
    return Ip / denom


def _k_refine_floor_crossing(arr: dict, g: float, tau: float,
                             beta: float, gamma: float, etas: dict) -> float:
    """Smallest k at which REFINE_SEC_F(all-formal) ≥ USGS_secondary (window-sum).

    Linear-in-k inference like _k_install_feasibility_crossing.
    """
    U_sum = float(np.sum(arr["sec_usgs"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        R0 = float(np.sum(_max_refine_sec_F(arr, g, tau, 1.0, beta, gamma, etas)))
        Rp = float(np.sum(_max_refine_sec_F(arr, g, tau, 0.0, beta, gamma, etas)))
    # REFINE_SEC_F(k) ≈ Rp + k·(R0 − Rp); solve for k at U_sum:
    a = R0 - Rp
    b = Rp - U_sum
    if abs(a) < 1e-12:
        return float("nan")
    return -b / a


def _unrecorded_feedstock_per_year(out: dict, sec_usgs: np.ndarray) -> np.ndarray:
    """Per-year overshoot = max(0, REFINE_SEC_F - USGS_sec).

    Under the v5 floor framing this is the EXPECTED implied unrecorded /
    informal-equivalent refined lead, not a failure signal.
    """
    return np.maximum(np.asarray(out["REFINE_SEC_F"]) - np.asarray(sec_usgs), 0.0)


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------

def _render_country_selector() -> str:
    """Data-gated country dropdown."""
    st.subheader("India Lead-Acid Battery Mass Balance — v4 calculator")
    st.caption(
        "A forward mass-balance chain for one country. The model exposes its "
        "**two anchor residuals** separately: stock-implied installs vs USGS-implied "
        "refining. They disagree by design; the UI does not combine them."
    )
    country = st.selectbox(
        "Country",
        options=list(COUNTRIES.keys()),
        index=0,
        key="v4_country",
        help="Only countries with USGS + BACI + stock data appear here. India is the only "
             "country supported in Phase D.",
    )
    return country


def _render_beta_gamma_inputs(preset: CountryPreset) -> tuple[float, float, float]:
    """β, γ, and k_stock editors in a single row.

    k_stock is the stock multiplier used everywhere downstream (§3–§8). It
    starts at 1.0 — a slider lets the user explore scaling the reported stock.
    """
    c1, c2, c3 = st.columns(3)
    with c1:
        beta = st.number_input(
            "β — battery share of refined-Pb demand",
            min_value=0.30, max_value=0.99, step=0.01,
            value=st.session_state.get("v4_beta", preset.beta),
            key="v4_beta",
            help=f"Country preset: {preset.beta:.2f}. Applies the refined-feed branch "
                 "only; battery parts (HS 850790) bypass β.",
        )
    with c2:
        gamma = st.number_input(
            "γ — total collection rate",
            min_value=0.30, max_value=1.00, step=0.01,
            value=st.session_state.get("v4_gamma", preset.gamma),
            key="v4_gamma",
            help=f"Country preset: {preset.gamma:.2f}. Total collection (formal + informal); "
                 "the split happens downstream at break/smelt.",
        )
    with c3:
        k_stock = st.slider(
            "k_stock — stock multiplier (canonical, used in §3–§8)",
            min_value=0.40, max_value=1.50, step=0.01,
            value=float(st.session_state.get("v4_k_stock", 1.0)),
            key="v4_k_stock",
            help="Default 1.0 (reported stock). Slider scales the stock series; "
                 "every downstream section uses this value.",
        )
    return float(beta), float(gamma), float(k_stock)


def _render_tau_override(tau_seg: float) -> float:
    """Toggle + optional slider to override the segment-derived τ_eff.

    Default off → returns tau_seg. When on, returns the slider value.
    """
    cols = st.columns([2, 5])
    with cols[0]:
        override = st.toggle(
            "Override τ with slider",
            value=bool(st.session_state.get("v4_tau_override", False)),
            key="v4_tau_override",
            help=f"Off: use τ_eff = {tau_seg:.2f} yr (harmonic mean of segments). "
                 "On: use the slider value as the canonical τ everywhere.",
        )
    with cols[1]:
        if override:
            tau_used = float(st.slider(
                "τ override (yrs)",
                min_value=2.0, max_value=12.0, step=0.05,
                value=float(st.session_state.get("v4_tau_value", float(tau_seg))),
                key="v4_tau_value",
            ))
        else:
            tau_used = float(tau_seg)
            st.caption(f"Canonical τ = **{tau_seg:.2f} yr** (from segments). "
                       "Toggle the override to set τ from a slider.")
    return tau_used


def _render_stock_editor(df_csv: pd.DataFrame) -> pd.DataFrame:
    """Editable stock table; computes & displays g."""
    st.markdown("**Lead in service (total stock, t Pb)**")
    default_stock = df_csv[["year", "stock_total_t_pb"]].copy()
    default_stock["year"] = default_stock["year"].astype(int)
    edited = st.data_editor(
        default_stock,
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
        key="v4_stock_editor",
        column_config={
            "year": st.column_config.NumberColumn("Year", disabled=True, format="%d"),
            "stock_total_t_pb": st.column_config.NumberColumn(
                "Total Pb in service (t)", min_value=0.0, format="%.0f",
            ),
        },
    )
    edited = edited.dropna(subset=["stock_total_t_pb"]).reset_index(drop=True)
    if len(edited) < 3:
        st.error(
            f"Stock table needs at least 3 years (currently {len(edited)}). "
            "Restore values to enable the calculator."
        )
        return edited

    try:
        g_val = _compute_g_from_stock(edited)
    except Exception as e:
        st.error(f"Could not fit growth rate: {e}")
        return edited

    st.caption(
        f"Fitted **g = {g_val:.4f} / yr** (log-linear regression on the table). "
        "g is invariant to a constant level multiplier; editing values changes the "
        "growth rate. Computed read-only."
    )
    return edited


def _render_segments_editor(segments_df: pd.DataFrame) -> pd.DataFrame:
    """Editable segment table; computes & displays τ_eff."""
    st.markdown("**Battery segments → τ_eff (harmonic mean)**")
    default_segs = segments_df.copy()
    edited = st.data_editor(
        default_segs,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key="v4_segments_editor",
        column_config={
            "segment": st.column_config.TextColumn("Segment"),
            "stock_share": st.column_config.NumberColumn(
                "Stock share", min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
            ),
            "lifetime_years": st.column_config.NumberColumn(
                "Lifetime (yrs)", min_value=0.5, step=0.5, format="%.1f",
            ),
        },
    )
    edited = edited.dropna(subset=["stock_share", "lifetime_years"]).reset_index(drop=True)
    if edited.empty:
        st.error("Segment table is empty.")
        return edited

    share_sum = float(edited["stock_share"].sum())
    if not np.isclose(share_sum, 1.0, atol=1e-6):
        st.error(
            f"Segment stock_share must sum to 1.0 (currently {share_sum:.4f}). "
            "Adjust the rows above."
        )
        return edited

    try:
        tau_val = _compute_tau_eff(edited)
    except Exception as e:
        st.error(f"Could not compute τ_eff: {e}")
        return edited

    st.caption(
        f"**τ_eff = {tau_val:.2f} yr** (1 / Σ(w_s / τ_s)). Computed read-only here; "
        "the k/τ slider below initialises at this value but moves independently."
    )
    return edited


def _render_readonly_data_panel(df_csv: pd.DataFrame, fit_window: tuple[int, int]):
    """Auto-pulled USGS + BACI trade aggregates for the fit window."""
    st.markdown("**Auto-pulled data (read-only): USGS production and BACI trade aggregates**")
    df = df_csv[(df_csv["year"] >= fit_window[0]) & (df_csv["year"] <= fit_window[1])].copy()
    feed_imp = sum(df[f"imp_{hs}_t_pb"] for hs in FEED_HS)
    feed_exp = sum(df[f"exp_{hs}_t_pb"] for hs in FEED_HS)
    batt_imp = df["imp_850710_t_pb"] + df["imp_850720_t_pb"]
    batt_exp = df["exp_850710_t_pb"] + df["exp_850720_t_pb"]

    out = pd.DataFrame({
        "Year":              df["year"].astype(int),
        "USGS primary (t)":  df["primary_pb_t_usgs"].round(0).astype(int),
        "USGS secondary (t)": df["secondary_pb_t_usgs"].round(0).astype(int),
        "Net ore (t)":       (df["imp_260700_t_pb"] - df["exp_260700_t_pb"]).round(0).astype(int),
        "Net FEED refined (t)": (feed_imp - feed_exp).round(0).astype(int),
        "Net PARTS 850790 (t)": (df["imp_850790_t_pb"] - df["exp_850790_t_pb"]).round(0).astype(int),
        "Net CRUDE 780199 (t)": (df["imp_780199_t_pb"] - df["exp_780199_t_pb"]).round(0).astype(int),
        "Net USED 854810 (t)":  (df["imp_854810_t_pb"] - df["exp_854810_t_pb"]).round(0).astype(int),
        "Net SCRAP 780200 (t)": (df["imp_780200_t_pb"] - df["exp_780200_t_pb"]).round(0).astype(int),
        "Net BATT 8507x0 (t)":  (batt_imp - batt_exp).round(0).astype(int),
    })
    st.dataframe(out, hide_index=True, use_container_width=True)
    st.caption(
        f"Window: {fit_window[0]}–{fit_window[1]} (raw; the chain smooths these with a "
        "3-yr centered rolling mean before computing residuals)."
    )


def _render_efficiency_panel() -> dict:
    """Standalone efficiency panel — formal + informal η at every stage."""
    with st.expander(
        "Process efficiencies (η, δ) — formal + informal lanes at every stage",
        expanded=False,
    ):
        st.caption(
            "Formal η are literature defaults; informal η are placeholders the "
            "dashboard exposes for sensitivity. Both lanes carry through every "
            "stage; refining and manufacturing are no longer assumed fully formal."
        )
        delta = st.number_input(
            "δ — Pb remaining at end-of-life (shared)",
            0.50, 1.00, EFF_DEFAULTS["delta"], 0.01, key="v4_eff_delta",
        )
        cf, ci = st.columns(2)
        with cf:
            st.markdown("**Formal lane**")
            eta_break_F  = st.number_input("η_break_F — formal breaking",     0.50, 1.00, EFF_DEFAULTS["eta_break_F"],  0.01, key="v4_eff_break_F")
            eta_smelt_F  = st.number_input("η_smelt_F — formal smelting",     0.50, 1.00, EFF_DEFAULTS["eta_smelt_F"],  0.01, key="v4_eff_smelt_F")
            eta_refine_F = st.number_input("η_refine_F — formal refining",    0.80, 1.00, EFF_DEFAULTS["eta_refine_F"], 0.01, key="v4_eff_refine_F")
            eta_mfg_F    = st.number_input("η_mfg_F — formal manufacturing",  0.80, 1.00, EFF_DEFAULTS["eta_mfg_F"],    0.01, key="v4_eff_mfg_F")
        with ci:
            st.markdown("**Informal lane**")
            eta_break_I  = st.number_input("η_break_I — informal breaking",   0.30, 1.00, EFF_DEFAULTS["eta_break_I"],  0.01, key="v4_eff_break_I")
            eta_smelt_I  = st.number_input("η_smelt_I — informal smelting",   0.30, 1.00, EFF_DEFAULTS["eta_smelt_I"],  0.01, key="v4_eff_smelt_I")
            eta_refine_I = st.number_input("η_refine_I — informal refining",  0.50, 1.00, EFF_DEFAULTS["eta_refine_I"], 0.01, key="v4_eff_refine_I",
                                           help="Capped below formal (0.99) by design. Default 0.95.")
            eta_mfg_I    = st.number_input("η_mfg_I — informal manufacturing", 0.50, 1.00, EFF_DEFAULTS["eta_mfg_I"],    0.01, key="v4_eff_mfg_I",
                                           help="Capped below formal (0.98) by design. Default 0.95.")
    return {
        "delta":        float(delta),
        "eta_break_F":  float(eta_break_F),
        "eta_break_I":  float(eta_break_I),
        "eta_smelt_F":  float(eta_smelt_F),
        "eta_smelt_I":  float(eta_smelt_I),
        "eta_refine_F": float(eta_refine_F),
        "eta_refine_I": float(eta_refine_I),
        "eta_mfg_F":    float(eta_mfg_F),
        "eta_mfg_I":    float(eta_mfg_I),
    }


def _render_phi_inputs() -> dict:
    """Editor for the four formal-share parameters.

    USAID-informed floors per stage: φ_smelt ≥ 0.70, φ_refine ≥ 0.80, φ_mfg ≥ 0.90.
    Ordering (break < smelt < refine < mfg < 1) is enforced as a warning,
    not a hard constraint.
    """
    with st.expander(
        "Formal shares φ — USAID floors per stage (break < smelt < refine < mfg)",
        expanded=False,
    ):
        st.caption(
            "Starting values + per-stage lower bounds reflect USAID's view that "
            "formal share is high and rises down the chain. Floors: "
            f"φ_smelt ≥ {PHI_FLOORS['phi_smelt_f']:.2f}, "
            f"φ_refine ≥ {PHI_FLOORS['phi_refine_f']:.2f}, "
            f"φ_mfg ≥ {PHI_FLOORS['phi_mfg_f']:.2f}. "
            "Ordering is enforced as a warning; implied crossovers in §3 flag "
            "where it is locally infeasible."
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            phi_b = st.number_input("φ_break_f",
                PHI_FLOORS["phi_break_f"], 0.99, REF_PHI["phi_break_f"], 0.01,
                key="v4_phi_break")
        with c2:
            phi_s = st.number_input("φ_smelt_f",
                PHI_FLOORS["phi_smelt_f"], 0.99, REF_PHI["phi_smelt_f"], 0.01,
                key="v4_phi_smelt",
                help=f"USAID floor: ≥ {PHI_FLOORS['phi_smelt_f']:.2f}")
        with c3:
            phi_r = st.number_input("φ_refine_f",
                PHI_FLOORS["phi_refine_f"], 0.99, REF_PHI["phi_refine_f"], 0.01,
                key="v4_phi_refine",
                help=f"USAID floor: ≥ {PHI_FLOORS['phi_refine_f']:.2f}")
        with c4:
            phi_m = st.number_input("φ_mfg_f",
                PHI_FLOORS["phi_mfg_f"], 0.99, REF_PHI["phi_mfg_f"], 0.01,
                key="v4_phi_mfg",
                help=f"USAID floor: ≥ {PHI_FLOORS['phi_mfg_f']:.2f}")
        phi = {
            "phi_break_f":  float(phi_b),
            "phi_smelt_f":  float(phi_s),
            "phi_refine_f": float(phi_r),
            "phi_mfg_f":    float(phi_m),
        }
        if not phi_is_ordered(phi):
            st.warning(
                f"Ordering violated: need φ_break ({phi_b:.2f}) < φ_smelt ({phi_s:.2f}) "
                f"< φ_refine ({phi_r:.2f}) < φ_mfg ({phi_m:.2f}) < 1. "
                "Implied crossovers in §3 will flag where this hurts."
            )
    return phi


# ---------------------------------------------------------------------------
# k/tau ridge and phi curves
# ---------------------------------------------------------------------------

def _ridge_curves(arr: dict, g: float, gamma: float, beta: float,
                  etas: dict, tau_grid: np.ndarray) -> pd.DataFrame:
    """For each tau, compute two structural k-thresholds:
      - k_install : the k at which INSTALL_target = INSTALL_implied(all-formal)
                    — above this, even the chain at max formal recovery
                    cannot supply the stock-derived install demand.
      - k_refine  : the smallest k at which REFINE_SEC_F(all-formal) ≥ USGS_sec
                    — below this, the chain undershoots the USGS floor.

    The two curves often disjoint: install drops, refine rises, with τ.
    Where they cross, both anchors can be satisfied simultaneously.
    """
    rows = []
    for tau in tau_grid:
        k_inst = _k_install_feasibility_crossing(arr, g, float(tau), beta, gamma, etas)
        k_refn = _k_refine_floor_crossing(arr, g, float(tau), beta, gamma, etas)
        rows.append({"tau": float(tau), "k_install": k_inst, "k_refine": k_refn})
    return pd.DataFrame(rows)


# The four closed-form φ↔η_I solvers (one per stage) live further down with the
# §6 renderer. They replace the legacy v4 break/smelt-only helpers, and the
# brief tour through residual-sweeps that was here in the intermediate revision.


# ---------------------------------------------------------------------------
# Feasibility-region map (Phase F)
# ---------------------------------------------------------------------------
#
# Two physical walls mapped over a (k, tau) grid. No optimizer; just forward
# evaluation of the max-recovery chain and the Pass-2 install ceiling, then
# inequality tests against USGS-observed magnitudes.
#
#   refine wall :  REFINE_SEC_max(k, tau)  >=  USGS_secondary
#                  (max-recovery chain: phi=1 at break & smelt, formal etas)
#   install wall:  INSTALL_target(k, tau)  <=  INSTALL_ceiling
#                  (USGS+trade max suppliable; k-independent ceiling)
#
# Both are enforced "all years strict" -- a cell is feasible only if the
# inequality holds in every fit-window year.

# Plausible-tau band (yrs) -- physical lifetime constraint, not a model output.
PLAUSIBLE_TAU_DEFAULT = (4.0, 6.0)

# Pinned k/tau reference ratio (labelled line on the map).
PINNED_K_OVER_TAU = 0.11

# Map grid resolution.
FEAS_K_GRID  = np.round(np.arange(0.10, 1.30 + 1e-9, 0.02), 4)
FEAS_TAU_GRID = np.round(np.arange(1.0, 12.0 + 1e-9, 0.25), 4)


def _max_formal_chain_at(arr: dict, k: float, tau: float,
                         g: float, gamma: float, beta: float,
                         etas: dict) -> dict:
    """Parallel chain at φ → all-formal for one (k, τ)."""
    return forward_parallel_chain(
        arr, k_stock=k, phi=PHI_MAX_FORMAL,
        g=g, tau=tau, beta=beta, gamma=gamma,
        **{k_: v for k_, v in etas.items() if k_ in V5_ETA_DEFAULTS},
    )


def _feasibility_grid(arr: dict, g: float, gamma: float, beta: float,
                      etas: dict,
                      k_grid: np.ndarray = FEAS_K_GRID,
                      tau_grid: np.ndarray = FEAS_TAU_GRID) -> dict:
    """Sweep (k, τ); return state array + per-τ k-thresholds.

    State coding:
      0 = infeasible-both (install ceiling exceeded AND refine floor undercut)
      1 = install-only feasible (install_target ≤ install_implied(all-formal))
      2 = refine-only feasible (REFINE_SEC_F(all-formal) ≥ USGS)
      3 = feasible-both

    All-year strict on each side. Uses parallel chain at max-formal.
    """
    n_k, n_tau = len(k_grid), len(tau_grid)
    state = np.zeros((n_k, n_tau), dtype=np.int8)
    k_min_refine = np.full(n_tau, np.nan, dtype=float)
    k_max_install = np.full(n_tau, np.nan, dtype=float)
    sec_usgs = arr["sec_usgs"]

    for j, tau in enumerate(tau_grid):
        # Compute chain at all k's for this tau
        REFINE_max = np.empty((n_k, len(arr["year"])))
        INSTALL_max = np.empty((n_k, len(arr["year"])))
        INSTALL_target = np.empty((n_k, len(arr["year"])))
        for i, k in enumerate(k_grid):
            out = _max_formal_chain_at(arr, float(k), float(tau),
                                       g, gamma, beta, etas)
            REFINE_max[i, :]    = out["REFINE_SEC_F"]
            INSTALL_max[i, :]   = out["INSTALL_implied"]
            INSTALL_target[i, :] = out["INSTALL_target"]

        refine_ok  = (REFINE_max  >= sec_usgs[None, :]).all(axis=1)
        install_ok = (INSTALL_target <= INSTALL_max).all(axis=1)

        state[install_ok & ~refine_ok, j] = 1
        state[~install_ok & refine_ok, j] = 2
        state[install_ok & refine_ok,  j] = 3

        if refine_ok.any():
            k_min_refine[j] = float(k_grid[np.argmax(refine_ok)])
        if install_ok.any():
            idxs = np.where(install_ok)[0]
            k_max_install[j] = float(k_grid[idxs[-1]])

    return {
        "state":          state,
        "k_grid":         k_grid,
        "tau_grid":       tau_grid,
        "k_min_refine":   k_min_refine,
        "k_max_install":  k_max_install,
    }


def _feasibility_closure_tau(grid: dict) -> float | None:
    """Largest tau in the grid that has ANY feasible-both cell.

    Above this tau the install ceiling and the refine wall cross; no (k, tau)
    is feasible. None if feasible-both exists at the largest tau in the grid
    (no closure inside the grid).
    """
    state = grid["state"]
    tau_grid = grid["tau_grid"]
    has_both = (state == 3).any(axis=0)
    if not has_both.any():
        return float("nan")
    if has_both.all():
        return None
    last_idx = int(np.max(np.where(has_both)[0]))
    return float(tau_grid[last_idx])


def _feasible_k_range_in_band(grid: dict, tau_low: float, tau_high: float) -> dict:
    """k-range of the feasible-both region intersected with [tau_low, tau_high].

    Returns {'k_min','k_max','tau_min','tau_max','n_cells'}; values are NaN if
    no feasible-both cell lies in the band.
    """
    state = grid["state"]
    k_grid, tau_grid = grid["k_grid"], grid["tau_grid"]
    KK, TT = np.meshgrid(k_grid, tau_grid, indexing="ij")
    mask = (state == 3) & (TT >= tau_low) & (TT <= tau_high)
    n = int(mask.sum())
    if n == 0:
        return {"k_min": float("nan"), "k_max": float("nan"),
                "tau_min": float("nan"), "tau_max": float("nan"),
                "n_cells": 0}
    return {
        "k_min":   float(KK[mask].min()),
        "k_max":   float(KK[mask].max()),
        "tau_min": float(TT[mask].min()),
        "tau_max": float(TT[mask].max()),
        "n_cells": n,
    }


def _feasibility_grid_mc(arr: dict, g: float, gamma: float, beta: float,
                         etas: dict, k_grid: np.ndarray, tau_grid: np.ndarray,
                         n_draws: int, seed: int = 17) -> dict:
    """Parametric MC on the feasibility boundary.

    Per draw, perturb the formal etas (delta, eta_break_F, eta_smelt_F,
    eta_refine, eta_mfg) ±5%, beta ±5%, and the 11 HS conversion factors ±5%.
    USGS secondary is unperturbed (it is the right-hand side of the wall).

    Returns:
      'k_min_refine_p5', 'k_min_refine_p95' arrays over tau_grid
      'k_max_install_p5','k_max_install_p95' arrays over tau_grid
    """
    rng = np.random.default_rng(seed)
    eff_keys_formal = ("delta", "eta_break_F", "eta_smelt_F",
                       "eta_refine_F", "eta_mfg_F")
    eff_keys_other  = tuple(k for k in EFF_DEFAULTS if k not in eff_keys_formal)

    n_tau = len(tau_grid)
    refine_mat = np.full((n_draws, n_tau), np.nan)
    install_mat = np.full((n_draws, n_tau), np.nan)

    for d in range(n_draws):
        etas_d = {}
        for kname, kval in etas.items():
            if kname in eff_keys_formal or kname in eff_keys_other:
                etas_d[kname] = float(kval * (1.0 + rng.uniform(-EFF_PERTURB, EFF_PERTURB)))
        factor_pert = {hs: float(default * (1.0 + rng.uniform(-HS_FACTOR_PERTURB,
                                                              HS_FACTOR_PERTURB)))
                       for hs, default in HS_FACTOR_DEFAULTS.items()}
        beta_d = float(beta * (1.0 + rng.uniform(-BETA_PERTURB, BETA_PERTURB)))

        # arr aggregates the per-HS columns (e.g. imp_feed = sum over FEED_HS).
        # Perturbing per-HS factors and then re-aggregating would require the
        # per-HS columns, which are not in arr. Approximation: apply an
        # average HS-factor scale to each aggregate. Bounded by the same ±5%
        # envelope; loses some cross-correlation between HS codes.
        def avg_scale(hs_list):
            return float(np.mean([factor_pert[hs] / HS_FACTOR_DEFAULTS[hs]
                                  for hs in hs_list]))
        single_hs_scale = {
            "imp_ore":   factor_pert[260700] / HS_FACTOR_DEFAULTS[260700],
            "exp_ore":   factor_pert[260700] / HS_FACTOR_DEFAULTS[260700],
            "imp_parts": factor_pert[850790] / HS_FACTOR_DEFAULTS[850790],
            "exp_parts": factor_pert[850790] / HS_FACTOR_DEFAULTS[850790],
            "imp_crude": factor_pert[780199] / HS_FACTOR_DEFAULTS[780199],
            "exp_crude": factor_pert[780199] / HS_FACTOR_DEFAULTS[780199],
            "imp_used":  factor_pert[854810] / HS_FACTOR_DEFAULTS[854810],
            "exp_used":  factor_pert[854810] / HS_FACTOR_DEFAULTS[854810],
            "imp_scrap": factor_pert[780200] / HS_FACTOR_DEFAULTS[780200],
            "exp_scrap": factor_pert[780200] / HS_FACTOR_DEFAULTS[780200],
        }
        s_feed = avg_scale(FEED_HS)
        s_batt = avg_scale([850710, 850720])
        arr_d = dict(arr)
        arr_d["imp_feed"] = arr["imp_feed"] * s_feed
        arr_d["exp_feed"] = arr["exp_feed"] * s_feed
        arr_d["imp_batt"] = arr["imp_batt"] * s_batt
        arr_d["exp_batt"] = arr["exp_batt"] * s_batt
        for key, scale in single_hs_scale.items():
            arr_d[key] = arr[key] * scale

        grid_d = _feasibility_grid(arr_d, g, gamma, beta_d, etas_d,
                                   k_grid=k_grid, tau_grid=tau_grid)
        refine_mat[d, :]  = grid_d["k_min_refine"]
        install_mat[d, :] = grid_d["k_max_install"]

    def pct(mat, q):
        out = np.full(n_tau, np.nan)
        for j in range(n_tau):
            v = mat[:, j]
            v = v[~np.isnan(v)]
            if v.size:
                out[j] = float(np.percentile(v, q))
        return out

    return {
        "k_min_refine_p5":  pct(refine_mat, 5),
        "k_min_refine_p95": pct(refine_mat, 95),
        "k_max_install_p5":  pct(install_mat, 5),
        "k_max_install_p95": pct(install_mat, 95),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_ridge(ridge_df: pd.DataFrame, tau_pick: float, k_pick: float,
                k_feas_crossing: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ridge_df["tau"], y=ridge_df["k_install"],
        mode="lines", name="install ceiling — max k (max-formal supply ≥ INSTALL_target)",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="τ=%{x:.2f} yr<br>k_install_ceil=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ridge_df["tau"], y=ridge_df["k_refine"],
        mode="lines", name="refine floor — min k (REFINE_SEC_F(max-formal) ≥ USGS_sec)",
        line=dict(color="#d62728", width=2, dash="dash"),
        hovertemplate="τ=%{x:.2f} yr<br>k_refine_floor=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[tau_pick], y=[k_pick],
        mode="markers", name=f"selected (τ={tau_pick:.2f}, k={k_pick:.3f})",
        marker=dict(color="black", size=12, symbol="x"),
    ))
    fig.add_hline(
        y=k_feas_crossing, line_dash="dot", line_color="#1f77b4", opacity=0.5,
        annotation_text=f"install ceiling at slider τ  k≈{k_feas_crossing:.2f}",
        annotation_position="top left",
    )
    fig.update_layout(
        height=380, margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="τ (effective battery lifetime, yrs)",
        yaxis_title="k_stock (stock multiplier)",
        legend=dict(orientation="h", y=-0.20, x=0.0),
        title="k/τ ridge — data-preferred k as a function of τ",
    )
    return fig


# _plot_phi_eta_curve (further down) is the §6 plotter — closed-form curves of
# implied φ vs η_I, four panels (one per stage).


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

HS_FACTOR_DEFAULTS = {
    260700: 0.60, 780110: 1.00, 780191: 0.95, 780199: 0.95,
    780200: 0.97, 850710: 0.65, 850720: 0.70, 850790: 0.80,
    854810: 0.70, 282410: 0.91, 282490: 0.75,
}

MC_DRAWS = 200
EFF_PERTURB = 0.05
HS_FACTOR_PERTURB = 0.05
BETA_PERTURB = 0.05
USGS_PRIMARY_PERTURB = 0.20   # high end of the spec's 15-20%


def _scale_hs_columns(df_csv: pd.DataFrame, factor_pert: dict,
                      user_stock_lookup: dict) -> pd.DataFrame:
    """Return df_csv with each HS column scaled by factor_pert[hs] /
    HS_FACTOR_DEFAULTS[hs] and stock_total_t_pb replaced by user_stock_lookup."""
    df = df_csv.copy()
    for hs in HS_FACTOR_DEFAULTS:
        for prefix in ("imp", "exp"):
            col = f"{prefix}_{hs}_t_pb"
            if col in df.columns:
                df[col] = df[col] * factor_pert[hs] / HS_FACTOR_DEFAULTS[hs]
    if user_stock_lookup:
        df["stock_total_t_pb"] = df["year"].astype(int).map(
            lambda y: user_stock_lookup.get(int(y), df_csv.loc[df_csv["year"] == y, "stock_total_t_pb"].iloc[0])
        )
    return df


def _run_monte_carlo(df_csv: pd.DataFrame, user_stock: pd.DataFrame,
                     fit_window: tuple[int, int], beta: float, gamma: float,
                     phi: dict, etas: dict, k_stock: float, tau: float,
                     n_draws: int = MC_DRAWS, seed: int = 42) -> dict:
    """Run parametric MC on the parallel chain at the user's committed inputs
    and chosen (k, τ).

    Per-draw perturbations:
      - all efficiencies ±5% uniform (formal + informal, all 8 + δ)
      - 11 HS conversion factors ±5% uniform
      - β ±5% uniform around the user's β
      - USGS primary ±20% uniform
      - USGS secondary UNPERTURBED (it is the refine-side floor target)

    Tracks REFINE_SEC_F + MFG_F + MFG_total bands and the install / refine
    residuals; also reports how many draws SURVIVE the (|res_install|≤5% AND
    shortfall_refine≤5%) joint criterion at the user's (k, τ, φ).
    """
    user_stock = user_stock.copy()
    user_stock["year"] = user_stock["year"].astype(int)
    user_stock_lookup = dict(zip(user_stock["year"].tolist(),
                                 user_stock["stock_total_t_pb"].astype(float).tolist()))

    g_val = _compute_g_from_stock(user_stock)

    rng = np.random.default_rng(seed)
    refine_F_arr, mfg_F_arr, mfg_tot_arr = [], [], []
    res_install_W_vals, shortfall_refine_W_vals, overshoot_refine_W_vals = [], [], []
    survivor_count = 0

    for _ in range(n_draws):
        etas_d = {k_: float(v * (1.0 + rng.uniform(-EFF_PERTURB, EFF_PERTURB)))
                  for k_, v in etas.items() if k_ in V5_ETA_DEFAULTS}
        factor_pert = {hs: float(default * (1.0 + rng.uniform(-HS_FACTOR_PERTURB, HS_FACTOR_PERTURB)))
                       for hs, default in HS_FACTOR_DEFAULTS.items()}
        beta_d = float(beta * (1.0 + rng.uniform(-BETA_PERTURB, BETA_PERTURB)))
        primary_scale = float(rng.uniform(1.0 - USGS_PRIMARY_PERTURB,
                                          1.0 + USGS_PRIMARY_PERTURB))

        df_pert = _scale_hs_columns(df_csv, factor_pert, user_stock_lookup)
        df_pert["primary_pb_t_usgs"] = df_pert["primary_pb_t_usgs"] * primary_scale

        try:
            arr_d = _build_arr_from_user_inputs(df_pert, user_stock, fit_window)
        except Exception:
            continue
        out_d = forward_parallel_chain(
            arr_d, k_stock=k_stock, phi=phi,
            beta=beta_d, gamma=gamma, g=g_val, tau=tau,
            **etas_d,
        )
        refine_F_arr.append(out_d["REFINE_SEC_F"])
        mfg_F_arr.append(out_d["MFG_F"])
        mfg_tot_arr.append(out_d["MFG_total"])
        res_install_W_vals.append(out_d["res_install_W"])
        shortfall_refine_W_vals.append(out_d["shortfall_refine_W"])
        overshoot_refine_W_vals.append(out_d["overshoot_refine_W"])
        if abs(out_d["res_install_W"]) <= 0.05 and out_d["shortfall_refine_W"] <= 0.05:
            survivor_count += 1

    if not refine_F_arr:
        return {"ok": False}

    refine_F_mat = np.vstack(refine_F_arr)
    mfg_F_mat    = np.vstack(mfg_F_arr)
    mfg_tot_mat  = np.vstack(mfg_tot_arr)
    return {
        "ok":               True,
        "n_draws":          len(refine_F_arr),
        "n_survivors":      survivor_count,
        "refine_F_p5":      np.percentile(refine_F_mat,  5, axis=0),
        "refine_F_med":     np.percentile(refine_F_mat, 50, axis=0),
        "refine_F_p95":     np.percentile(refine_F_mat, 95, axis=0),
        "mfg_F_p5":         np.percentile(mfg_F_mat,  5, axis=0),
        "mfg_F_med":        np.percentile(mfg_F_mat, 50, axis=0),
        "mfg_F_p95":        np.percentile(mfg_F_mat, 95, axis=0),
        "mfg_tot_p5":       np.percentile(mfg_tot_mat,  5, axis=0),
        "mfg_tot_med":      np.percentile(mfg_tot_mat, 50, axis=0),
        "mfg_tot_p95":      np.percentile(mfg_tot_mat, 95, axis=0),
        "res_install_W":    (float(np.percentile(res_install_W_vals, 5)),
                             float(np.percentile(res_install_W_vals, 50)),
                             float(np.percentile(res_install_W_vals, 95))),
        "shortfall_refine_W": (float(np.percentile(shortfall_refine_W_vals, 5)),
                               float(np.percentile(shortfall_refine_W_vals, 50)),
                               float(np.percentile(shortfall_refine_W_vals, 95))),
        "overshoot_refine_W": (float(np.percentile(overshoot_refine_W_vals, 5)),
                               float(np.percentile(overshoot_refine_W_vals, 50)),
                               float(np.percentile(overshoot_refine_W_vals, 95))),
    }


def _mc_cache_key(beta: float, gamma: float, user_stock: pd.DataFrame,
                  segments_df: pd.DataFrame, etas: dict, phi: dict,
                  k_stock: float, tau: float, fit_window: tuple[int, int]) -> str:
    """Stable hash of the inputs that drive MC, for caching in session state."""
    payload = {
        "beta": round(beta, 6), "gamma": round(gamma, 6),
        "stock": user_stock.round(2).to_dict(orient="records"),
        "segs":  segments_df.round(4).to_dict(orient="records"),
        "etas":  {k: round(v, 6) for k, v in etas.items()},
        "phi":   {k: round(v, 6) for k, v in phi.items()},
        "k_stock": round(k_stock, 6), "tau": round(tau, 6),
        "fit_window": list(fit_window),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Render output sections
# ---------------------------------------------------------------------------

def _render_calculation_walkthrough(out_point: dict, arr: dict,
                                    g: float, tau: float,
                                    beta: float, gamma: float, etas: dict,
                                    phi: dict, k_used: float) -> None:
    """Step-by-step view of the parallel chain between inputs (§1) and residuals (§4).

    Each expander shows the equation, both lanes (F + I) where applicable, and
    implied informal→formal crossovers at stage boundaries.
    """
    years = arr["year"].astype(int)
    delta = etas["delta"]
    eta_break_F  = etas["eta_break_F"];  eta_break_I  = etas["eta_break_I"]
    eta_smelt_F  = etas["eta_smelt_F"];  eta_smelt_I  = etas["eta_smelt_I"]
    eta_refine_F = etas["eta_refine_F"]; eta_refine_I = etas["eta_refine_I"]
    eta_mfg_F    = etas["eta_mfg_F"];    eta_mfg_I    = etas["eta_mfg_I"]

    def _row(label, values):
        return {"Quantity": label,
                **{int(y): _fmt2sig(float(v)) for y, v in zip(years, values)}}

    def _show(rows):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.caption(
        f"All quantities in t Pb, 2 sig figs. Parameters: β = {beta:.2f}, "
        f"γ = {gamma:.2f}, τ = {tau:.2f} yr, g = {g:.4f}, k = {k_used:.2f}. "
        f"φ = ({phi['phi_break_f']:.2f}, {phi['phi_smelt_f']:.2f}, "
        f"{phi['phi_refine_f']:.2f}, {phi['phi_mfg_f']:.2f}) — "
        f"formal share at break/smelt/refine/mfg."
    )

    # ---------------- Step 1: stock growth rate g ----------------
    with st.expander("Step 1 — Stock growth rate g (log-linear fit on stock)"):
        st.latex(r"g = \mathrm{slope}\bigl(\ln(\mathrm{stock}_t)\ \mathrm{vs.}\ t\bigr)")
        stock = arr["stock"]
        _show([
            _row("stock_total (t Pb)", stock),
            _row("ln(stock)",          np.log(stock)),
        ])
        st.markdown(f"**g = {g:.4f} /yr**  (≈ {g*100:.2f}%/yr)")

    # ---------------- Step 2: τ_eff harmonic mean ----------------
    with st.expander("Step 2 — Effective lifetime τ_eff (harmonic mean of segments)"):
        st.latex(r"\tau_{\mathrm{eff}} = \frac{1}{\sum_s w_s / \tau_s}")
        st.caption("Stock-share-weighted harmonic mean. Pulled toward the "
                   "shortest-lived segment (heavier weight at retirement).")
        st.markdown(f"**τ_eff = {tau:.4f} yr**")

    # ---------------- Step 3: retirement rate and RETIRE ----------------
    with st.expander("Step 3 — Retirement rate r and RETIRE(t)"):
        st.latex(r"r = \frac{g}{e^{g\tau}-1}, \qquad "
                 r"\mathrm{RETIRE}(t) = k\cdot \mathrm{stock}(t) \cdot r")
        r = out_point["retire_rate"]
        eff_stock = out_point["eff_stock"]
        RETIRE = out_point["RETIRE"]
        st.markdown(f"**r = {r:.4f}/yr**  (i.e. ~{r*100:.1f}% of stock retires per year)")
        _show([
            _row("k · stock(t)", eff_stock),
            _row("RETIRE(t) = k·stock · r", RETIRE),
        ])

    # ---------------- Step 4: COLLECT (shared, pre-split) -------------
    with st.expander("Step 4 — COLLECT (single shared pool, before the lanes split)"):
        st.latex(r"\mathrm{COLLECT} = \gamma\cdot\mathrm{RETIRE}")
        st.caption(
            "Under the parallel chain the formal/informal split is at BREAK "
            "(Step 5), not here. COLLECT is the single shared pool. Used-battery "
            "trade (854810) is FORMAL-only and attaches at the formal break "
            "input — see Step 5."
        )
        _show([
            _row("RETIRE",                    out_point["RETIRE"]),
            _row("γ · RETIRE = COLLECT",      out_point["COLLECT"]),
        ])

    # ---------------- Step 5: BREAK (parallel split) ------------------
    with st.expander("Step 5 — BREAK (parallel formal + informal lanes, split at φ_break_f)"):
        st.latex(
            r"\mathrm{in\_break}_F = \mathrm{COLLECT}\cdot\phi^{\mathrm{break}}_f"
            r" + \mathrm{imp\_used} - \mathrm{exp\_used}"
        )
        st.latex(
            r"\mathrm{in\_break}_I = \mathrm{COLLECT}\cdot(1-\phi^{\mathrm{break}}_f)"
        )
        st.latex(
            r"\mathrm{out\_break}_F = \mathrm{in\_break}_F\cdot\delta\cdot\eta^{\mathrm{break}}_F, \qquad"
            r"\mathrm{out\_break}_I = \mathrm{in\_break}_I\cdot\delta\cdot\eta^{\mathrm{break}}_I"
        )
        st.caption(
            f"δ = {delta:.2f}, η_break_F = {eta_break_F:.2f}, "
            f"η_break_I = {eta_break_I:.2f}. φ_break_f = {phi['phi_break_f']:.2f}. "
            "Used-battery trade (854810) is formal-only and attaches at the "
            "formal break input (imports flow through legal customs into formal breakers)."
        )
        _show([
            _row("COLLECT (shared pool)",       out_point["COLLECT"]),
            _row("imp_used (854810)",           arr["imp_used"]),
            _row("exp_used (854810)",           arr["exp_used"]),
            _row("in_break_F",                  out_point["in_break_F"]),
            _row("in_break_I",                  out_point["in_break_I"]),
            _row("out_break_F (formal)",        out_point["out_break_F"]),
            _row("out_break_I (informal)",      out_point["out_break_I"]),
            _row("BREAK_total = sum",           out_point["BREAK_total"]),
        ])

    # ---------------- Step 6: SCRAP supply + SMELT (parallel) --------
    with st.expander("Step 6 — SMELT (formal-only scrap trade; informal share = 1 − φ_smelt_f)"):
        st.latex(
            r"\mathrm{scrap\_total} = \mathrm{out\_break}_F + \mathrm{out\_break}_I"
            r" + \mathrm{imp\_scrap} - \mathrm{exp\_scrap}"
        )
        st.latex(
            r"\mathrm{in\_smelt}_F = \mathrm{scrap\_total}\cdot\phi^{\mathrm{smelt}}_f, \qquad"
            r"\mathrm{in\_smelt}_I = \mathrm{scrap\_total}\cdot(1-\phi^{\mathrm{smelt}}_f)"
        )
        st.latex(
            r"\mathrm{out\_smelt}_F = \mathrm{in\_smelt}_F\cdot\eta^{\mathrm{smelt}}_F, \qquad"
            r"\mathrm{out\_smelt}_I = \mathrm{in\_smelt}_I\cdot\eta^{\mathrm{smelt}}_I"
        )
        st.caption(
            f"η_smelt_F = {eta_smelt_F:.2f}, η_smelt_I = {eta_smelt_I:.2f}, "
            f"φ_smelt_f = {phi['phi_smelt_f']:.2f}. Scrap trade attaches to the "
            "formal lane only (informal lane is purely domestic). Implied "
            "informal→formal crossover at this boundary: in_smelt_F − out_break_F."
        )
        _show([
            _row("out_break_F",                 out_point["out_break_F"]),
            _row("out_break_I",                 out_point["out_break_I"]),
            _row("imp_scrap (780200, formal)",  arr["imp_scrap"]),
            _row("exp_scrap (780200, formal)",  arr["exp_scrap"]),
            _row("scrap_total",                 out_point["scrap_total"]),
            _row("in_smelt_F",                  out_point["in_smelt_F"]),
            _row("in_smelt_I",                  out_point["in_smelt_I"]),
            _row("out_smelt_F (formal)",        out_point["out_smelt_F"]),
            _row("out_smelt_I (informal)",      out_point["out_smelt_I"]),
            _row("⟹ xover_smelt = in_smelt_F − out_break_F",
                 out_point["xover_smelt"]),
        ])

    # ---------------- Step 7: REFINE (parallel) ----------------------
    with st.expander("Step 7 — REFINE (parallel; USGS anchors FORMAL refined only)"):
        st.latex(
            r"\mathrm{crude\_total} = \mathrm{out\_smelt}_F + \mathrm{out\_smelt}_I"
            r" + \mathrm{imp\_crude}_{780199} - \mathrm{exp\_crude}_{780199}"
        )
        st.latex(
            r"\mathrm{in\_refine}_F = \mathrm{crude\_total}\cdot\phi^{\mathrm{refine}}_f, \qquad"
            r"\mathrm{in\_refine}_I = \mathrm{crude\_total}\cdot(1-\phi^{\mathrm{refine}}_f)"
        )
        st.latex(
            r"\mathrm{REFINE\_SEC}_F = \mathrm{in\_refine}_F\cdot\eta^{\mathrm{refine}}_F, \qquad"
            r"\mathrm{REFINE\_SEC}_I = \mathrm{in\_refine}_I\cdot\eta^{\mathrm{refine}}_I"
        )
        st.caption(
            f"η_refine_F = {eta_refine_F:.2f}, η_refine_I = {eta_refine_I:.2f}, "
            f"φ_refine_f = {phi['phi_refine_f']:.2f}. 780199 (crude) trade is "
            "formal-only. **USGS secondary now anchors REFINE_SEC_F only**, as a "
            "one-sided floor: REFINE_SEC_F ≥ USGS_sec is required; overshoot is "
            "expected (it represents informal/unrecorded formal-equivalent refined "
            "lead). Crossover at this boundary: in_refine_F − out_smelt_F."
        )
        _show([
            _row("out_smelt_F",                 out_point["out_smelt_F"]),
            _row("out_smelt_I",                 out_point["out_smelt_I"]),
            _row("imp_crude (780199, formal)",  arr["imp_crude"]),
            _row("exp_crude (780199, formal)",  arr["exp_crude"]),
            _row("crude_total",                 out_point["crude_total"]),
            _row("in_refine_F",                 out_point["in_refine_F"]),
            _row("in_refine_I",                 out_point["in_refine_I"]),
            _row("REFINE_SEC_F (formal)",       out_point["REFINE_SEC_F"]),
            _row("REFINE_SEC_I (informal)",     out_point["REFINE_SEC_I"]),
            _row("USGS secondary (anchor on F)",arr["sec_usgs"]),
            _row("⟹ xover_refine = in_refine_F − out_smelt_F",
                 out_point["xover_refine"]),
        ])

    # ---------------- Step 8: refined POOL + MFG (parallel) ----------
    with st.expander("Step 8 — REFINED pool, β branch, and parallel manufacturing"):
        st.latex(
            r"\mathrm{refined\_total} = \mathrm{REFINE\_SEC}_F + \mathrm{REFINE\_SEC}_I"
            r" + \mathrm{REFINE\_PRIMARY}_{\mathrm{USGS}} + \mathrm{imp\_feed} - \mathrm{exp\_feed}"
        )
        st.latex(
            r"\mathrm{in\_mfg}_F = \mathrm{refined\_total}\cdot\phi^{\mathrm{mfg}}_f, \qquad"
            r"\mathrm{in\_mfg}_I = \mathrm{refined\_total}\cdot(1-\phi^{\mathrm{mfg}}_f)"
        )
        st.latex(
            r"\mathrm{MFG}_F = \mathrm{in\_mfg}_F\cdot\beta\cdot\eta^{\mathrm{mfg}}_F"
            r"\ +\ \max(0,\mathrm{NET\_PARTS})\cdot\eta^{\mathrm{mfg}}_F"
        )
        st.latex(
            r"\mathrm{MFG}_I = \mathrm{in\_mfg}_I\cdot\beta\cdot\eta^{\mathrm{mfg}}_I"
        )
        st.caption(
            f"β = {beta:.2f}, η_mfg_F = {eta_mfg_F:.2f}, η_mfg_I = {eta_mfg_I:.2f}, "
            f"φ_mfg_f = {phi['phi_mfg_f']:.2f}. Primary lead and FEED trade enter the "
            "formal refined pool. Battery parts (HS 850790) are battery-committed: "
            "they bypass β and route to FORMAL mfg with η_mfg_F. Oxides "
            "(282410/282490) stay in FEED and remain subject to β."
        )
        _show([
            _row("REFINE_SEC_F",                out_point["REFINE_SEC_F"]),
            _row("REFINE_SEC_I",                out_point["REFINE_SEC_I"]),
            _row("REFINE_PRIMARY (USGS)",       out_point["REFINE_PRIMARY"]),
            _row("imp_feed",                    arr["imp_feed"]),
            _row("exp_feed",                    arr["exp_feed"]),
            _row("refined_total",               out_point["refined_total"]),
            _row("in_mfg_F",                    out_point["in_mfg_F"]),
            _row("in_mfg_I",                    out_point["in_mfg_I"]),
            _row("NET_PARTS (850790)",          out_point["NET_PARTS"]),
            _row("MFG_F (formal)",              out_point["MFG_F"]),
            _row("MFG_I (informal)",            out_point["MFG_I"]),
            _row("MFG_total = sum",             out_point["MFG_total"]),
            _row("xover_mfg = in_mfg_F − REFINE_SEC_F (info only)",
                 out_point["xover_mfg"]),
        ])

    # ---------------- Step 9: INSTALL ---------------------------------
    with st.expander("Step 9 — Installation: both lanes' batteries flow to installs"):
        st.latex(
            r"\mathrm{INSTALL\_implied} = \mathrm{MFG}_F + \mathrm{MFG}_I"
            r" + \mathrm{imp\_batt} - \mathrm{exp\_batt}"
        )
        st.latex(
            r"\mathrm{INSTALL\_target} = \Delta\mathrm{Stock} + \mathrm{RETIRE}"
        )
        st.caption(
            "Finished-battery trade (850710 + 850720) attaches to the FORMAL lane "
            "and contributes directly to installs. Both formal and informal "
            "batteries get installed — the lanes rejoin at install. Residuals "
            "(§4): install is two-sided; refine is a one-sided shortfall + "
            "overshoot magnitude reported separately."
        )
        _show([
            _row("MFG_F",                       out_point["MFG_F"]),
            _row("MFG_I",                       out_point["MFG_I"]),
            _row("MFG_total",                   out_point["MFG_total"]),
            _row("imp_batt (850710+850720, formal)", arr["imp_batt"]),
            _row("exp_batt (850710+850720, formal)", arr["exp_batt"]),
            _row("INSTALL_implied",             out_point["INSTALL_implied"]),
            _row("ΔStock = k·stock − k·stock_prev", out_point["dStock"]),
            _row("RETIRE",                      out_point["RETIRE"]),
            _row("INSTALL_target",              out_point["INSTALL_target"]),
        ])


def _render_residuals_and_flags(out_point: dict, arr: dict,
                                beta: float, gamma: float, etas: dict,
                                g: float, tau: float, k_used: float):
    """Two-residual readouts (one-sided refine floor + two-sided install)
    plus crossover and overshoot indicators.

    out_point: forward_parallel_chain output at the user's committed k.
    """
    years = arr["year"]

    st.markdown("##### Two residuals — install (two-sided), refine (one-sided floor)")
    df_res = pd.DataFrame({
        "Year": years.astype(int),
        "Install fit (signed %)":     np.round(out_point["res_install_per_year"] * 100, 2),
        "Refine shortfall (one-sided %)":
            np.round(out_point["shortfall_refine_per_year"] * 100, 2),
        "Refine overshoot (kt Pb)":
            np.round(out_point["overshoot_refine_per_year"] / 1000.0, 1),
    })
    st.dataframe(df_res, hide_index=True, use_container_width=True)

    mean_res_inst = float(np.mean(out_point["res_install_per_year"]))
    mean_shortfall = float(np.mean(out_point["shortfall_refine_per_year"]))
    sum_overshoot  = float(np.sum(out_point["overshoot_refine_per_year"]))

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Install fit — mean (signed %)",
        f"{mean_res_inst*100:+.1f}%",
        help="(INSTALL_implied − INSTALL_target) / INSTALL_target. Two-sided. "
             "Negative → chain falls short of the stock-derived target. "
             "Window-sum residual: "
             f"{out_point['res_install_W']*100:+.2f}%.",
    )
    c2.metric(
        "Refine shortfall (one-sided %)",
        f"{mean_shortfall*100:+.1f}%",
        help="max(0, USGS_sec − REFINE_SEC_F) / USGS_sec. ONE-SIDED FLOOR — "
             "0% if REFINE_SEC_F ≥ USGS_sec. Window-sum: "
             f"{out_point['shortfall_refine_W']*100:.2f}%.",
    )
    c3.metric(
        "Refine overshoot (sum, kt Pb)",
        f"{sum_overshoot/1000:,.0f} kt",
        help="Sum over the window of max(0, REFINE_SEC_F − USGS_sec). "
             "The implied informal / unrecorded formal-equivalent refined Pb. "
             "Allowed and expected, not a failure.",
    )

    st.divider()

    # ---- (b) Infeasibility flag — install-side under parallel chain --------
    # Compare INSTALL_target at the user's k against the all-formal INSTALL_implied
    # at the same (k, τ) — the chain's supply ceiling under maximum formal recovery.
    INSTALL_target  = out_point["INSTALL_target"]
    INSTALL_ceiling = _max_install_implied(arr, g, tau, k_used, beta, gamma, etas)
    shortfall       = INSTALL_target - INSTALL_ceiling
    infeasible_yrs  = [int(y) for y, sf in zip(years, shortfall) if sf > 0]

    # ---- (c) Overshoot interpretation -------------------------------------
    # Under the floor, overshoot magnitude = implied unrecorded refined lead.
    overshoot_per_yr = _unrecorded_feedstock_per_year(out_point, arr["sec_usgs"])
    mean_overshoot   = float(np.mean(overshoot_per_yr))

    # ---- (d) Crossover health: smelt + refine boundaries ------------------
    neg_xover_smelt = [int(y) for y, x in zip(years, out_point["xover_smelt"])  if x < -1e-3]
    neg_xover_ref   = [int(y) for y, x in zip(years, out_point["xover_refine"]) if x < -1e-3]

    st.markdown("##### Indicators (distinct, can co-occur)")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        if infeasible_yrs:
            st.error(
                f"**(b) Install ceiling exceeded** — stock-derived INSTALL_target "
                f"exceeds the parallel chain's max-formal INSTALL_implied at k = "
                f"{k_used:.2f}, τ = {tau:.2f} yr in years {infeasible_yrs}.\n\n"
                f"Mean per-year shortfall: **{_fmt_kt(float(np.mean(shortfall)))}** "
                f"(window-sum {_fmt_kt(float(np.sum(shortfall)))}).\n\n"
                "**Even at φ → all-formal the chain cannot supply this stock. "
                "Stock is too high, or τ too low, or trade-side supply too small.**"
            )
        else:
            st.success(
                "**(b) Install side feasible** at the current (k, τ): the "
                "parallel chain at max-formal can supply INSTALL_target."
            )
    with ic2:
        if mean_overshoot > 0:
            st.info(
                f"**(c) Refine overshoot present** — mean **{_fmt_kt(mean_overshoot)}/yr** "
                f"of formal refined Pb above USGS secondary.\n\n"
                "Under the **one-sided floor** framing this is the implied "
                "unrecorded / informal formal-equivalent refined lead. "
                "Allowed and expected; not a failure."
            )
        else:
            st.warning(
                "**(c) No overshoot — REFINE_SEC_F sits below USGS secondary.** "
                "The chain undershoots the floor. Raise φ_refine_f or "
                "upstream φ's to lift formal refined output."
            )
    with ic3:
        if neg_xover_smelt or neg_xover_ref:
            st.error(
                f"**(d) Crossover infeasibility** — implied informal→formal "
                f"crossover went NEGATIVE at:\n\n"
                f"- smelt: years {neg_xover_smelt if neg_xover_smelt else '—'}\n"
                f"- refine: years {neg_xover_ref if neg_xover_ref else '—'}\n\n"
                "A negative crossover means the upstream formal output exceeds "
                "what the downstream formal lane can absorb at this φ. The "
                "ordering is locally infeasible there."
            )
        else:
            st.success(
                "**(d) Crossovers healthy** — implied informal→formal crossover "
                "is non-negative at every boundary and year."
            )

    st.caption(
        "Install is two-sided; refine is a one-sided floor + overshoot magnitude "
        "reported separately. Overshoot is the implied unrecorded informal-formal-"
        "equivalent refined lead. The crossover indicator is your sanity check "
        "on the φ ordering."
    )


def _render_flow_table_live(arr: dict, g: float, gamma: float, beta: float,
                            phi: dict, etas: dict,
                            k_sel: float, tau_sel: float):
    """Per-step flow table at the (k, τ) currently picked on the ridge.

    Shows both lanes at every stage + implied crossovers.
    """
    out_live = forward_parallel_chain(
        arr, k_stock=k_sel, phi=phi,
        g=g, tau=tau_sel, beta=beta, gamma=gamma,
        **{k_: v for k_, v in etas.items() if k_ in V5_ETA_DEFAULTS},
    )
    st.markdown(
        f"##### Per-step flow table at selected (k = **{k_sel:.3f}**, τ = **{tau_sel:.2f} yr**)"
    )
    st.caption(
        f"Flows at current ridge point. Both lanes shown at every stage; "
        "crossovers report the implied informal→formal pull at each boundary. "
        "**Absolute scale is conditional** on k; shape is fixed, level is not."
    )
    keys = [
        "COLLECT",
        "out_break_F", "out_break_I", "BREAK_total",
        "scrap_total",
        "out_smelt_F", "out_smelt_I", "SMELT_total",
        "crude_total",
        "REFINE_SEC_F", "REFINE_SEC_I", "REFINE_SEC_total",
        "refined_total",
        "MFG_F", "MFG_I", "MFG_total",
        "INSTALL_implied", "INSTALL_target",
        "xover_smelt", "xover_refine", "xover_mfg",
    ]
    years = arr["year"]
    data = {"Stage": keys}
    for i, yr in enumerate(years):
        data[str(int(yr))] = [_fmt2sig(out_live[k][i]) for k in keys]
    df = pd.DataFrame(data)
    for col in df.columns:
        df[col] = df[col].astype(str)
    st.dataframe(df, hide_index=True, use_container_width=True)


def _render_kt_ridge_static(arr: dict, g: float, gamma: float, beta: float,
                            phi: dict, etas: dict,
                            k_main: float, tau_main: float) -> None:
    """k/τ ridge plot (read-only). Renders the install ceiling and refine
    floor crossings over a τ grid; marks the current (k, τ) from §1.

    No slider here — k and τ come from §1.
    """
    st.caption(
        "Two threshold curves on (k, τ). The install ceiling is the largest k "
        "the chain at max-formal can supply. The refine floor is the smallest "
        "k that lifts REFINE_SEC_F to USGS_sec. The black marker is your "
        "current point from §1. Edit k or τ in §1 to move it."
    )

    tau_grid = np.linspace(2.0, 12.0, 51)
    ridge_df = _ridge_curves(arr, g, gamma, beta, etas, tau_grid)

    k_inst_at_tau = float(_k_install_feasibility_crossing(arr, g, tau_main, beta, gamma, etas))
    fig = _plot_ridge(ridge_df, tau_pick=tau_main, k_pick=k_main,
                      k_feas_crossing=k_inst_at_tau)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def _phi_eta_curve_break(arr: dict, k: float, g: float, tau: float,
                         gamma: float, beta: float, etas: dict, phi: dict,
                         eta_break_I_grid: np.ndarray) -> pd.DataFrame:
    """Sweep η_break_I; solve φ_break_f that closes USGS = Σ REFINE_SEC_F
    (window-sum), holding φ_smelt_f, φ_refine_f at their current values."""
    r = retire_rate(g, tau)
    COLLECT_sum = float(np.sum(gamma * arr["stock"] * k * r))
    delta_      = etas["delta"]
    eta_bF      = etas["eta_break_F"]
    eta_sF      = etas["eta_smelt_F"]; eta_sI = etas["eta_smelt_I"]
    eta_rF      = etas["eta_refine_F"]
    Δused_sum   = float(np.sum(arr["imp_used"] - arr["exp_used"]))
    Δscrap_sum  = float(np.sum(arr["imp_scrap"] - arr["exp_scrap"]))
    Δcrude_sum  = float(np.sum(arr["imp_crude"] - arr["exp_crude"]))
    USGS_sum    = float(np.sum(arr["sec_usgs"]))
    h_smelt     = phi["phi_smelt_f"] * eta_sF + (1.0 - phi["phi_smelt_f"]) * eta_sI
    phi_rf      = phi["phi_refine_f"]
    rows = []
    for eta_I in eta_break_I_grid:
        denom = delta_ * COLLECT_sum
        if denom <= 0 or h_smelt <= 0 or phi_rf <= 0:
            rows.append({"eta_break_I": float(eta_I),
                         "phi_break_f": float("nan"), "feasible": False})
            continue
        h_break_target = (
            (USGS_sum / (phi_rf * eta_rF) - Δcrude_sum) / h_smelt
            - delta_ * eta_bF * Δused_sum - Δscrap_sum
        ) / denom
        if eta_bF == eta_I:
            rows.append({"eta_break_I": float(eta_I),
                         "phi_break_f": float("nan"), "feasible": False})
            continue
        phi_b = (h_break_target - eta_I) / (eta_bF - eta_I)
        rows.append({"eta_break_I": float(eta_I),
                     "phi_break_f": float(phi_b),
                     "feasible": bool(0.0 <= phi_b <= 1.0)})
    return pd.DataFrame(rows)


def _phi_eta_curve_smelt(arr: dict, k: float, g: float, tau: float,
                         gamma: float, beta: float, etas: dict, phi: dict,
                         eta_smelt_I_grid: np.ndarray) -> pd.DataFrame:
    """Sweep η_smelt_I; solve φ_smelt_f from USGS = Σ REFINE_SEC_F."""
    r = retire_rate(g, tau)
    COLLECT_sum = float(np.sum(gamma * arr["stock"] * k * r))
    delta_      = etas["delta"]
    eta_bF      = etas["eta_break_F"]; eta_bI = etas["eta_break_I"]
    eta_sF      = etas["eta_smelt_F"]
    eta_rF      = etas["eta_refine_F"]
    Δused_sum   = float(np.sum(arr["imp_used"] - arr["exp_used"]))
    Δscrap_sum  = float(np.sum(arr["imp_scrap"] - arr["exp_scrap"]))
    Δcrude_sum  = float(np.sum(arr["imp_crude"] - arr["exp_crude"]))
    USGS_sum    = float(np.sum(arr["sec_usgs"]))
    h_break     = phi["phi_break_f"] * eta_bF + (1.0 - phi["phi_break_f"]) * eta_bI
    scrap_sum   = (delta_ * h_break * COLLECT_sum
                   + delta_ * eta_bF * Δused_sum + Δscrap_sum)
    phi_rf      = phi["phi_refine_f"]
    rows = []
    for eta_I in eta_smelt_I_grid:
        if scrap_sum <= 0 or phi_rf <= 0:
            rows.append({"eta_smelt_I": float(eta_I),
                         "phi_smelt_f": float("nan"), "feasible": False})
            continue
        h_smelt_target = (USGS_sum / (phi_rf * eta_rF) - Δcrude_sum) / scrap_sum
        if eta_sF == eta_I:
            rows.append({"eta_smelt_I": float(eta_I),
                         "phi_smelt_f": float("nan"), "feasible": False})
            continue
        phi_s = (h_smelt_target - eta_I) / (eta_sF - eta_I)
        rows.append({"eta_smelt_I": float(eta_I),
                     "phi_smelt_f": float(phi_s),
                     "feasible": bool(0.0 <= phi_s <= 1.0)})
    return pd.DataFrame(rows)


def _phi_eta_curve_refine(arr: dict, k: float, g: float, tau: float,
                          gamma: float, beta: float, etas: dict, phi: dict,
                          eta_refine_I_grid: np.ndarray) -> pd.DataFrame:
    """Sweep η_refine_I; solve φ_refine_f from USGS = Σ REFINE_SEC_F.

    Note: REFINE_SEC_F = crude_total · φ_refine_f · η_refine_F does NOT depend
    on η_refine_I, so the solved φ_refine_f is a constant across the sweep.
    The curve is therefore flat — informative because it tells you the
    informal refine efficiency does not move the formal-anchored solve.
    """
    r = retire_rate(g, tau)
    COLLECT_sum = float(np.sum(gamma * arr["stock"] * k * r))
    delta_      = etas["delta"]
    eta_bF      = etas["eta_break_F"]; eta_bI = etas["eta_break_I"]
    eta_sF      = etas["eta_smelt_F"]; eta_sI = etas["eta_smelt_I"]
    eta_rF      = etas["eta_refine_F"]
    Δused_sum   = float(np.sum(arr["imp_used"] - arr["exp_used"]))
    Δscrap_sum  = float(np.sum(arr["imp_scrap"] - arr["exp_scrap"]))
    Δcrude_sum  = float(np.sum(arr["imp_crude"] - arr["exp_crude"]))
    USGS_sum    = float(np.sum(arr["sec_usgs"]))
    h_break     = phi["phi_break_f"] * eta_bF + (1.0 - phi["phi_break_f"]) * eta_bI
    h_smelt     = phi["phi_smelt_f"] * eta_sF + (1.0 - phi["phi_smelt_f"]) * eta_sI
    scrap_sum   = (delta_ * h_break * COLLECT_sum
                   + delta_ * eta_bF * Δused_sum + Δscrap_sum)
    crude_sum   = h_smelt * scrap_sum + Δcrude_sum
    if crude_sum <= 0:
        phi_rf = float("nan")
    else:
        phi_rf = USGS_sum / (eta_rF * crude_sum)
    feasible = (not np.isnan(phi_rf)) and (0.0 <= phi_rf <= 1.0)
    rows = [{"eta_refine_I": float(e), "phi_refine_f": float(phi_rf),
             "feasible": bool(feasible)} for e in eta_refine_I_grid]
    return pd.DataFrame(rows)


def _phi_eta_curve_mfg(arr: dict, k: float, g: float, tau: float,
                       gamma: float, beta: float, etas: dict, phi: dict,
                       eta_mfg_I_grid: np.ndarray) -> pd.DataFrame:
    """Sweep η_mfg_I; solve φ_mfg_f from Σ INSTALL_implied = Σ INSTALL_target."""
    r = retire_rate(g, tau)
    stock = arr["stock"] * k
    stock_pre = float(arr["stock_pre"]) * k
    prev = np.concatenate([[stock_pre], stock[:-1]])
    INSTALL_target_sum = float(np.sum((stock - prev) + stock * r))

    COLLECT_sum = float(np.sum(gamma * stock * r))
    delta_      = etas["delta"]
    eta_bF      = etas["eta_break_F"]; eta_bI = etas["eta_break_I"]
    eta_sF      = etas["eta_smelt_F"]; eta_sI = etas["eta_smelt_I"]
    eta_rF      = etas["eta_refine_F"]; eta_rI = etas["eta_refine_I"]
    eta_mF      = etas["eta_mfg_F"]
    Δused_sum   = float(np.sum(arr["imp_used"] - arr["exp_used"]))
    Δscrap_sum  = float(np.sum(arr["imp_scrap"] - arr["exp_scrap"]))
    Δcrude_sum  = float(np.sum(arr["imp_crude"] - arr["exp_crude"]))
    Δfeed_sum   = float(np.sum(arr["imp_feed"] - arr["exp_feed"]))
    Δbatt_sum   = float(np.sum(arr["imp_batt"] - arr["exp_batt"]))
    prim_sum    = float(np.sum(arr["prim_usgs"]))
    parts_pos_sum = float(np.sum(np.maximum(arr["imp_parts"] - arr["exp_parts"], 0.0)))

    h_break = phi["phi_break_f"] * eta_bF + (1.0 - phi["phi_break_f"]) * eta_bI
    h_smelt = phi["phi_smelt_f"] * eta_sF + (1.0 - phi["phi_smelt_f"]) * eta_sI
    h_refine = phi["phi_refine_f"] * eta_rF + (1.0 - phi["phi_refine_f"]) * eta_rI

    scrap_sum   = delta_ * h_break * COLLECT_sum + delta_ * eta_bF * Δused_sum + Δscrap_sum
    crude_sum   = h_smelt * scrap_sum + Δcrude_sum
    refined_sum = h_refine * crude_sum + prim_sum + Δfeed_sum
    M_const     = eta_mF * parts_pos_sum + Δbatt_sum

    rows = []
    for eta_I in eta_mfg_I_grid:
        if refined_sum <= 0 or beta <= 0:
            rows.append({"eta_mfg_I": float(eta_I),
                         "phi_mfg_f": float("nan"), "feasible": False})
            continue
        h_mfg_target = (INSTALL_target_sum - M_const) / (beta * refined_sum)
        if eta_mF == eta_I:
            rows.append({"eta_mfg_I": float(eta_I),
                         "phi_mfg_f": float("nan"), "feasible": False})
            continue
        phi_m = (h_mfg_target - eta_I) / (eta_mF - eta_I)
        rows.append({"eta_mfg_I": float(eta_I),
                     "phi_mfg_f": float(phi_m),
                     "feasible": bool(0.0 <= phi_m <= 1.0)})
    return pd.DataFrame(rows)


def _plot_phi_eta_curve(df: pd.DataFrame, x_col: str, y_col: str,
                        x_label: str, y_label: str, title: str,
                        eta_I_marker: float, phi_floor: float | None = None) -> go.Figure:
    fig = go.Figure()
    feasible = df[df["feasible"]]
    infeas   = df[~df["feasible"]]
    fig.add_trace(go.Scatter(
        x=feasible[x_col], y=feasible[y_col],
        mode="lines+markers", name="φ ∈ [0, 1] (feasible)",
        line=dict(color="#2ca02c", width=2),
    ))
    if not infeas.empty:
        fig.add_trace(go.Scatter(
            x=infeas[x_col], y=infeas[y_col],
            mode="markers", name="φ outside [0, 1]",
            marker=dict(color="#888", size=6, symbol="x"),
        ))
    fig.add_vline(
        x=eta_I_marker, line_dash="dot", line_color="#444", opacity=0.6,
        annotation_text=f"η_I = {eta_I_marker:.2f}",
        annotation_position="bottom right",
    )
    fig.add_hline(y=0.0, line_dash="dot", line_color="#888", opacity=0.5)
    fig.add_hline(y=1.0, line_dash="dot", line_color="#888", opacity=0.5,
                  annotation_text="feasibility ceiling",
                  annotation_position="top left")
    if phi_floor is not None and phi_floor > 0:
        fig.add_hline(y=phi_floor, line_dash="dot", line_color="#c33",
                      opacity=0.55,
                      annotation_text=f"USAID floor {phi_floor:.2f}",
                      annotation_position="top right")
    fig.update_layout(
        height=320, margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title=x_label, yaxis_title=y_label,
        title=title, legend=dict(orientation="h", y=-0.22, x=0.0),
        yaxis_range=[-0.2, 1.6],
    )
    return fig


def _render_phi_eta_curves(arr: dict, k: float, g: float, tau: float,
                           gamma: float, beta: float, etas: dict, phi: dict):
    """Four φ↔η_I closed-form panels. Each panel sweeps the corresponding
    informal η_I; solves the φ that would close the relevant anchor (USGS
    refined floor for break/smelt/refine; install equality for mfg).

    The current η_I value in §2 marks where the user sits on each curve;
    the implied φ at that marker is the φ value consistent with the anchor
    given the rest of §1/§2.
    """
    st.caption(
        "USGS pins the **blended** formal-refined output; the install anchor "
        "pins blended mfg output. Each panel shows the formal share φ at one "
        "stage that closes the corresponding anchor, as a function of the "
        "assumed informal η at that stage, holding the other three φ's at "
        "their current §2 values. **The current η_I from §2 (vertical dotted "
        "line) reads off your implied φ on the curve.** Red dotted = USAID φ "
        "floor for that stage."
    )

    e_break_I = etas["eta_break_I"]
    e_smelt_I = etas["eta_smelt_I"]
    e_ref_I   = etas["eta_refine_I"]
    e_mfg_I   = etas["eta_mfg_I"]

    g_break = np.linspace(0.40, 0.95, 56)
    g_smelt = np.linspace(0.40, 0.97, 58)
    g_ref   = np.linspace(0.50, 0.97, 48)
    g_mfg   = np.linspace(0.50, 0.97, 48)

    df_b = _phi_eta_curve_break(arr, k, g, tau, gamma, beta, etas, phi, g_break)
    df_s = _phi_eta_curve_smelt(arr, k, g, tau, gamma, beta, etas, phi, g_smelt)
    df_r = _phi_eta_curve_refine(arr, k, g, tau, gamma, beta, etas, phi, g_ref)
    df_m = _phi_eta_curve_mfg(arr, k, g, tau, gamma, beta, etas, phi, g_mfg)

    rows_top = st.columns(2)
    rows_bot = st.columns(2)

    with rows_top[0]:
        st.plotly_chart(_plot_phi_eta_curve(
            df_b, x_col="eta_break_I", y_col="phi_break_f",
            x_label="η_break_I", y_label="implied φ_break_f",
            title=f"Breaking — USGS-anchored (k={k:.2f}, τ={tau:.2f})",
            eta_I_marker=e_break_I, phi_floor=PHI_FLOORS["phi_break_f"],
        ), use_container_width=True, theme="streamlit")
    with rows_top[1]:
        st.plotly_chart(_plot_phi_eta_curve(
            df_s, x_col="eta_smelt_I", y_col="phi_smelt_f",
            x_label="η_smelt_I", y_label="implied φ_smelt_f",
            title=f"Smelting — USGS-anchored (k={k:.2f}, τ={tau:.2f})",
            eta_I_marker=e_smelt_I, phi_floor=PHI_FLOORS["phi_smelt_f"],
        ), use_container_width=True, theme="streamlit")
    with rows_bot[0]:
        st.plotly_chart(_plot_phi_eta_curve(
            df_r, x_col="eta_refine_I", y_col="phi_refine_f",
            x_label="η_refine_I", y_label="implied φ_refine_f",
            title=f"Refining — USGS-anchored (FLAT: REFINE_SEC_F ⊥ η_refine_I)",
            eta_I_marker=e_ref_I, phi_floor=PHI_FLOORS["phi_refine_f"],
        ), use_container_width=True, theme="streamlit")
        st.caption(
            "REFINE_SEC_F = crude × φ_refine × η_refine_F doesn't depend on "
            "η_refine_I; the curve is flat. The horizontal line is the φ "
            "value that closes USGS at the current upstream φ's."
        )
    with rows_bot[1]:
        st.plotly_chart(_plot_phi_eta_curve(
            df_m, x_col="eta_mfg_I", y_col="phi_mfg_f",
            x_label="η_mfg_I", y_label="implied φ_mfg_f",
            title=f"Manufacturing — install-anchored (k={k:.2f}, τ={tau:.2f})",
            eta_I_marker=e_mfg_I, phi_floor=PHI_FLOORS["phi_mfg_f"],
        ), use_container_width=True, theme="streamlit")
        st.caption(
            "φ_mfg doesn't enter REFINE_SEC_F, so the mfg curve is solved "
            "against the install anchor: INSTALL_implied = INSTALL_target."
        )


def _plot_feasibility_map(grid: dict, tau_low: float, tau_high: float,
                          k_over_tau_ratio: float,
                          mc_band: dict | None = None,
                          k_marker: float | None = None,
                          tau_marker: float | None = None) -> go.Figure:
    """Render the four-state feasibility heatmap with overlays."""
    k_grid = grid["k_grid"]
    tau_grid = grid["tau_grid"]
    state = grid["state"]

    # Discrete colorscale: 0=neither, 1=install-only, 2=refine-only, 3=both
    colors = ["#444444", "#1f77b4", "#d62728", "#2ca02c"]
    labels = {0: "infeasible (both walls fail)",
              1: "install-only feasible",
              2: "refine-only feasible",
              3: "feasible-both (possible)"}
    # Plotly colorscale positions must lie in [0, 1]. With zmin=0, zmax=3
    # and 4 discrete states, we split [0, 1] into 4 equal bands.
    n = len(colors)
    colorscale = []
    for i, c in enumerate(colors):
        colorscale.append([i / n, c])
        colorscale.append([min((i + 1) / n, 1.0), c])

    custom_labels = np.empty_like(state, dtype=object)
    for v, lab in labels.items():
        custom_labels[state == v] = lab

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=state, x=tau_grid, y=k_grid,
        colorscale=colorscale, zmin=0, zmax=3,
        showscale=False,
        customdata=custom_labels,
        hovertemplate=("τ=%{x:.2f} yr<br>k=%{y:.2f}<br>"
                       "state: %{customdata}<extra></extra>"),
    ))

    # Wall lines: k_min_refine(τ) below this k -> refine-infeasible
    fig.add_trace(go.Scatter(
        x=tau_grid, y=grid["k_min_refine"],
        mode="lines", name="refine wall (max-recovery REFINE_SEC ≥ USGS_sec)",
        line=dict(color="#a00", width=2),
        hovertemplate="τ=%{x:.2f}<br>k_min_refine=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=tau_grid, y=grid["k_max_install"],
        mode="lines", name="install wall (INSTALL_target ≤ USGS+trade ceiling)",
        line=dict(color="#06a", width=2),
        hovertemplate="τ=%{x:.2f}<br>k_max_install=%{y:.3f}<extra></extra>",
    ))

    # Plausible-τ band
    fig.add_vrect(
        x0=tau_low, x1=tau_high,
        fillcolor="rgba(255,255,255,0.18)", line_width=0,
        annotation_text=f"plausible τ ({tau_low:.1f}–{tau_high:.1f} yr)",
        annotation_position="top",
    )

    # Pinned k/τ reference line
    fig.add_trace(go.Scatter(
        x=tau_grid, y=k_over_tau_ratio * tau_grid,
        mode="lines", name=f"k = {k_over_tau_ratio:.2f} · τ  (pinned ratio)",
        line=dict(color="#fff", width=1.5, dash="dot"),
        hoverinfo="skip",
    ))

    # Optional MC envelopes around both walls
    if mc_band is not None:
        for key_lo, key_hi, color, label in (
            ("k_min_refine_p5",  "k_min_refine_p95",
             "rgba(170,0,0,0.30)", "refine-wall ±MC (p5–p95)"),
            ("k_max_install_p5", "k_max_install_p95",
             "rgba(0,90,170,0.30)", "install-wall ±MC (p5–p95)"),
        ):
            y_lo = mc_band[key_lo]
            y_hi = mc_band[key_hi]
            fig.add_trace(go.Scatter(
                x=np.concatenate([tau_grid, tau_grid[::-1]]),
                y=np.concatenate([y_hi, y_lo[::-1]]),
                fill="toself", fillcolor=color,
                line=dict(color="rgba(0,0,0,0)"),
                name=label, hoverinfo="skip",
            ))

    if k_marker is not None and tau_marker is not None:
        fig.add_trace(go.Scatter(
            x=[float(tau_marker)], y=[float(k_marker)],
            mode="markers",
            name=f"current §1 (τ={tau_marker:.2f}, k={k_marker:.2f})",
            marker=dict(color="#000", size=14, symbol="x",
                        line=dict(color="#fff", width=2)),
            hovertemplate=("current §1 point<br>τ=%{x:.2f}<br>k=%{y:.2f}"
                           "<extra></extra>"),
        ))

    fig.update_layout(
        height=520, margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="τ (effective battery lifetime, yrs)",
        yaxis_title="k (stock multiplier)",
        legend=dict(orientation="h", y=-0.18, x=0.0),
        title="Feasibility-region map — possibility, not fit quality",
    )
    return fig


def _feasibility_mc_cache_key(beta: float, gamma: float, etas: dict,
                              k_grid: np.ndarray, tau_grid: np.ndarray) -> str:
    payload = {
        "beta": round(beta, 6), "gamma": round(gamma, 6),
        "etas": {k: round(v, 6) for k, v in etas.items()},
        "k_grid": [round(float(x), 4) for x in k_grid],
        "tau_grid": [round(float(x), 4) for x in tau_grid],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _render_feasibility_map(arr: dict, g: float, gamma: float, beta: float,
                            etas: dict,
                            k_main: float | None = None,
                            tau_main: float | None = None):
    st.markdown(
        "##### Feasibility-region map — bounds the (k, τ) solution space"
    )
    st.caption(
        "Each cell is shaded by **physical possibility, not fit quality**. "
        "‘Feasible-both’ = the chain can supply USGS-observed secondary AND "
        "can install the stock-derived target at that (k, τ). The map *bounds* "
        "the solution region; it does not select a point within it. Pinning a "
        "point still requires an external τ."
    )

    c_band, c_ratio = st.columns([4, 2])
    with c_band:
        tau_low, tau_high = st.slider(
            "Plausible-τ band (yrs) — physical lifetime constraint, "
            "not a model output",
            min_value=1.0, max_value=12.0, step=0.25,
            value=PLAUSIBLE_TAU_DEFAULT,
            key="v4_feas_band",
            help="Lead-acid batteries do not last under ~2–3 yr (SLI) or above "
                 "~10–12 yr (stationary). The band is a sanity envelope; "
                 "feasibility itself does not bound τ from below.",
        )
    with c_ratio:
        k_over_tau = st.number_input(
            "Pinned k/τ ratio (overlay)",
            min_value=0.02, max_value=0.30, step=0.01,
            value=PINNED_K_OVER_TAU,
            key="v4_feas_kt_ratio",
            help="The identified ridge has k ≈ ratio · τ. Default 0.11 is a "
                 "working central value; adjust to compare other working "
                 "hypotheses.",
        )

    grid = _feasibility_grid(arr, g=g, gamma=gamma, beta=beta, etas=etas)

    # MC overlay (opt-in)
    c_mc, c_mc_caption = st.columns([3, 5])
    run_mc = c_mc.button("Add wall-position uncertainty (±5% MC)",
                         key="v4_feas_mc_run")
    cache = st.session_state.get("v4_feas_mc_cache", {})
    key = _feasibility_mc_cache_key(beta, gamma, etas,
                                    grid["k_grid"], grid["tau_grid"])
    if run_mc:
        with st.spinner("Running 100 MC draws on the wall positions…"):
            t0 = time.time()
            mc = _feasibility_grid_mc(arr, g=g, gamma=gamma, beta=beta, etas=etas,
                                      k_grid=grid["k_grid"],
                                      tau_grid=grid["tau_grid"],
                                      n_draws=100)
            mc["_elapsed"] = time.time() - t0
        cache[key] = mc
        st.session_state["v4_feas_mc_cache"] = cache
    mc_band = cache.get(key)
    if mc_band is not None:
        c_mc_caption.caption(
            f"MC overlay active (100 draws, {mc_band.get('_elapsed', 0):.1f}s). "
            "Wall positions carry ±5% parametric uncertainty; this is a "
            "bounded wiggle, **distinct** from the k/τ structural degeneracy."
        )

    fig = _plot_feasibility_map(grid, tau_low=tau_low, tau_high=tau_high,
                                k_over_tau_ratio=k_over_tau,
                                mc_band=mc_band,
                                k_marker=k_main, tau_marker=tau_main)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # Headline read-out: closure τ_max + k-range in plausible-τ band
    tau_max = _feasibility_closure_tau(grid)
    band = _feasible_k_range_in_band(grid, tau_low=tau_low, tau_high=tau_high)

    cA, cB = st.columns(2)
    with cA:
        if tau_max is None:
            st.info(
                "**Feasibility closure τ_max**: not reached inside the grid "
                f"(feasible-both still exists at τ = {grid['tau_grid'][-1]:.2f} yr)."
            )
        elif math.isnan(tau_max):
            st.error(
                "**Feasibility closure τ_max**: the feasible-both region is "
                "empty everywhere in the grid. Try widening the grid or "
                "loosening the formal efficiencies."
            )
        else:
            st.warning(
                f"**Feasibility closure τ_max ≈ {tau_max:.2f} yr.** "
                "Above this τ the install and refine walls cross — no (k, τ) "
                "is physically possible. *A hard upper bound on τ.*"
            )
    with cB:
        if band["n_cells"] == 0:
            st.error(
                f"**No feasible-both cell in the plausible-τ band "
                f"({tau_low:.1f}–{tau_high:.1f} yr).** Either the band is "
                "outside the feasible region, or the assumptions need to "
                "loosen."
            )
        else:
            st.success(
                f"**Feasible-both k-range in plausible-τ band "
                f"({tau_low:.1f}–{tau_high:.1f} yr): "
                f"k ∈ [{band['k_min']:.2f}, {band['k_max']:.2f}]** "
                f"(τ-window where it exists: "
                f"{band['tau_min']:.2f}–{band['tau_max']:.2f} yr; "
                f"{band['n_cells']} cells)."
            )

    st.caption(
        "**Captions / honesty.** *(a)* Above τ_max the two walls cross and no "
        "(k, τ) is feasible — a hard upper bound on τ. *(b)* Feasibility does "
        "NOT bound τ from below; the short-τ / low-k corner stays feasible. "
        "The plausible-τ band is a physical-life constraint, not a model "
        "output. *(c)* Where the φ curves above show implied φ > 1, the "
        "corresponding (k, τ) is shaded refine-infeasible here — same wall, "
        "different view. *(d)* The surviving region is a **bound** on the "
        "answer, not the answer. No point is preferred inside it."
    )


def _render_monte_carlo(df_csv: pd.DataFrame, user_stock: pd.DataFrame,
                        segments_df: pd.DataFrame, fit_window: tuple[int, int],
                        beta: float, gamma: float, phi: dict, etas: dict,
                        k_sel: float, tau_sel: float, arr: dict):
    st.markdown("##### Monte Carlo — parametric uncertainty (opt-in)")
    st.caption(
        f"±{int(EFF_PERTURB*100)}% on all efficiencies (formal AND informal) and the "
        f"{len(HS_FACTOR_DEFAULTS)} HS conversion factors; ±{int(BETA_PERTURB*100)}% "
        f"on β; ±{int(USGS_PRIMARY_PERTURB*100)}% on USGS primary. USGS secondary is "
        "left unperturbed (it is the refine-side floor target). **Parametric "
        "uncertainty only — does NOT capture k/τ structural ambiguity or the φ "
        "ordering, which are shown by the ridge and the φ-sensitivity curves.**"
    )
    key = _mc_cache_key(beta, gamma, user_stock, segments_df, etas, phi,
                        k_sel, tau_sel, fit_window)
    cached = st.session_state.get("v4_mc_cache", {})
    cached_for_key = cached.get(key)

    cols = st.columns([3, 2, 5])
    run = cols[0].button("Run uncertainty (Monte Carlo)", key="v4_mc_run")
    if cached_for_key:
        cols[1].caption(f"cached n={cached_for_key['n_draws']}")

    if run:
        with st.spinner(f"Running {MC_DRAWS} draws…"):
            t0 = time.time()
            result = _run_monte_carlo(
                df_csv=df_csv, user_stock=user_stock, fit_window=fit_window,
                beta=beta, gamma=gamma, phi=phi, etas=etas,
                k_stock=k_sel, tau=tau_sel,
                n_draws=MC_DRAWS,
            )
            elapsed = time.time() - t0
        result["elapsed_s"] = elapsed
        cached[key] = result
        st.session_state["v4_mc_cache"] = cached
        cached_for_key = result
        st.caption(f"MC complete in {elapsed:.1f}s ({result.get('n_draws', 0)} ok).")

    if not cached_for_key or not cached_for_key.get("ok"):
        st.info("Press **Run uncertainty (Monte Carlo)** to draw parametric bands.")
        return

    n = int(cached_for_key.get("n_draws", 0))
    n_surv = int(cached_for_key.get("n_survivors", 0))
    res_install_W = cached_for_key["res_install_W"]
    shortfall_W   = cached_for_key["shortfall_refine_W"]
    overshoot_W   = cached_for_key["overshoot_refine_W"]

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Survivors (both anchors ≤ 5%)",
              f"{n_surv} / {n}",
              help="Draws closing |res_install_W| ≤ 5% AND shortfall_refine_W ≤ 5% "
                   "at the user's (k, τ, φ).")
    c1.metric("Install — MC median (window-sum)",
              f"{res_install_W[1]*100:+.1f}%",
              help=f"p5={res_install_W[0]*100:+.1f}%, p95={res_install_W[2]*100:+.1f}%")
    c2.metric("Refine shortfall — MC median",
              f"{shortfall_W[1]*100:+.1f}%",
              help=f"One-sided. p5={shortfall_W[0]*100:.1f}%, "
                   f"p95={shortfall_W[2]*100:.1f}%")
    c3.metric("Refine overshoot — MC median",
              f"{overshoot_W[1]*100:+.1f}%",
              help=f"Implied unrecorded. p5={overshoot_W[0]*100:.1f}%, "
                   f"p95={overshoot_W[2]*100:.1f}%")

    years = arr["year"]
    band_df = pd.DataFrame({
        "Year":                years.astype(int),
        "REFINE_SEC_F p5":     np.round(cached_for_key["refine_F_p5"], 0).astype(int),
        "REFINE_SEC_F median": np.round(cached_for_key["refine_F_med"], 0).astype(int),
        "REFINE_SEC_F p95":    np.round(cached_for_key["refine_F_p95"], 0).astype(int),
        "MFG_F p5":            np.round(cached_for_key["mfg_F_p5"], 0).astype(int),
        "MFG_F median":        np.round(cached_for_key["mfg_F_med"], 0).astype(int),
        "MFG_F p95":           np.round(cached_for_key["mfg_F_p95"], 0).astype(int),
        "MFG_total p5":        np.round(cached_for_key["mfg_tot_p5"], 0).astype(int),
        "MFG_total median":    np.round(cached_for_key["mfg_tot_med"], 0).astype(int),
        "MFG_total p95":       np.round(cached_for_key["mfg_tot_p95"], 0).astype(int),
    })
    st.dataframe(band_df, hide_index=True, use_container_width=True)
    st.caption(
        "Bands are 5th / median / 95th percentiles across draws. The survivor count "
        "is the most honest summary — it says how often the current (k, τ, φ) "
        "configuration closes BOTH anchors under perturbation, without blending."
    )


# ---------------------------------------------------------------------------
# Main render entry point
# ---------------------------------------------------------------------------

def render_india_v4_tab() -> None:
    """Render the India v4 calculator tab (Phase D)."""
    country = _render_country_selector()
    preset = COUNTRIES[country]

    # Load source data (cached)
    df_csv = _load_raw_csv(preset.inputs_csv)
    segments_default = _load_segments_df(preset.segments_csv)

    st.divider()
    st.markdown("#### 1. Country inputs (editable)")

    beta, gamma, k_main = _render_beta_gamma_inputs(preset)

    col_stock, col_segs = st.columns([3, 2])
    with col_stock:
        user_stock = _render_stock_editor(df_csv)
    with col_segs:
        user_segments = _render_segments_editor(segments_default)

    # τ override panel (sits below the segments editor, reads its harmonic mean)
    if not user_segments.empty and np.isclose(
        float(user_segments["stock_share"].sum()), 1.0, atol=1e-6
    ):
        tau_seg = _compute_tau_eff(user_segments)
        tau_main = _render_tau_override(tau_seg)
    else:
        tau_main = float("nan")
        tau_seg  = float("nan")

    st.markdown("")
    _render_readonly_data_panel(df_csv, preset.fit_window)

    st.divider()
    st.markdown("#### 2. Process efficiencies and formal shares")
    etas = _render_efficiency_panel()
    phi = _render_phi_inputs()

    st.divider()
    st.markdown("#### 3. Compute the point estimate")

    # Validate inputs before allowing Compute
    inputs_valid = True
    if len(user_stock) < 3:
        inputs_valid = False
    if user_segments.empty or not np.isclose(float(user_segments["stock_share"].sum()), 1.0, atol=1e-6):
        inputs_valid = False

    compute_btn = st.button(
        "Compute (point estimate at k = 1, reference split)",
        disabled=not inputs_valid, type="primary", key="v4_compute_btn",
    )
    if compute_btn:
        st.session_state["v4_initialized"] = True
    if not st.session_state.get("v4_initialized", False):
        st.info("Press **Compute** to render the residuals, flag, flow table, and curves.")
        return

    # Build smoothed arr at the user's stock + USGS + BACI
    try:
        arr = _build_arr_from_user_inputs(df_csv, user_stock, preset.fit_window)
    except Exception as e:
        st.error(f"Could not build the chain inputs: {e}")
        return

    g_val = _compute_g_from_stock(user_stock)

    # Point estimate at the user's (k, τ, φ) — single canonical state.
    out_point = forward_parallel_chain(
        arr,
        k_stock=k_main, phi=phi,
        beta=beta, gamma=gamma,
        g=g_val, tau=tau_main,
        **{k_: v for k_, v in etas.items() if k_ in V5_ETA_DEFAULTS},
    )

    st.markdown("##### Calculation walkthrough — parallel formal/informal chain (every step)")
    st.caption(
        f"Each expander shows one equation and its per-year inputs and outputs at "
        f"k = {k_main:.2f}, τ = {tau_main:.2f} yr, and the user's φ vector. "
        "Both lanes appear at every stage. Implied informal→formal crossovers "
        "are reported at each boundary."
    )
    _render_calculation_walkthrough(
        out_point, arr,
        g=g_val, tau=tau_main,
        beta=beta, gamma=gamma, etas=etas,
        phi=phi, k_used=k_main,
    )

    st.divider()
    st.markdown(f"#### 4. Two residuals + indicators (k = {k_main:.2f}, τ = {tau_main:.2f})")
    _render_residuals_and_flags(
        out_point, arr,
        beta=beta, gamma=gamma, etas=etas,
        g=g_val, tau=tau_main, k_used=k_main,
    )

    st.divider()
    st.markdown(
        f"#### 5. k/τ ridge — install ceiling vs USGS refine floor "
        f"(reading from §1: k = {k_main:.2f}, τ = {tau_main:.2f})"
    )
    _render_kt_ridge_static(
        arr, g_val, gamma, beta, phi=phi, etas=etas,
        k_main=k_main, tau_main=tau_main,
    )
    _render_flow_table_live(
        arr, g_val, gamma, beta,
        phi=phi, etas=etas, k_sel=k_main, tau_sel=tau_main,
    )

    st.divider()
    st.markdown("#### 6. φ ↔ η_I closed-form curves — one per stage (live)")
    _render_phi_eta_curves(
        arr, k=k_main, g=g_val, tau=tau_main,
        gamma=gamma, beta=beta, etas=etas, phi=phi,
    )

    st.divider()
    st.markdown(f"#### 7. Feasibility-region map (with (k, τ) = ({k_main:.2f}, {tau_main:.2f}) marker)")
    _render_feasibility_map(
        arr, g=g_val, gamma=gamma, beta=beta, etas=etas,
        k_main=k_main, tau_main=tau_main,
    )

    st.divider()
    st.markdown("#### 8. Monte Carlo (opt-in)")
    _render_monte_carlo(
        df_csv=df_csv, user_stock=user_stock, segments_df=user_segments,
        fit_window=preset.fit_window,
        beta=beta, gamma=gamma, phi=phi,
        etas=etas, k_sel=k_main, tau_sel=tau_main, arr=arr,
    )

    with st.expander("Reading the dashboard"):
        st.markdown(
            "- The **two residuals are always shown separately**. There is no blended "
            "fit score by design.\n"
            "- **(b)** = an input is impossible (install ceiling exceeded). **(c)** = a "
            "surplus needs explaining. They can fire together; that is the anchor "
            "disagreement, not a bug.\n"
            "- The **k/τ ridge** shows the structural degeneracy: only k/τ is "
            "identified, not k and τ separately. The slider explores the ridge.\n"
            "- The **two φ curves** are profiled stage-by-stage. They do NOT close on "
            "each other; separating formal vs informal at each stage requires "
            "facility-level data (e.g. the Battery Index).\n"
            "- **Monte Carlo** is parametric only. It does not capture the k/τ or "
            "φ ambiguities — those are the curves above."
        )
